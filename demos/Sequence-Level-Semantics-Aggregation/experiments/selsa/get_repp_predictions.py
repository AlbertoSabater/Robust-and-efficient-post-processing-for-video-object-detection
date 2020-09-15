#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:17:10 2020

@author: asabater
"""

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
#this_dir = os.path.dirname(__file__)
#sys.path.insert(0, os.path.join(this_dir, '..', '..', 'rcnn_selsa'))
sys.path.insert(0, 'rcnn_selsa')

import _init_paths

import cv2
import argparse
import os
import sys
import time
import logging
from config.config import config, update_config
import pickle

import argparse
import pprint
import logging
import time
import os
import numpy as np
import mxnet as mx

import mxnet as mx
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger
    
from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval, pred_eval_multiprocess
from utils.load_model import load_param
  
from tqdm import tqdm 


def transform_selsa_results(preds_filename, res, path_dataset):
    preds_frame = res[0][0][0]
    image_ids = res[0][1]
    
    with open(path_dataset + 'ImageSets/VID_val_videos_eval.txt', 'r') as f: frame_data = f.read().splitlines()
    
    
    # =============================================================================
    # Store predictions by video along with scores vector to post-processing
    # =============================================================================
    
    
    import pandas as pd
    
    imageset = {}
    for frame in frame_data:
    	frame, i = frame.split()
    	imageset[i] = frame
    
    frame_image_ids = [ imageset[str(i)] for i in image_ids ]
    video_image_ids = [ '/'.join(imageset[str(i)].split('/')[:-1]) for i in image_ids ]
    imageset = pd.DataFrame({'pred_id': image_ids, 'frame': frame_image_ids, 
    						 'video': video_image_ids, 'preds': preds_frame})
    
    
    from PIL import Image
    
    store_filename = preds_filename.replace('.pckl', '_repp.pckl').replace('_raw_', '_').replace('/res', '/preds')
    store_file = open(store_filename, 'wb')
    
    for vid, g in tqdm(imageset.groupby('video')):
    	
    	preds_video_frame = {}
    	for ind, r in g.iterrows():
    		
    		preds = r.preds
    		image_id = r.frame
    		frame_num = image_id.split('/')[-1]
    	
      		image = Image.open(path_dataset + '/Data/VID/' + image_id + '.JPEG')
    		img_size = image.size
    		ih, iw = img_size[::-1]
    		width_diff = max(0, (ih-iw)//2)
    		height_diff = max(0, (iw-ih)//2)
    		
    		for pred in preds:
    			bbox = pred[:4]
    			scores = pred[4:]
    
    			x_min, y_min, x_max, y_max = bbox
    			y_min, x_min = max(0, y_min), max(0, x_min)
    			y_max, x_max = min(img_size[1], y_max), min(img_size[0], x_max)
    			width = x_max - x_min
    			height = y_max - y_min
    			if width <= 0 or height <= 0: continue
    		
    			bbox_center = [ (x_min + width_diff + width/2)/max(iw,ih),
    							  (y_min + height_diff + height/2)/max(iw,ih)]
    			
    		
    			pred = {
    					'image_id': image_id,
    					'bbox': [ x_min, y_min, width, height ],
    					'scores': scores,
    					'bbox_center': bbox_center
    					}
    
    			if frame_num in preds_video_frame: preds_video_frame[frame_num].append(pred)
    			else: preds_video_frame[frame_num] = [pred]
    
    	preds_video_frame = {k: preds_video_frame[k] for k in sorted(preds_video_frame)}
    	pickle.dump(('val/' + vid, preds_video_frame), store_file)
    
    store_file.close()			
    print('Stream predictions stored:', store_filename)



  
    
def run_selsa():
    def parse_args():
        parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
        parser.add_argument('--cfg', default='experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_aug.yaml',
                            help='experiment configure file name', type=str)
    
        args, rest = parser.parse_known_args()
        update_config(args.cfg)
    
        # rcnn
        parser.add_argument('--orig_pred', help='True to get and evaluate original predictions. False to get predictions suited for REPP', 
                            action='store_true')
        parser.add_argument('--vis', help='turn on visualization', action='store_true')
        parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
        parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
        parser.add_argument('--dataset_path', help='path of the Imagenet VID dataset', type=str)
        parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
        parser.add_argument('--sample-stride', help='sample stride', default=-1, type=int)
        parser.add_argument('--key-frame-interval', help='key frame interval', default=-1, type=int)
        parser.add_argument('--video-shuffle', help='video shuffle', action='store_true')
        parser.add_argument('--test-pretrained', default='./model/pretrained_model/selsa_rcnn_vid', help='test pretrained model', type=str)
        args = parser.parse_args()
        return args
    
    args = parse_args()
    
    config.gpus = '0'
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    
    print(args)
    
     
    logger, final_output_path, tb_log_path = create_logger(config.output_path, config.log_path, args.cfg,
                                                               config.dataset.test_image_set)
    
    trained_model = os.path.join(final_output_path, '..', '_'.join(
            [iset for iset in config.dataset.image_set.split('+')]),
                                     config.TRAIN.model_prefix)
     
    test_epoch = config.TEST.test_epoch
    if args.test_pretrained:
        trained_model = args.test_pretrained
        test_epoch = 0
    
    cfg = config
    cfg_path = args.cfg
    thresh = args.thresh
    orig_pred = args.orig_pred
    
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))
    
    # load symbol and testing data
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    
    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)
    
    dataset = config.dataset.dataset
    image_set = config.dataset.test_image_set
    root_path = config.dataset.root_path
    dataset_path = args.dataset_path                  # config.dataset.dataset_path
    motion_iou_path = config.dataset.motion_iou_path
    output_path = None
    enable_detailed_eval = True
    
    print('='*60)
    print(image_set, root_path, dataset_path, motion_iou_path, output_path,
                         enable_detailed_eval)
    imdb = eval(dataset)(image_set, root_path, dataset_path, motion_iou_path, result_path=output_path,
                         enable_detailed_eval=enable_detailed_eval)
    roidb = imdb.gt_roidb()
    
    
    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    roidbs_seg_lens = np.zeros(gpu_num, dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        if 'frame_seg_len' in x:
            roidbs_seg_lens[gpu_id] += x['frame_seg_len']
        elif 'video_len' in x:
            roidbs_seg_lens[gpu_id] += x['video_len']
       
    # get test data iter
    test_datas = [TestLoader(x, cfg, batch_size=1, shuffle=args.shuffle, video_shuffle=cfg.TEST.video_shuffle,
                             has_rpn=config.TEST.HAS_RPN) for x in roidbs]
        
     
    from function.test_rcnn import get_predictor
    
    # load model
    print('load param from', trained_model, test_epoch)
    arg_params, aux_params = load_param(trained_model, test_epoch, process=True)
    
    # create predictor
    feat_predictors = [get_predictor(feat_sym, feat_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]])
                       for i in range(gpu_num)]
    aggr_predictors = [get_predictor(aggr_sym, aggr_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]])
                       for i in range(gpu_num)]
               
    
    
    key_predictors = feat_predictors
    cur_predictors = aggr_predictors
    vis=False
    logger=None
    ignore_cache=True
    

        
    t = time.time()
    store_filename = './predictions/res_{}_{}_raw_score{}_op{}.pckl'.format(
            config.dataset.test_image_set, cfg_path.split('/')[-1][:-5], thresh, orig_pred)
    print(store_filename)
    if os.path.isfile(store_filename):
        print('Loading results:', store_filename)
        res = pickle.load(open(store_filename, 'rb'))
        
    else:
        res = [pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, 
                         orig_pred,            
                         vis, thresh, logger, ignore_cache), ]
        print('Time: {:.2f}'.format(time.time()-t))
        
        if not os.path.isdir('./predictions'): os.makedirs('./predictions/')
        pickle.dump(res, open(store_filename, 'wb'))
        
        print(' ** Transform SELSA predictions to REPP format')
    transform_selsa_results(store_filename, res, args.dataset_path)
    
    if orig_pred:
        info_str = imdb.evaluate_detections_multiprocess(res)
        print(info_str)
        return info_str
        
    
    
    
if __name__ == '__main__':
    run_selsa()

