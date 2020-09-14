#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:44:09 2020

@author: asabater
"""

import _init_paths


import cv2
import argparse
import os
import sys
import time
import logging
from config.config import config, update_config

import argparse
import pprint
import logging
import time
import os
import numpy as np
import mxnet as mx
from tqdm import tqdm
import pickle

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval, pred_eval_multiprocess
from utils.load_model import load_param



def transform_selsa_results(preds_filename, res, path_dataset):
    preds_frame = res[0][0][0]
    image_ids = res[0][1]
    
    #imageset_filename = './data/ILSVRC/ImageSets/VID_val_frames.txt'
    with open(path_dataset + 'ImageSets/VID_val_videos_eval.txt', 'r') as f: frame_data = f.read().splitlines()
    
    
    # =============================================================================
    # Store predictions by video along with scores vector to post-processing
    # =============================================================================
    
    imageset_video_frame = {}
    
    for frame in frame_data:
    	frame, i = frame.split()
    	video = '/'.join(frame.split('/')[:-1])
    	if video not in imageset_video_frame: imageset_video_frame[video] = {}
    	
    	imageset_video_frame[video][i] = frame
    
    
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
    
    #store_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/preds_raw_scores_aug_score{}.pckl'.format(min_score)
    store_filename = preds_filename.replace('.pckl', '_repp.pckl').replace('_raw_', '_').replace('/res', '/preds')
    store_file = open(store_filename, 'wb')
    
    for vid, g in tqdm(imageset.groupby('video')):
    	
    	preds_video_frame = {}
    	for ind, r in g.iterrows():
    		
    		preds = r.preds
    #		preds = preds_frame[r.pred_id-1]
    		image_id = r.frame
    		frame_num = image_id.split('/')[-1]
    	
#      		image = Image.open(path_dataset + 'val/' + image_id + '.JPEG')
      		image = Image.open(path_dataset + '/Data/VID/' + image_id + '.JPEG')
    		img_size = image.size
    		ih, iw = img_size[::-1]
    		width_diff = max(0, (ih-iw)//2)
    		height_diff = max(0, (iw-ih)//2)
    		
    		for pred in preds:
    			bbox = pred[:4]
    			scores = pred[4:]
    #			x1,y1,x2,y2 = bbox		
    
    #			y_min, x_min, y_max, x_max = bbox
    			x_min, y_min, x_max, y_max = bbox
    			y_min, x_min = max(0, y_min), max(0, x_min)
    			y_max, x_max = min(img_size[1], y_max), min(img_size[0], x_max)
    			width = x_max - x_min
    			height = y_max - y_min
    			if width <= 0 or height <= 0: continue
    		
    			bbox_center = [ (x_min + width_diff + width/2)/max(iw,ih),
    							  (y_min + height_diff + height/2)/max(iw,ih)]
    			
    #			if scores.max() < min_score: continue
    		
    			pred = {
    					'image_id': image_id,
    					'bbox': [ x_min, y_min, width, height ],
    					'scores': scores,
    					'bbox_center': bbox_center
    					}
    
    			if frame_num in preds_video_frame: preds_video_frame[frame_num].append(pred)
    			else: preds_video_frame[frame_num] = [pred]
    #			print(bbox)
    
    	preds_video_frame = {k: preds_video_frame[k] for k in sorted(preds_video_frame)}
    	pickle.dump(('val/' + vid, preds_video_frame), store_file)
    
    store_file.close()			
    print('Stream predictions stored:', store_filename)
    

def run_fgfa():
    def parse_args():
        parser = argparse.ArgumentParser(description='Test a R-FCN network')
        # general
    #    parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str)
        parser.add_argument('--cfg', help='experiment configure file name', 
                    default='../experiments/fgfa_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml', required=False, type=str)
    
        args, rest = parser.parse_known_args()
        update_config(args.cfg)
    
        # rcnn
        parser.add_argument('--orig_pred', help='True to get and evaluate original predictions. False to get predictions suited for REPP', 
                            action='store_true')
        parser.add_argument('--vis', help='turn on visualization', action='store_true')
        parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
        parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
        parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
        parser.add_argument('--dataset_path', help='path of the Imagenet VID dataset', type=str)
        args = parser.parse_args()
        return args
    
    
    
    args = parse_args()
    #curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join('../external/mxnet', config.MXNET_VERSION))
    
    import mxnet as mx
    from function.test_rcnn import test_rcnn, get_predictor
    from utils.create_logger import create_logger
    
    #ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    ctx = [mx.gpu(int(i)) for i in ['0'] ]
    print args
    
#    config.dataset.test_image_set = 'VID_val_videos_short'
    
    
    config.output_path = '.' + config.output_path
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    
    
    # %%
    
    dataset = config.dataset.dataset
    image_set = config.dataset.test_image_set
    root_path = '../data'    # config.dataset.root_path
    dataset_path = args.dataset_path
    motion_iou_path = './.' + config.dataset.motion_iou_path
    enable_detailed_eval = config.dataset.enable_detailed_eval
    output_path = final_output_path
    
    cfg = config
    
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))
    
    # load symbol and testing data
    
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    
    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)
    
    imdb = eval(dataset)(image_set, root_path, dataset_path, motion_iou_path, result_path=output_path, enable_detailed_eval=enable_detailed_eval)
    roidb = imdb.gt_roidb()
    
    
    # %%
    
    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    roidbs_seg_lens = np.zeros(gpu_num, dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        roidbs_seg_lens[gpu_id] += x['frame_seg_len']
    
    # get test data iter
    test_datas = [TestLoader(x, cfg, batch_size=1, shuffle=args.shuffle, has_rpn=config.TEST.HAS_RPN) for x in roidbs]
    
    
    # %%
    
    prefix = '../model/rfcn_fgfa_flownet_vid'
    epoch = 0
    
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    
    # create predictor
    feat_predictors = [get_predictor(feat_sym, feat_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i in range(gpu_num)]
    aggr_predictors = [get_predictor(aggr_sym, aggr_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i in range(gpu_num)]
    
    
    # %%
    
    
    import pickle
    
    
    key_predictors = feat_predictors
    cur_predictors = aggr_predictors
    vis=args.vis
    ignore_cache=True
    
    thresh = args.thresh
        
        

    t = time.time()
#    store_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/res_{}_{}_raw_aug_score{}_op{}.pckl'.format(
    store_filename = '../predictions/res_{}_{}_raw_score{}_op{}.pckl'.format(
            config.dataset.test_image_set, args.cfg.split('/')[-1][:-5], thresh, args.orig_pred)
    print(store_filename)
    if os.path.isfile(store_filename):
        print('Loading results:', store_filename)
        res = pickle.load(open(store_filename, 'rb'))
        
    else:
        res = [pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, 
                         vis, thresh, logger, ignore_cache),]
        print('Time: {:.2f}'.format(time.time()-t))
        
        if not os.path.isdir('./predictions'): os.makedirs('./predictions/')
        pickle.dump(res, open(store_filename, 'wb'))
        
        print(' ** Transform FGFA predictions to REPP format')
    transform_selsa_results(store_filename, res, args.dataset_path)
      
    if args.orig_pred:
        info_str = imdb.evaluate_detections_multiprocess(res)
        print(info_str)
        return info_str    
      
    
    
    
if __name__ == '__main__':
    run_fgfa()







