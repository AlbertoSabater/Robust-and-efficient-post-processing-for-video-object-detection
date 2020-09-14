"""
Created on Thu Feb  6 17:08:16 2020

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
  
  
    
#orig_pred = True
#thresh=1e-3
def run_selsa(orig_pred, thresh, cfg_path):
    def parse_args():
        parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
        # general
        parser.add_argument('--cfg', default=cfg_path,
    #    parser.add_argument('--cfg', default='cfgs/resnet_v1_101_rcnn_selsa_aug.yaml',
                            help='experiment configure file name', type=str)
    
        args, rest = parser.parse_known_args()
        update_config(args.cfg)
    
        # rcnn
        parser.add_argument('--vis', help='turn on visualization', action='store_true')
        parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
        parser.add_argument('--thresh', help='valid detection threshold', default=5e-3, type=float)
        parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
        parser.add_argument('--sample-stride', help='sample stride', default=-1, type=int)
        parser.add_argument('--key-frame-interval', help='key frame interval', default=-1, type=int)
        parser.add_argument('--video-shuffle', help='video shuffle', action='store_true')
        parser.add_argument('--test-pretrained', default='./model/pretrained_model/selsa_rcnn_vid', help='test pretrained model', type=str)
        args = parser.parse_args()
        return args
    
    args = parse_args()
    #curr_path = os.path.abspath(os.path.dirname(__file__))
    #sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))
    #sys.path.insert(0, os.path.join(curr_path, '../..'))
    
    
    config.gpus = '0'
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    
    print(args)
    
    # if args.sample_stride != -1:
    #     config.TEST.sample_stride = args.sample_stride
    # if args.key_frame_interval != -1:
    #     config.TEST.KEY_FRAME_INTERVAL = args.key_frame_interval
    # if args.video_shuffle:
    #     config.TEST.video_shuffle = args.video_shuffle
     
    logger, final_output_path, tb_log_path = create_logger(config.output_path, config.log_path, args.cfg,
                                                               config.dataset.test_image_set)
     
    
    trained_model = os.path.join(final_output_path, '..', '_'.join(
            [iset for iset in config.dataset.image_set.split('+')]),
                                     config.TRAIN.model_prefix)
     
    test_epoch = config.TEST.test_epoch
    if args.test_pretrained:
        trained_model = args.test_pretrained
        test_epoch = 0
    
    
    
    ## %%
    

    
    cfg = config
    
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))
    
    # load symbol and testing data
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    
    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)
    
    
    #config.dataset.test_image_set = 'VID_val_videos_short'
    
    dataset = config.dataset.dataset
    image_set = config.dataset.test_image_set
    root_path = config.dataset.root_path
    dataset_path = '/home/asabater/projects/ILSVRC2015/'                  # config.dataset.dataset_path
    motion_iou_path = config.dataset.motion_iou_path
    output_path = None
    enable_detailed_eval = True
    
    imdb = eval(dataset)(image_set, root_path, dataset_path, motion_iou_path, result_path=output_path,
                         enable_detailed_eval=enable_detailed_eval)
    roidb = imdb.gt_roidb()
    
    
    ## %%
    
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
        
        
    ## %%         
     
    from function.test_rcnn import get_predictor
    
    # load model
    print('load param from', trained_model, test_epoch)
    arg_params, aux_params = load_param(trained_model, test_epoch, process=True)
    
    # create predictor
    feat_predictors = [get_predictor(feat_sym, feat_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]])
                       for i in range(gpu_num)]
    aggr_predictors = [get_predictor(aggr_sym, aggr_sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]])
                       for i in range(gpu_num)]
               
    
    ## %%
    
    key_predictors = feat_predictors
    cur_predictors = aggr_predictors
    vis=False
    #thresh=0.005
    # thresh=0
    logger=None
    ignore_cache=True
    
    # cfg.TEST.max_per_image = 2
    
    
    
    # ====
    # ALL
    # =================================================
    # motion [0.0 1.0], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 0.8266
    # =================================================
    # motion [0.0 0.7], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 0.6699
    # =================================================
    # motion [0.7 0.9], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 0.8138
    # =================================================
    # motion [0.9 1.0], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 0.8794
    # * Time: 50774.21
    # * Time: 50837.79
    
    # ====
    # SHORT
    # =================================================
    # motion [0.0 1.0], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 1.0000
    # =================================================
    # motion [0.0 0.7], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 1.0000
    # =================================================
    # motion [0.7 0.9], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 1.0000
    # =================================================
    # motion [0.9 1.0], area [0.0 0.0 100000.0 100000.0]
    # Mean AP@0.5 = 1.0000
        
    t = time.time()
    store_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/new_res_{}_{}_raw_aug_score{}_op{}.pckl'.format(
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
        pickle.dump(res, open(store_filename, 'wb'))
    
    if orig_pred:
        info_str = imdb.evaluate_detections_multiprocess(res)
        print(info_str)
        return info_str
        

# %%

#orig_pred = True
#thresh=1e-3
        
# TOTAL     FAST    MED     SLOW

# 0.8266    0.6706  0.8135  0.8795
stats1 = run_selsa(orig_pred = True, thresh = 1e-3, cfg_path = 'experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_noaug.yaml')

# 0.8269    0.6717  0.8134  0.8802
stats2 = run_selsa(orig_pred = True, thresh = 1e-3, cfg_path = 'experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_aug.yaml')

stats3 = run_selsa(orig_pred = False, thresh = 1e-3, cfg_path = 'experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_noaug.yaml')
stats4 = run_selsa(orig_pred = False, thresh = 1e-3, cfg_path = 'experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_aug.yaml')



# %%

import pickle

pickle.dump(res, open('/mnt/hdd/egocentric_results/ilsvrc/SELSA/res_raw_aug.pckl', 'wb'))


# %%

res = pickle.load(open('/mnt/hdd/egocentric_results/ilsvrc/SELSA/res_raw_aug.pckl', 'rb'))

# Group by video
# Store pickle stream

preds_video = { vid: [] for vid in videos}

preds, frame_ids = res

for pred in res:
    pass


