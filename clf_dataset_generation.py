#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:44:51 2020

@author: asabater
"""

import os
import sys
sys.path.append('demos/YOLOv3')
sys.path.append('demos/YOLOv3/keras_yolo3')
sys.path.append('demos/YOLOv3/keras_yolo3/yolo3/')

from roi_data_generator import data_generator_wrapper, get_random_data_cv2

import datetime
from tqdm import tqdm
import pandas as pd
import pickle
from PIL import Image
import sys
import numpy as np
import math
from scipy import signal
from functools import reduce
import json

import copy
import argparse


# =============================================================================
# Script to build a dataset made of features from triplet annotations (Anchor, Positive, Negative)
# Used to train a clasification model for the REPP linking Logistic Regression
# =============================================================================

from repp_utils import get_pair_features



# Input feats: IOU, wA/wN, hA/hN, distasnce between centers, distance between centers / w/h,
# euclidean ROI descriptors distance
# Cannot use scores dot product -> difficult of getting it from gt ROIS
def get_and_store_dataset_aug(path_annotations, path_dataset, input_shape, backbone, branch_model, downsample_rate, random, mode):
    
    store_dir = './data_annotations/triplet_annotations/matching_models_dataset_{}{}.pckl'.format(mode,
                                                                   '_appearance' if branch_model is not None else '')
    print('Store path:', store_dir)

    with open(path_annotations, 'r') as f: annotations = f.read().splitlines()
    
    feat_embs = []
    feat_names = ['center_distances_corrected', 'height_rel', 'iou', 'width_rel']
    if branch_model is not None: feat_names.append('descriptor_dist')
    
    for ann in tqdm(annotations):
        sample_data = get_random_data_cv2(ann, path_dataset, input_shape, downsample_rate, random=random, fix_coords=False)
        imgs = sample_data[:3]
        roi_boxes = copy.deepcopy(sample_data[3:])

        bbox_centers = []
        for i in range(3):
            bbox_centers.append(((roi_boxes[i][0][0] + (roi_boxes[i][0][2]/2)) * downsample_rate / input_shape[0], 
                    (roi_boxes[i][0][1] + (roi_boxes[i][0][3]/2)) * downsample_rate / input_shape[0]))
            
            
        if branch_model is not None:        # Add appearance features
            embs = []
            for img, roi_box in zip(imgs, roi_boxes):
                roi_box[:, 2] = max(1., roi_box[:, 2]); roi_box[:, 3] = max(1., roi_box[:, 3])
                pred_feat_map = backbone.predict(np.expand_dims(img, axis=0))
                pred_bbox_embs = branch_model.predict([pred_feat_map, np.expand_dims(roi_box, axis=0)])[0]
    
                embs.append(pred_bbox_embs)
        
            # Input feats: IOU, wA/wN, hA/hN, distasnce between centers, distance between centers / w/h,
                # euclidean ROI descriptors distance
                # Cannot use scores dot product -> difficult of getting it from gt ROIS
            feat_embs.append(
                    [ get_pair_features({'bbox': sample_data[3][0].copy(), 'emb': embs[0],
                                          'bbox_center': bbox_centers[0]},
                                         {'bbox': sample_data[4][0].copy(), 'emb': embs[1],
                                          'bbox_center': bbox_centers[1]}, feat_names), 
                      get_pair_features({'bbox': sample_data[3][0].copy(), 'emb': embs[0],
                                          'bbox_center': bbox_centers[0]},
                                         {'bbox': sample_data[5][0].copy(), 'emb': embs[2],
                                          'bbox_center': bbox_centers[2]}, feat_names) ])
        else:                           # No add appearance features
            feat_embs.append(
                    [ get_pair_features({'bbox': sample_data[3][0].copy(),
                                          'bbox_center': bbox_centers[0]},
                                         {'bbox': sample_data[4][0].copy(),
                                          'bbox_center': bbox_centers[1]}, feat_names), 
                      get_pair_features({'bbox': sample_data[3][0].copy(),
                                          'bbox_center': bbox_centers[0]},
                                         {'bbox': sample_data[5][0].copy(),
                                          'bbox_center': bbox_centers[2]}, feat_names) ])            
        
    
    X,Y = [], []
    X += [ A for A,N in feat_embs ]; Y += [1]*len(feat_embs)
    X += [ N for A,N in feat_embs ]; Y += [0]*len(feat_embs)
    X = pd.DataFrame(X); np.array(Y)

    if not os.path.isdir('/'.join(store_dir.split('/')[:-1])):
        os.makedirs('/'.join(store_dir.split('/')[:-1]))
    pickle.dump([X,Y], open(store_dir, 'wb'))    
    
    print(' * Stored:', store_dir)




def main():
    parser = argparse.ArgumentParser(description='Creates the dataset to train the matching classifier from triplet annotations')
    parser.add_argument('--downsample_rate', help='number of times that the input is downsampled', default=16, type=int)
    parser.add_argument('--image_size', help='input image size', default=512, type=int)
    parser.add_argument('--no_random', help='true too apply data augmentation', action='store_false')
    parser.add_argument('--add_appearance', help='true add appearance features', action='store_true')
    parser.add_argument('--path_dataset', help='path of the Imagenet VID dataset', type=str)
    args = parser.parse_args()
    
    
    downsample_rate = args.downsample_rate
    image_size = (args.image_size,args.image_size)
    rndm = args.no_random
    add_appearace_similarity = args.add_appearance
    path_dataset = args.path_dataset
    
    print(downsample_rate, image_size, rndm, add_appearace_similarity, path_dataset)
    
    if add_appearace_similarity:
        
        import roi_nn
        import train_utils

        base_path_model = './demos/YOLOv3/pretrained_models/ILSVRC/1203_1758_model_8/'
        path_roi_model = base_path_model + 'embedding_model/'
        path_weights = train_utils.get_best_weights(base_path_model)
        backbone = roi_nn.get_backbone(path_weights, downsample_rate = downsample_rate)
        branch_model = roi_nn.load_branch_body(path_roi_model)
    else: backbone, branch_model = None, None
    
    np.random.seed(0)
    
    path_annotations = './data_annotations/triplet_annotations/triplet_annotations_train.txt'
    get_and_store_dataset_aug(path_annotations, path_dataset, image_size, backbone, branch_model, downsample_rate, rndm, 'train')
            
    path_annotations = './data_annotations/triplet_annotations/triplet_annotations_val.txt'
    get_and_store_dataset_aug(path_annotations, path_dataset, image_size, backbone, branch_model, downsample_rate, rndm, 'val')


if __name__ == '__main__': main()

