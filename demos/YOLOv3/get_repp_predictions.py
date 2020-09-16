#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:17:03 2020

@author: asabater
"""

# =============================================================================
# Script to make YOLO predictions either form a video or annotations file
# =============================================================================


import os

import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle
import argparse

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')
import keras_yolo3.train as ktrain

sys.path.append('../..')
from roi_nn import load_branch_body
from eyolo import load_yolo_model_raw


def video_iterator(video_file):
    import cv2
    vid = cv2.VideoCapture(video_file)
    num_frame = 0
    while True:
        ret, frame = vid.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        num_frame += 1
        yield image, '{:06d}'.format(num_frame), False
        

def annotations_iterator(annotations_file, path_dataset):
    with open(annotations_file, 'r') as f: annotations = sorted(f.read().splitlines())
    for ann in tqdm(annotations):
        image_id = ann.split()[0]
        image = Image.open(path_dataset + image_id)
        image_id = '.'.join(image_id.split('.')[:-1])
        vid = '/'.join(image_id.split('/')[:-1])
        yield image, image_id, vid



# Calculates YOLOv3 predictions given a data iterator
# Each video is dumped after its processing to a common pickle file. Pickle stream
# If an appearance embedding model is provided and add_appearance is set, 
#   appearance vectors are stored with the predictions
def get_scores_predictions_and_embeddings(store_filename, base_model, branch_model, add_appearance, repp_format):

    # Open pickle stream file
    file_writter = open(store_filename, 'wb')

    preds_video = {}
    last_video = ''
    for img, image_id, vid in iterator:
        
        frame_id = image_id.split('/')[-1]
        
        # Video finished. Pickle dumping
        if last_video != vid and last_video != '':
            pickle.dump((last_video, preds_video), file_writter)
            preds_video = {}
            last_video = vid

        
        last_video = vid
        
        img_size = img.size
        ih, iw = img_size[::-1]
        width_diff = max(0, (ih-iw)//2)
        height_diff = max(0, (iw-ih)//2)
        
        # Compute image values for the RoI extraction from the feature maps
        if add_appearance:
            h = w = image_size[0] // downsample_rate
            scale = min(w/iw, h/ih)
            nw, nh = int(iw*scale), int(ih*scale)
            dx, dy = (w-nw)//2, (h-nh)//2
        
        # Get YOLO predictions
        preds = base_model.get_prediction(img)
        
        preds_frame = []
        for i in range(len(preds[0])):
            
            # Compute bbox center
            y_min, x_min, y_max, x_max = preds[0][i]
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(img_size[1], y_max), min(img_size[0], x_max)
            width, height = x_max - x_min, y_max - y_min
            if width <= 0 or height <= 0: continue
            bbox_center = [ (x_min + width_diff + width/2)/max(iw,ih),
                              (y_min + height_diff + height/2)/max(iw,ih)]
            
            # Initialize predictions
            pred = { 'image_id': image_id, 'bbox': [ x_min, y_min, width, height ], 'bbox_center': bbox_center }
            
            # Compute the appearance embedding vectors
            if add_appearance:
                roi_x_min, roi_y_min = dx + x_min*scale, dy + y_min*scale
                roi_width = width*scale; roi_height = height*scale
                roi_width = max(1., roi_width); roi_height = max(1., roi_height)
                emb = branch_model.predict([preds[2][0], np.array([[[roi_x_min, roi_y_min, roi_width, roi_height]]])])[0]
                pred['emb'] = emb
                
            # Scores are given by a vector
            if repp_format: 
                pred['scores'] = preds[1][i]
            # Single score and category provided
            else: 
                pred['score'] = float(preds[1][i])
                pred['category_id']: int(preds[2][i])
            
            preds_frame.append(pred)
            
        preds_video[frame_id] = preds_frame

    # Video finished. Pickle dumping
    pickle.dump((last_video, preds_video), file_writter)
    preds_video = {}
    
    file_writter.close()




if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Perform YOLOv3 predictions either from an annotations file or a video')
    parser.add_argument('--yolo_path', required=True, type=str, help='path to the trained YOLO folder')
    parser.add_argument('--score', type=float, default=0.005, help='therhold to filter out low-scoring predictions')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--max_boxes', type=int, default=20, help='maximum boxes per image')
    parser.add_argument('--repp_format', action='store_true', help='store the predictions in REPP format (with an array of class confidence scores instead of one predictions per class)')
    parser.add_argument('--add_appearance', action='store_true', help='compute one appearance embedding for each detection')
    parser.add_argument('--from_video', type=str, required=False, help='path of the video to perform predictions')
    parser.add_argument('--from_annotations', type=str, required=False, help='path to the annotations file to perform predictions')
    parser.add_argument('--dataset_path', type=str, required=False, help='path to the dataset images. Required when making predictions form annotations')
    
    args = parser.parse_args()
    
    
    assert not (args.from_video is not None and args.from_annotations is not None), 'Only one data source (video or annotations) must be specified'
    assert args.from_video is not None or args.from_annotations is not None, 'One data source (video or annotations) must be specified'
    assert not (args.from_annotations is not None and args.dataset_path is None), 'Dataset path of the annotations data must be specified'
    
    
    # Load YOLO settings
    train_params = json.load(open(args.yolo_path + 'train_params.json', 'r'))
    path_weights = args.yolo_path + 'weights/weights.h5'
    image_size = train_params['input_shape']
    
    
    if not args.repp_format and add_appearance: 
        print(' * REPP format not specified. Suppressing appearance computation')
        args.add_appearance = False
    
    # Load appearance embeddings model
    if args.add_appearance:
        path_roi_model = args.yolo_path + 'embedding_model/'
        path_roi_model_params = json.load(open(path_roi_model+'train_params.json', 'r'))
        downsample_rate = path_roi_model_params['downsample_rate']
        branch_model = load_branch_body(path_roi_model)
    else: downsample_rate, branch_model = None, None
    
    
    # Load YOLO model
    base_model, _ = load_yolo_model_raw(args.yolo_path, path_weights, image_size, args.repp_format, 
                                          downsample_rate, args.score, args.iou_thr, args.max_boxes)
    

    
    if args.from_video is not None:
        iterator = video_iterator(args.from_video)
        store_filename = './predictions/preds{}{}_{}.pckl'.format(
                            '_repp' if args.repp_format else '',
                            '_app' if args.add_appearance else '',
                            args.from_video.split('/')[-1].split('.')[0])
        # raise ValueError('Not implemented')
    else:
        iterator = annotations_iterator(args.from_annotations, args.dataset_path)    
        store_filename = './predictions/preds{}{}_{}.pckl'.format(
                            '_repp' if args.repp_format else '',
                            '_app' if args.add_appearance else '',
                            args.from_annotations.split('/')[-1].split('.')[0])
    
    
    get_scores_predictions_and_embeddings(store_filename, base_model, branch_model, args.add_appearance, args.repp_format)
    
    print('Predictions stored:', store_filename)
