#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:36:54 2020

@author: asabater
"""


import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"] = "0";  

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def get_session(gpu_fraction=0.90):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())

from eyolo import EYOLO, load_yolo_model_raw

import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../..')
from roi_nn import load_branch_body



model_folder = './pretrained_models/ILSVRC/1203_1758_model_8/'
train_params = json.load(open(model_folder + 'train_params.json', 'r'))
path_weights = model_folder + 'weights/weights.h5'


image_size = (512,512)
score = 0.005
iou_thr = 0.5
max_boxes = 20

scores_vector = True
add_appearance = True
repp = True


if add_appearance:
	path_roi_model = model_folder + 'embedding_model/'
	path_roi_model_params = json.load(open(path_roi_model+'train_params.json', 'r'))
	downsample_rate = path_roi_model_params['downsample_rate']
	branch_model = load_branch_body(path_roi_model)
else: downsample_rate = None



yolo, train_params = load_yolo_model_raw(model_folder, path_weights, image_size, scores_vector, 
										  downsample_rate, score=0.005, iou_thr=0.5, max_boxes=20)



# %%

import cv2
from PIL import Image


def video_iterator(video_file):
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



annotations_file = '../../data_annotations/annotations_val_skms-1_mvl2.txt'
path_dataset = '/home/asabater/projects/ILSVRC2015/Data/VID/'


# iterator = video_iterator('./test_images/video_1.mp4')
iterator = annotations_iterator(annotations_file, path_dataset)


preds_total = {}
preds_video = {}
last_video = ''
for img, image_id, vid in iterator:
	
	if repp and last_video != vid and last_video != '':
		print('** Video finished, applying post-processing')
# 		preds_video = repp(preds_video)
		preds_total[last_video] = preds_video
		preds_video = {}
	
	last_video = vid
	
	img_size = img.size
	ih, iw = img_size[::-1]
	width_diff = max(0, (ih-iw)//2)
	height_diff = max(0, (iw-ih)//2)
	
	if add_appearance:
		h = w = image_size[0] // downsample_rate
		scale = min(w/iw, h/ih)
		nw, nh = int(iw*scale), int(ih*scale)
		dx, dy = (w-nw)//2, (h-nh)//2
		
	preds = yolo.get_prediction(img)
	preds_frame = []
	for i in range(len(preds[0])):
		y_min, x_min, y_max, x_max = preds[0][i]
		y_min, x_min = max(0, y_min), max(0, x_min)
		y_max, x_max = min(img_size[1], y_max), min(img_size[0], x_max)
		width, height = x_max - x_min, y_max - y_min
		if width <= 0 or height <= 0: continue
	
		bbox_center = [ (x_min + width_diff + width/2)/max(iw,ih),
						  (y_min + height_diff + height/2)/max(iw,ih)]
		
		pred = { 'image_id': image_id, 'bbox': [ x_min, y_min, width, height ], 'bbox_center': bbox_center }
		
		if add_appearance:
			roi_x_min, roi_y_min = dx + x_min*scale, dy + y_min*scale
			roi_width = width*scale; roi_height = height*scale
			roi_width = max(1., roi_width); roi_height = max(1., roi_height)
			emb = branch_model.predict([preds[2][0], np.array([[[roi_x_min, roi_y_min, roi_width, roi_height]]])])[0]
			pred['emb'] = emb
			
		if scores_vector: pred['scores'] = preds[1][i]
		else: 
			pred['score'] = float(preds[1][i])
			pred['category_id']: int(preds[2][i])
		

		preds_frame.append(pred)
		
	preds_video[image_id] = preds_frame


if repp:
	print('** Video finished, applying post-processing')
# 	preds_video = repp(preds_video)
	preds_total[last_video] = preds_video
	del preds_video






