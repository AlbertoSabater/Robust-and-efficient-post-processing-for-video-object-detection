#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:56:41 2019

@author: asabater
"""

import os

os.chdir('/home/asabater/projects/darknet/')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"; 

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import json
from tqdm import tqdm



def convertBack(x, y, w, h):
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
	for detection in detections:
		x, y, w, h = detection[2][0],\
			detection[2][1],\
			detection[2][2],\
			detection[2][3]
		xmin, ymin, xmax, ymax = convertBack(
			float(x), float(y), float(w), float(h))
		pt1 = (xmin, ymin)
		pt2 = (xmax, ymax)
		cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
		cv2.putText(img,
					detection[0].decode() +
					" [" + str(round(detection[1] * 100, 2)) + "]",
					(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					[0, 255, 0], 2)
	return img



def predict_and_store_from_annotations_darknet(configPath, weightPath, metaPath, annotations_file, score, nms, load=False):

	preds_filename = weightPath.replace('.weights', '.json').replace('backup/', 'results/preds_score{}_nms{}_'.format(score, nms))
	if os.path.isfile(preds_filename): 
		print(' * Loading:', preds_filename)
		if load: return json.load(open(preds_filename, 'r')), preds_filename
		else: return None, preds_filename
	
	netMain = darknet.load_net_custom(configPath.encode(
		"ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	metaMain = darknet.load_meta(metaPath.encode("ascii"))
#	try:
#		with open(metaPath) as metaFH:
#			metaContents = metaFH.read()
#			import re
#			match = re.search("names *= *(.*)$", metaContents,
#							  re.IGNORECASE | re.MULTILINE)
#			if match:
#				result = match.group(1)
#			else:
#				result = None
#			try:
#				if os.path.exists(result):
#					with open(result) as namesFH:
#						namesList = namesFH.read().strip().split("\n")
#						altNames = [x.strip() for x in namesList]
#			except TypeError:
#				print('except 1')
#				pass
#	except Exception:
#		print('except 2')		
#		pass
		
	
	
	with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
	
	darknet_image = darknet.make_image(darknet.network_width(netMain),
										darknet.network_height(netMain),3)
	
	
	class_names = [ metaMain.names[i] for i in range(metaMain.classes) ]
	
	detections = []
	preds = []
	for i, img in tqdm(enumerate(annotations), total=len(annotations)):
	#	print(i, 'Loading', img)
		frame_read = cv2.imread(img)
		frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,
								   (darknet.network_width(netMain),
									darknet.network_height(netMain)),
								   interpolation=cv2.INTER_LINEAR)
		height, width, _ = frame_read.shape
		scale_h, scale_w = height/darknet.network_height(netMain), width/darknet.network_width(netMain), 
		   
	#	print(i, frame_rgb.shape, frame_resized.shape)
	
		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
	#	print(i, 'darknet_image created')
		dets = darknet.detect_image(netMain, metaMain, darknet_image, thresh=score, 
								hier_thresh=score, nms=nms, debug=False)
		
		for cat, s, bbox in dets:
			x,y,w,h = list(bbox)
			xmin, ymin, xmax, ymax = convertBack(x*scale_w,y*scale_h,w*scale_w,h*scale_h)
			preds.append({
							'image_id': '.'.join('/'.join(img.split('/')[-3:]).split('.')[:-1]),
							'category_id': class_names.index(cat),
							'bbox': [ xmin, ymin, xmax-xmin, ymax-ymin ],
							'score': s
						})
	
		detections.append(dets)
	#	print(i, 'Detected')
	
	#	image = cvDrawBoxes(det, frame_resized)
	#	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#	cv2.imshow('Demo', image)
	#	cv2.waitKey(50)	
		
	json.dump(preds, open(preds_filename, 'w'))
	print(preds_filename, 'stored')

	if load: preds, preds_filename
	else: return None, preds_filename
	
	
# %%

configPath = "/mnt/hdd/egocentric_results/darknet/5_adl_v2_pan_lstm/yolo_v3_spp_lstm.cfg"
weightPath = "/mnt/hdd/egocentric_results/darknet/5_adl_v2_pan_lstm/backup/yolo_v3_spp_lstm_best.weights"
metaPath = "/mnt/hdd/egocentric_results/darknet/5_adl_v2_pan_lstm/adl_v2.data"

#configPath = "test_adl_v2_spp/yolov3-spp_adl_v2_27.cfg"
#weightPath = "/home/asabater/projects/darknet/test_adl_v2_spp/backup/yolov3-spp_adl_v2_27_best.weights"
#metaPath = "test_adl_v2_spp/adl_v2.data"

score = .00005
nms = .45


annotations_file = '/home/asabater/projects/Egocentric-object-detection/dataset_scripts/adl/annotations_adl_val_v2_27_darknet.txt'
_, preds_filename = predict_and_store_from_annotations_darknet(configPath, weightPath, metaPath, annotations_file, score, nms, load=False)

os.chdir('/home/asabater/projects/Egocentric-object-detection/')

import evaluate_model

eval_stats, videos = evaluate_model.get_full_evaluation(annotations_file.replace('_darknet', ''), preds_filename, class_names=None, full=False)
print(eval_stats)


exit

# %%

#weights_dir = '/home/asabater/projects/darknet/test_adl_v2/backup/'
##weights_dir = '/home/asabater/projects/darknet/test_adl_v2_spp/backup/'
#results = []
#
#files = [ f for f in os.listdir(weights_dir) if 'best' not in f and 'last' not in f and 'final' not in f ]
#for weights in files[::8]:
#	weightPath = weights_dir + weights
#
#	os.chdir('/home/asabater/projects/darknet/')
#	annotations_file = '/home/asabater/projects/Egocentric-object-detection/dataset_scripts/adl/annotations_adl_val_v2_27_darknet.txt'
#	_, preds_filename = predict_and_store_from_annotations_darknet(configPath, 
#										weightPath, metaPath, annotations_file, score, nms, load=False)
#	
#	os.chdir('/home/asabater/projects/Egocentric-object-detection/')
#	
#	import evaluate_model
#	eval_stats, videos = evaluate_model.get_full_evaluation(annotations_file.replace('_darknet', ''), 
#														 preds_filename, class_names=None, full=False)
#	
#	print(weights.split('/')[-1], eval_stats['total'][-1])
#	
#	results.append((weightPath, eval_stats['total'][1]))
#
#
## %%
#	
#for w,s in results:
#	print('{} | {}'.format(w.split('_')[-1], s*100))	
