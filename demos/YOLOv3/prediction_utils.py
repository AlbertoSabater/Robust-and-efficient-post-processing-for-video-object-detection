#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:24:20 2019

@author: asabater
"""

import os

from eyolo import EYOLO
from PIL import Image
from test_post_processing import seq_nms_fgfa

from tqdm import tqdm
import os
import train_utils
import time
import json
import sys
import pandas as pd
import numpy as np
import pickle
import keras



# Stores an json file with the predictions calculated by the given annotations_file
# Uses best weights or last calculated weights depending on best_weights
def predict_and_store_from_annotations(model_folder, train_params, annotations_file, 
									   output_dir, image_size, score, nms_iou, best_weights=True,
									   raw_eval='def', load=False, model_path=None):
	
	if raw_eval == 'raw_scores':
		preds_filename = '{}preds_{}_{}_is{}_score{}_raw_score.json'.format(output_dir, 
					   'bw' if best_weights else 'stage2',
					   annotations_file.split('/')[-1][:-4], 
					   image_size[0], score, nms_iou)
	elif raw_eval == 'raw':
		preds_filename = '{}preds_{}_{}_is{}_score{}_raw.json'.format(output_dir, 
					   'bw' if best_weights else 'stage2',
					   annotations_file.split('/')[-1][:-4], 
					   image_size[0], score)
	elif 'def':
		preds_filename = '{}preds_{}_{}_is{}_score{}_iou{}.json'.format(output_dir, 
					   'bw' if best_weights else 'stage2',
					   annotations_file.split('/')[-1][:-4], 
					   image_size[0], score, nms_iou)
	else: raise ValueError('Unrecognized eval error')
		
	
	if os.path.isfile(preds_filename): 
		print(' * Loading:', preds_filename)
		if load: return json.load(open(preds_filename, 'r')), preds_filename
		else: return None, preds_filename
	
	print(' * Predicting:', preds_filename)
	
	if model_path is None:
		if best_weights:
			model_path = train_utils.get_best_weights(model_folder)
		else:
			model_path = model_folder + 'weights/trained_weights_final.h5'
	
	model = EYOLO(
					model_image_size = image_size,
					model_path = model_path,
					classes_path = train_params['path_classes'],
					anchors_path = train_params['path_anchors'],
					score = score,
					iou = nms_iou,	  # 0.5
					td_len = train_params.get('td_len', None),
					mode = train_params.get('mode', None),
					spp = train_params.get('spp', False),
					raw_eval = raw_eval
				)
	
	
	# Create pred file
	with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
	preds = []
	
	prediction_time = 0
	total = len(annotations)
	total_prediction_time = time.time()
	for ann in tqdm(annotations[:total], total=total, file=sys.stdout):
		img = ann.split()[0]
		
#		image = cv2.imread(train_params['path_dataset'] + img)
#		image = Image.fromarray(image)
		images = [ Image.open(train_params['path_dataset'] + img) for img in img.split(',') ]
		img_size = images[0].size
		max_size = max(img_size[0], img_size[1])
#		images = images[0] if len(images) == 1 else np.stack(images)
		t = time.time()
		boxes, scores, classes = model.get_prediction(images)
		prediction_time += time.time() - t
		
		for i in range(len(boxes)):
			left, bottom, right, top = [ min(int(b), max_size)  for b in boxes[i].tolist() ]
			left, bottom = max(0, left), max(0, bottom)
			right, top = min(img_size[1], right), min(img_size[0], top)
			width = top-bottom
			height = right-left
			preds.append({
							'image_id': '.'.join(img.split('.')[:-1]),
							'category_id': int(classes[i]),
#							'bbox': [ x_min, y_min, width, height ],
							'bbox': [ bottom, left, width, height ],
							'score': float(scores[i]),
						})
	
	print('Total prediction time: ', time.time()-total_prediction_time)
	print('Prediction time: {} secs || {:.2f}'.format(prediction_time, prediction_time/len(annotations)))
	
#	keras.backend.clear_session()
	model.yolo_model = None
	del model.yolo_model
	model = None
	del model
	
#	fps = len(annotations)/(time.time()-t)
	json.dump(preds, open(preds_filename, 'w'))
	
	if load: return preds, preds_filename
	else: return None, preds_filename


def filter_predictions_by_score(preds_filename, filter_out_score, load=False, force=False):

	out_filename = preds_filename.replace(
			[ s for s in preds_filename.split('_') if s.startswith('score') ][0], 'score{}'.format(filter_out_score))

	if force and os.path.isfile(out_filename): os.remove(out_filename)

	if os.path.isfile(out_filename): 
		print(' * Loading:', preds_filename)
		if load: return json.load(open(preds_filename, 'r')), out_filename
		else: return None, out_filename

	preds = json.load(open(preds_filename, 'r'))
	preds = [ p for p in preds if p['score'] >= filter_out_score ]
	
	json.dump(preds, open(out_filename, 'w'))

	if load: return preds, out_filename
	else: return None, out_filename
	
