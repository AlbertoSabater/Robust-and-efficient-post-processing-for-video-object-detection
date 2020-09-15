 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:53:31 2019

@author: asabater
"""

import os

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')

import json
import os
import numpy as np
import pandas as pd
import keras_yolo3.train as ktrain
import train_utils

from tensorboard.backend.event_processing import event_accumulator
import datetime
import time

import cv2
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pyperclip
import matplotlib.pyplot as plt
import time
import prediction_utils


MIN_SCORE = 0.00005


# Returns training time, best train loss and val loss
def get_train_resume(model_folder):
	tb_files = [ model_folder + f for f in os.listdir(model_folder) if f.startswith('events.out.tfevents') ]
	
	train_losses, val_losses = [], []
	times = []
	for tbf in tb_files:
#		print(tbf)
		try:
			ea = event_accumulator.EventAccumulator(tbf).Reload()
			train_losses += [ e.value for e in ea.Scalars('loss') ]
			val_losses += [ e.value for e in ea.Scalars('val_loss') ]
			times += [ e.wall_time for e in ea.Scalars('val_loss') ]
		except: continue
	
	val_loss = min(val_losses)
	train_loss = train_losses[val_losses.index(val_loss) ]
	
	train_init, train_end = min(times), max(times)
	
	train_init = datetime.datetime.fromtimestamp(train_init)
	train_end = datetime.datetime.fromtimestamp(train_end)
	
	train_diff = (train_end - train_init)
	train_diff = '{}d {:05.2f}h'.format(train_diff.days, train_diff.seconds/3600)
	
	return train_diff, train_loss, val_loss


# Perform mAP evaluation given the preddictions file and the groundtruth filename
# Evaluation is performed to all the datset, pero class and per subdataset (if exists) if full = True
def get_full_evaluation(annotations_file, preds_filename, class_names, full, force=False):
	
	eval_filename = preds_filename.replace('preds', 'stats')
	eval_filename_full = eval_filename.replace('.json', '_full.json')
	ann = annotations_file[:-4] + '_coco.json'
#	print(' * Eval:', eval_filename)
	
#	print(' * Annotations:', ann)
#	print(' * Preds:', preds_filename)

	if force:
		if os.path.isfile(eval_filename_full): os.remove(eval_filename_full)
		if os.path.isfile(eval_filename): os.remove(eval_filename)


	if full and os.path.isfile(eval_filename_full):
		# If full exists return full
#		print(' * Loading:', eval_filename_full)
		eval_stats =  json.load(open(eval_filename_full, 'r'))
		videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
		return eval_stats, videos
	elif os.path.isfile(eval_filename):
		# If regular eval exists load id
#		print(' * Loading:', eval_filename)
		eval_stats =  json.load(open(eval_filename, 'r'))
		
		if not full:
			videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
			return eval_stats, videos	  
	else:
		# if no eval has been performed, initialize the dictionary and COCO
		eval_stats = {}

	cocoGt = COCO(ann)
	print(preds_filename)
	cocoDt = cocoGt.loadRes(preds_filename)
	cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')		
	
	if 'total' not in eval_stats:
		print(' * Evaluating total')
		cocoEval.evaluate()
		cocoEval.accumulate()
		cocoEval.summarize()
		eval_stats['total'] = cocoEval.stats.tolist()
		json.dump(eval_stats, open(eval_filename, 'w'))
		
	if not full:
		# if not full return regular eval if not, continue the evaluation
		videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
		return eval_stats, videos  
	
	else:
	
		params = cocoEval.params
		imgIds = params.imgIds
		catIds = params.catIds
		
		# Evaluate each category
		for c in range(len(class_names)):
			print(' * Evaluationg the category |{}|'.format(class_names[c]))
			params.catIds = [c]
			cocoEval.evaluate()
			cocoEval.accumulate(params)
			cocoEval.summarize()
			eval_stats['cat_{}'.format(c)] = cocoEval.stats.tolist()
			params.catIds = catIds
		
		
		# Evaluate each video and each category per video
		videos = sorted(list(set([ i.split('/')[-2] for i in imgIds ])))
		for v in videos:
			print(' * Evaluating the video |{}|'.format(v))
			eval_stats[v] = {}
			v_imgIds = [ i for i in imgIds if v in i ]
			params.imgIds = v_imgIds
			cocoEval.evaluate()
			cocoEval.accumulate(params)
			cocoEval.summarize()
			eval_stats[v]['total'] = cocoEval.stats.tolist()
			
			for c in range(len(class_names)):
				print(' * Evaluationg the category |{}| in the video |{}|'.format(class_names[c], v))
				params.catIds = [c]
				cocoEval.evaluate()
				cocoEval.accumulate(params)
				cocoEval.summarize()
				eval_stats[v]['cat_{}'.format(c)] = cocoEval.stats.tolist()
				params.catIds = catIds
			
			params.imgIds = imgIds
			params.catIds = catIds
			
		json.dump(eval_stats, open(eval_filename_full, 'w'))

	return eval_stats, videos


def get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou):
	result = '{model_folder}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:d}\t{training_time}'.format(
				model_folder = '/'.join(model_folder.split('/')[-2:]), 
				input_shape = train_params['input_shape'],
				annotations = train_params['path_annotations'],
				anchors = train_params['path_anchors'],
				pretraining = train_params['path_weights'],
				frozen_training = train_params['freeze_body'],
				training_time = train_diff
			)
	
	result += '\t{train_loss:.5f}\t{val_loss:.5f}'.format(train_loss=train_loss, val_loss=val_loss).replace('.', ',')
	result += '\t{score:.5f}\t{iou:.2f}'.format(score=score, iou=iou).replace('.', ',')
	
	result += '\t{mAP}\t{mAP50}\t{mAP75}\t{mAPS}\t{mAPM}\t{mAPL}'.format(
				mAP=eval_stats['total'][0]*100, mAP50=eval_stats['total'][1]*100, 
				mAP75=eval_stats['total'][2]*100, mAPS=eval_stats['total'][3]*100, 
				mAPM=eval_stats['total'][4]*100, mAPL=eval_stats['total'][5]*100, 
			).replace('.', ',')
	
	result += '\t{fps:.2f}'.format(fps=fps)

	return result


def get_excel_resume_full(model_folder, train_params, train_loss, val_loss, 
						  eval_stats_train, eval_stats_val, train_diff, best_weights):
	
	if 'egocentric_results' in train_params['path_weights']:
		path_weights = '/'.join(train_params['path_weights'].split('/')[4:6])
	else:
		path_weights = train_params['path_weights'] 
	
	tiny_version = len(ktrain.get_anchors(train_params['path_anchors'])) == 6
	if tiny_version: model = 'tiny'
	elif train_params.get('spp', False): model = 'spp'
	else: model = 'yolo'
	
	mode = '{} | {} | {}'.format(
			'bw' if best_weights else 's2',
			model,
			train_params['mode'] if train_params.get('mode', None) is not None else '-')
		
	result = '{model_folder}\t{version}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:d}\t{mode}\t{training_time}'.format(
				model_folder = '/'.join(model_folder.split('/')[-2:]), 
				version = train_params.get('version', ''),
				input_shape = 'multi_scale' if train_params.get('multi_scale', False) else train_params['input_shape'],
				annotations = train_params['path_annotations'],
				anchors = train_params['path_anchors'],
				pretraining = path_weights,
				frozen_training = train_params['freeze_body'],
				mode = mode,
				training_time = train_diff
			)
	
	result += '\t{train_loss}\t{val_loss}'.format(
				train_loss=train_loss, val_loss=val_loss).replace('.', ',')
	
	if eval_stats_train is not None:
		result += '\t{mAPtrain:.5f}\t{mAP50train:.5f}\t{R100train:.5f}'.format(
					mAPtrain = eval_stats_train['total'][0]*100 if 'total' in eval_stats_train else 0,
					mAP50train = eval_stats_train['total'][1]*100 if 'total' in eval_stats_train else 0,
					R100train = eval_stats_train['total'][7]*100 if 'total' in eval_stats_train else 0,
				).replace('.', ',')
	else:
		result += '\t'*3
		
	result += '\t{mAPval:.5f}\t{mAP50val:.5f}\t{mAP75val:.5f}\t{R100val:.5f}'.format(
				mAPval = eval_stats_val['total'][0]*100 if 'total' in eval_stats_val else 0,
				mAP50val = eval_stats_val['total'][1]*100 if 'total' in eval_stats_val else 0,
				mAP75val = eval_stats_val['total'][2]*100 if 'total' in eval_stats_val else 0,
				R100val = eval_stats_val['total'][7]*100 if 'total' in eval_stats_val else 0,
			).replace('.', ',')

	result += '\t{mAPS}\t{mAPM}\t{mAPL}'.format(
				mAPS=eval_stats_val['total'][3]*100 if 'total' in eval_stats_val else 0, 
				mAPM=eval_stats_val['total'][4]*100 if 'total' in eval_stats_val else 0, 
				mAPL=eval_stats_val['total'][5]*100 if 'total' in eval_stats_val else 0, 
			).replace('.', ',')
	
	return result


def plot_prediction_resume(eval_stats, videos, class_names, by, annotations_file, model_num, plot):
	occurrences = {}
	for v in videos:
		v_res = {}
		for c in range(len(class_names)):
			v_res[class_names[c]] = eval_stats[v]['cat_{}'.format(c)][1]
		occurrences[v] = v_res
		
	
	occurrences = pd.DataFrame.from_dict(occurrences, orient='index')
	if by == 'video': occurrences = occurrences = occurrences.transpose()
	occurrences = occurrences.replace({-1: np.nan})
	
	meds = occurrences.mean()
	meds = meds.sort_values(ascending=False)
	occurrences = occurrences[meds.index]
	
	ann = 'Train' if 'train' in annotations_file else 'Val'
	
	if plot:
		if by == 'video':
			keys = meds.index
			bars = [ eval_stats[k]['total'][1] for k in keys ]
		else:
			keys = [ 'cat_{}'.format(class_names.index(k)) for k in meds.index ]
			bars = [ eval_stats[k][1] for k in keys ]
		occurrences.boxplot(figsize=(20,7));
		plt.bar(range(1, len(bars)+1), bars, alpha=0.4)
		plt.axhline(eval_stats['total'][1], c='y', label="Total mAP@50")
		plt.xticks(rotation=50);
		plt.suptitle('Model {} | {} | Boxcox by: {}'.format(model_num, ann, by), fontsize=20);
		plt.show();
	
	return occurrences



def main(path_results, dataset_name, model_num, score, iou, num_annotation_file=1, 
		 plot=True, full=True, best_weights=True):
	
	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	class_names = ktrain.get_classes(train_params['path_classes'])
#	train_params['path_annotations'][1] = train_params['path_annotations'][1].replace('_sk10', '_sk20')
	
	annotations_file = train_params['path_annotations'][num_annotation_file]
	if 'adl' in annotations_file and train_params.get('size_suffix', '') != '':
		annotations_file = annotations_file.replace(train_params.get('size_suffix', ''), '')
#	annotations_file = annotations_file.replace('.txt', '_pr416.txt')
		
	print(' * Exploring:', annotations_file)
	
	if best_weights:
		preds_filename = '{}preds_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
		eval_filename = '{}stats_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
	else:
		preds_filename = '{}preds_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
		eval_filename = '{}stats_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)

#	print(preds_filename)
	print('='*80)
	
	train_diff, train_loss, val_loss = get_train_resume(model_folder)
	_ = prediction_utils.predict_and_store_from_annotations(model_folder, train_params, annotations_file, preds_filename, score, iou, best_weights=best_weights)
	occurrences = None
	eval_stats, videos = get_full_evaluation(annotations_file, preds_filename, class_names, full)
	
#	resume = get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou)
	plot_prediction_resume(eval_stats, videos, class_names, 'video', annotations_file, model_num, plot);
	occurrences = plot_prediction_resume(eval_stats, videos, class_names, 'class', annotations_file, model_num, plot)
#	print(resume)
	
	return model_folder, class_names, videos, occurrences, eval_stats, train_params, (train_loss, val_loss, train_diff)


# %%
	

#dataset_name = 'ilsvrc'
#
#for model in os.listdir(path_results + dataset_name):
#	model_folder = path_results + dataset_name + '/' + model
#	print(model_folder)
#	tp = json.load(open(model_folder + '/train_params.json', 'r'))
#	anns = tp['path_annotations']
#	anns[1] = './dataset_scripts/ilsvrc/annotations_val_sk20.txt'
#	tp.update({'path_annotations': anns})
##	tp.update({'eval_train_score': MIN_SCORE, 'eval_val_score': 0.0})
#	json.dump(tp, open(model_folder + '/train_params.json', 'w'))

	
# %%

def main_evaluation():
	
	# %%
	
	path_results = '/mnt/hdd/egocentric_results/'
	dataset_name = 'adl' 		# ilsvrc | adl
	
	#score = 
	iou = 0.5
	plot = False
	full = 	True
#	best_weights = False
#	filter_out_score = [0.1,0.2,0.3,0.4,0.5]
	filter_out_score = None
	
	model_num = 87

	if dataset_name != 'adl':
		annotation_files = [(0.005, 1), (0.005, 0)]
	else:
		annotation_files = [(0, 1), (MIN_SCORE, 0)]
	
	times = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	eval_stats_arr = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	videos_arr = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	
	for score, num_annotation_file in annotation_files:		 # ,0
	
		print('='*80)
		print('dataset = {}, num_ann = {}, model = {}'.format(
				dataset_name, num_annotation_file, model_num))	
		print('='*80)
		
		times[num_annotation_file] = time.time()
	

		if num_annotation_file == 1:
			model_folder, _, _, _, eval_stats_t, _, loss_t = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot=False, full=True, best_weights=True, filter_out_score=filter_out_score)
			
			model_folder, _, videos, _, eval_stats_f, tp, loss_f = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot=False, full=True, best_weights=False, filter_out_score=filter_out_score)
		
			print('Val. best_weigths mAP: {:.3f}'.format(eval_stats_t['total'][1]*100))
			print('Val. stage_2 mAP: {:.3f}'.format(eval_stats_f['total'][1]*100))
		
			if eval_stats_t['total'][1] >= eval_stats_f['total'][1]:
				eval_stats_arr[num_annotation_file] = eval_stats_t
				best_weights = True
			else:
				eval_stats_arr[num_annotation_file] = eval_stats_f
				best_weights = False

			videos_arr[num_annotation_file] = videos
			
		else:
			model_folder, class_names, videos, occurrences, eval_stats, train_params, (train_loss, val_loss, train_diff) = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot, full, best_weights=best_weights, filter_out_score=None)
			
			eval_stats_arr[num_annotation_file] = eval_stats
			videos_arr[num_annotation_file] = videos
			
		times[num_annotation_file] = (time.time() - times[num_annotation_file])/60
	
	eval_stats_train, eval_stats_val = eval_stats_arr
	videos_train, videos_val = videos_arr
	full_resume = get_excel_resume_full(model_folder, train_params, 
								 train_loss, val_loss, 
								 eval_stats_train, eval_stats_val, train_diff,
								 best_weights)
	
	
	print('='*80)
	print(full_resume)
	pyperclip.copy(full_resume)
	print('='*80)
	print('Val mAP@50: {:.3f}'.format(eval_stats_val['total'][1]*100))


# %%
	
def main_v2(path_results, dataset_name, model_num, input_shape=None, eval_train=True, full=False):
	
	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	class_names = ktrain.get_classes(train_params['path_classes'])
	
	annotation_files = train_params['path_annotations']
#	annotation_files[1] = annotation_files[1].replace('_sk20', '_sk3')
#	train_params['eval_val_score'] = 0.05
	
	if input_shape is not None:
		train_params['input_shape'] = input_shape
	
	
#	train_params['eval_val_score'] = 0.005
#	train_params['eval_train_score'] = 0.005
	
	# Pred and eval val bw
	_, preds_filename_val_bw = prediction_utils.predict_and_store_from_annotations(model_folder, 
										train_params, annotation_files[1], model_folder, 
										image_size=train_params['input_shape'], 
										score=train_params.get('eval_val_score'), nms_iou=0.5,
										best_weights=True, raw_eval='def')
	eval_stats_val_bw, videos = get_full_evaluation(annotation_files[1], preds_filename_val_bw, class_names, full)
	
	# Pred and eval val st2
	_, preds_filename_val_st2 = prediction_utils.predict_and_store_from_annotations(model_folder, 
										train_params, annotation_files[1], model_folder, 
										train_params['input_shape'],
										score=train_params.get('eval_val_score'), nms_iou=0.5, 
										best_weights=False, raw_eval='def')
	eval_stats_val_st2, videos = get_full_evaluation(annotation_files[1], preds_filename_val_st2, class_names, full)
	
	print('='*80)
	print('mAP@50 | val_bw: {:.2f} | val_st2: {:2f}'.format(eval_stats_val_bw['total'][1]*100, eval_stats_val_st2['total'][1]*100))
	print('='*80)
	
	# Pred and eval train
	best_weights = eval_stats_val_bw['total'][1] > eval_stats_val_st2['total'][1]
	if eval_train:
		_, preds_filename_train = prediction_utils.predict_and_store_from_annotations(model_folder, 
											train_params, annotation_files[0], model_folder, 
											train_params['input_shape'],
											score=train_params.get('eval_train_score'), nms_iou=0.5, 
											best_weights=best_weights, raw_eval='def')
		eval_stats_train, videos = get_full_evaluation(annotation_files[0], preds_filename_train, class_names, full)
	else:
		eval_stats_train = None
	
	
	train_diff, train_loss, val_loss = get_train_resume(model_folder)
	eval_stats_val = eval_stats_val_bw if best_weights else eval_stats_val_st2
	full_resume = get_excel_resume_full(model_folder, train_params, 
								 train_loss, val_loss, 
								 eval_stats_train, eval_stats_val, train_diff,
								 best_weights)
	
	print('='*80)
	print(full_resume)
	pyperclip.copy(full_resume)
	print('='*80)
	if eval_train:
		print('mAP@50 | val_bw: {:.4f} | val_st2: {:.4f} | train: {:.4f}'.format(
					eval_stats_val_bw['total'][1]*100, eval_stats_val_st2['total'][1]*100, eval_stats_train['total'][1]*100
				))
		print('R10 | val_bw: {:.4f} | val_st2: {:.4f} | train: {:.4f}'.format(
					eval_stats_val_bw['total'][7]*100, eval_stats_val_st2['total'][7]*100, eval_stats_train['total'][7]*100
				))
	else:
		print('mAP@50 | val_bw: {:.4f} | val_st2: {:.4f}'.format(
					eval_stats_val_bw['total'][1]*100, eval_stats_val_st2['total'][1]*100
				))
		print('R10 | val_bw: {:.4f} | val_st2: {:.4f}'.format(
					eval_stats_val_bw['total'][7]*100, eval_stats_val_st2['total'][7]*100
				))
	print('='*80)
	
#	return full_resume
	return eval_stats_train, eval_stats_val, videos, class_names


def main_evaluation_v2():
	
	path_results = '/mnt/hdd/egocentric_results/'
	dataset_name = 'ilsvrc'  		# adl / ilsvrc / voc / kitchen
	full = False
	model_num = 8
	input_shape = (512,512) 			# 416 512 608
	eval_train = True
	
	import tensorflow as tf
	import keras.backend.tensorflow_backend as ktf

	def get_session(gpu_fraction=0.90):
	    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
	                                allow_growth=True)
	    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	ktf.set_session(get_session())
	
	_ = main_v2(path_results, dataset_name, model_num, input_shape=input_shape, eval_train=eval_train, full=full)
	
	
	
# %%
	
if __name__ == "__main__": main_evaluation_v2()

