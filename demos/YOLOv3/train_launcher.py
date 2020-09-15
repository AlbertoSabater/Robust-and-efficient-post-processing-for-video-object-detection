#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:40:46 2019

@author: asabater
"""

import os

import train
import time

from evaluate_model import MIN_SCORE

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import resume_train

def get_session(gpu_fraction=0.90):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
if os.environ["CUDA_VISIBLE_DEVICES"] == 0: ktf.set_session(get_session())



version = ''


# %%

if False: 
	pass
	# %%
	
	version = '_v3_35' 		# _v2_27    _v3_8  	# _v3_35
	suff = 1000

	train_params = {
			'path_results': '/mnt/hdd/egocentric_results/',
			'path_anchors': 'base_models/yolo_anchors.txt',
#			'path_anchors': 'base_models/tiny_yolo_anchors.txt',
			
			# COCO
#			'dataset_name': 'coco',
#			'path_dataset': '/mnt/hdd/datasets/coco/',
#			'path_annotations': ['./dataset_scripts/coco/annotations_coco_train.txt',
#								'./dataset_scripts/coco/annotations_coco_val.txt'],
#			'path_classes': './dataset_scripts/coco/coco_classes.txt',
	
			# VOC
#			'dataset_name': 'voc',
#			'path_dataset': '/mnt/hdd/datasets/VOC/',
#			'path_annotations': ['./dataset_scripts/voc/annotations_voc_train.txt',
#								'./dataset_scripts/voc/annotations_voc_val.txt'],
#			'path_classes': './dataset_scripts/adl/adl_classes{}.txt'.format(version),
			
			# ADL
#			'dataset_name': 'adl',
#			'path_dataset': '/home/asabater/projects/ADL_dataset/',
#			'path_annotations': ['./dataset_scripts/adl/annotations_adl_train{}.txt'.format(version),
#								'./dataset_scripts/adl/annotations_adl_val{}.txt'.format(version)],
#			'path_classes': './dataset_scripts/adl/adl_classes{}.txt'.format(version),
			
			# ADL + OPEN_IMAGES
			##
#			'dataset_name': 'adl',
#			'path_dataset': '',
#			'path_annotations': ['./dataset_scripts/adl/annotations_adl_train{}_extra{}.txt'.format(version, suff),
#								'./dataset_scripts/adl/annotations_adl_val{}_noextra.txt'.format(version)],
##			'path_classes': './dataset_scripts/adl/adl_classes{}.txt'.format(version),
			
			# TEST SAMPLES
#			'dataset_name': 'adl',
#			'path_dataset': '/home/asabater/projects/ADL_dataset/',
#			'path_annotations': ['./dataset_scripts/adl/short_v3_8_train.txt',
#								'./dataset_scripts/adl/short_v3_8_val.txt'],
##			'path_classes': './dataset_scripts/adl/adl_classes{}.txt'.format(version),
			
			# OPEN_IMAGES
#			'dataset_name': 'adl',
#			'path_dataset': '',
#			'path_annotations': ['./dataset_scripts/open_images/annotations_oi_train{}.txt'.format(version),
#								'./dataset_scripts/adl/annotations_adl_val{}_noextra.txt'.format(version)],
##			'path_classes': './dataset_scripts/adl/adl_classes{}.txt'.format(version),

			# ILSVRC
#			'dataset_name': 'ilsvrc',
#			'path_dataset': '/home/asabater/projects/ILSVRC2015/Data/VID/',
#			'path_annotations': ['./dataset_scripts/ilsvrc/annotations_train{}.txt'.format('_sk20'),
#								 './dataset_scripts/ilsvrc/annotations_val{}.txt'.format('_sk20')],
#			'path_classes': './dataset_scripts/ilsvrc/imagenet_vid_classes.txt',
			
			# ILSVRC + DET
			'dataset_name': 'ilsvrc',
			'path_dataset': '',
			'path_annotations': ['./dataset_scripts/ilsvrc/annotations_train_vid_det_fgfa_split.txt',
								 './dataset_scripts/ilsvrc/annotations_val{}_fgfa.txt'.format('_sk15')],
			'path_classes': './dataset_scripts/ilsvrc/imagenet_vid_classes.txt',
			
			# EPIC_KITCHES
#			'dataset_name': 'kitchen',
#			'path_dataset': '',
#			'path_annotations': ['./dataset_scripts/kitchen/annotations_kitchen_train{}.txt'.format(version),
#								 './dataset_scripts/kitchen/annotations_kitchen_val{}.txt'.format(version)],
#			'path_classes': './dataset_scripts/kitchen/kitchen_classes{}.txt'.format(version),
			


#			'spp': True, 'path_weights': 'base_models/yolov3-spp.h5',
			'spp': False, 'path_weights': 'base_models/yolo.h5',
#			'spp': False, 'path_weights': 'base_models/yolo_tiny.h5',
#			'spp': False, 'path_weights': '',
#			'spp': False, 'path_weights': '/mnt/hdd/egocentric_results/coco/1010_1108_model_0/weights/ep043-loss45.84016-val_lossinf-val_mAP0.4054.h5',

			'input_shape': [608,608], 			# [416,416] [512,512] [608,608]
			'size_suffix': '', 'version': version,
			'mode': None,
			'multi_scale': True,
			'freeze_body': 2,
			'frozen_epochs': 5,
			'loss_percs': {},
			'eval_train_score': MIN_SCORE,
			'eval_val_score': 0,
			'mAP_metric': True,
			'mAP_lr': True
			}

	try: 
		train.main(train_params)
	except Exception as e:
		
		print(e) 
		
		model_num = int(sorted(os.listdir(train_params['path_results']+train_params['dataset_name']))[-1].split('_')[-1])
		
		# Recover from a GPU memory error
		while True:
#			break
			try:
				print('*** TRYING TO RECOVER MODEL NUM:', model_num)
				from keras import backend as K
				import gc
				K.clear_session()
				gc.collect()
				res = resume_train.resume_training(train_params['path_results'], train_params['dataset_name'], model_num)
				break
			except Exception as e:
				print(e)
				continue
#				break
		

# %%

if True:
	# %%
	
	path_results =  '/mnt/hdd/egocentric_results/'
	dataset_name = 'ilsvrc'
	model_num = 9
	
#	path_results =  '/mnt/hdd/egocentric_results/'
#	dataset_name = 'voc'
#	model_num = 5
	
	# Recover from a GPU memory error
	while True:
		try:
			print('*******************')
			print('*** TRYING TO RECOVER MODEL NUM:', model_num)
			resume_train.resume_training(path_results, dataset_name, model_num)
			break
		except Exception as e:
			print(e)	
			continue


