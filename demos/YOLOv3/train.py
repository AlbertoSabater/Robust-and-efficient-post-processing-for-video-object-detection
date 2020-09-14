#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:38:36 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from keras.optimizers import Adam

import train_utils
import numpy as np
import json
import os

from data_factory import data_generator_wrapper_custom
from emodel import create_model, loss, xy_loss, wh_loss, confidence_loss, \
					confidence_loss_obj, confidence_loss_noobj, class_loss, \
					precision, recall, f1

import evaluate_model
import prediction_utils
import time
import pyperclip
import datetime
import argparse


print_indnt = 12
print_line = 100


# %%

def log(msg):
	with open("log.txt", "a") as log_file:
		   log_file.write('{} | {}\n'.format(str(datetime.datetime.now()), msg))


def get_train_params_from_args():
	
	description = '''
		Script to train a YOLO v3 model. Note that the classes especified in path_classes
		must be the same used in path_annotations_train and path_annotations_val.\n
		If using --spp make sure to load the weights that share the same NN architecture.
	'''
	
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("path_results", help="path where the store the training results")
	parser.add_argument("dataset_name", help="subfolder where to store the training results")
	parser.add_argument("path_classes", help="dataset classes file")
	parser.add_argument("path_anchors", help="anchors file")
	parser.add_argument("path_annotations_train", help="train annotations file")
	parser.add_argument("path_annotations_val", help="validation annotations file")
	parser.add_argument("path_weights", help="path to pretrained weights")
	parser.add_argument("freeze_body", help="0 to not freezing\n1 to freeze backbone\n2 to freeze all the model")

	parser.add_argument("--path_dataset", help="path to each training image if not specified in annotations file", default='', type=str)
	parser.add_argument("--frozen_epochs", help="number of frozen training epochs. Default 15", type=int, default=15)
	parser.add_argument("--input_shape", help="training/validation input image shape. Must be a multiple of 32. Default 416", type=int, default=416)
	parser.add_argument("--spp", help="to use Spatial Pyramid Pooling", action='store_true')
	parser.add_argument("--multi_scale", help="to use multi-scale training", action='store_true')
	
	args = parser.parse_args()

	train_params = {
			'path_results': args.path_results,
			'dataset_name': args.dataset_name,
			'path_dataset': args.path_dataset,
			'path_classes': args.path_classes,
			'path_anchors': args.path_anchors,
			'path_annotations': [args.path_annotations_train, args.path_annotations_val],
			'path_weights': args.path_weights,
			'freeze_body': int(args.freeze_body),
			'frozen_epochs': int(args.frozen_epochs),
			'input_shape': [int(args.input_shape), int(args.input_shape)],
			'spp': args.spp,
			'multi_scale': args.multi_scale,
			'size_suffix': '', 'version': '',
			'mode': None,
			'loss_percs': {}, 		# Use this parameter to weight loss components
									# keys: [xy, wh, confidence_obj, confidence_noobj, class]
			}
	
	return train_params


def load_data_and_initialize_training(path_results, dataset_name, path_dataset, 
								   path_annotations, mode, resume_training=False, **kwargs):
	
	if not resume_training:
		# Remove folders of non finished training
		title = 'Remove null trainings'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
		train_utils.remove_null_trainings(path_results, dataset_name)
		print('='*print_line)

	# Load train and val annotations
	np.random.seed(10101)
	with open(path_annotations[0]) as f: lines_train = f.readlines()
	with open(path_annotations[1]) as f: lines_val = f.readlines()
	num_train, num_val = len(lines_train), len(lines_val)
	np.random.shuffle(lines_train), np.random.shuffle(lines_val)
	np.random.seed(None)

	lines_train = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
						+ ' ' + ' '.join(ann.split(' ')[1:]) for ann in lines_train ]
	lines_val = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
						+ ' ' + ' '.join(ann.split(' ')[1:]) for ann in lines_val ]
	


	if not resume_training:

		# If model use recurrent layers, calculate the recurrence lenght
		td_len = None if mode is None else len(lines_train[0].split(' ')[0].split(','))

		# Set batch size according to the model type
		if mode is None:
			batch_size_frozen = 32		  # 32
			batch_size_unfrozen = 4		 # note that more GPU memory is required after unfreezing the body
		else:
			batch_size_frozen = 8		  # 32
			batch_size_unfrozen = 2		 # note that more GPU memory is required after unfreezing the body

		# Initialize model folder
		title = 'Create and get model folders'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
		path_model = train_utils.create_model_folder(path_results, dataset_name)
		model_num = int(path_model.split('/')[-2].split('_')[-1])
		print(path_model)
		print('='*print_line)
		log('NEW TRAIN {}'.format(model_num))
	
		return lines_train, lines_val, \
				{'batch_size_frozen': batch_size_frozen, 'batch_size_unfrozen': batch_size_unfrozen,
				  'num_val': num_val, 'num_train': num_train, 'td_len': td_len, 'model_num': model_num,
				  'path_model': path_model}

	return lines_train, lines_val, None


def store_train_params(train_params):
	log('TRAIN PARAMS: {}'.format(train_params))
	print(train_params)
	with open(train_params['path_model'] + 'train_params.json', 'w') as f:
		json.dump(train_params, f)
	print("train_params stored as json")

	title = 'Excel params'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	excel_resume = evaluate_model.get_excel_resume_full(train_params['path_model'], train_params, '', '', 
						  {}, {}, '', None)
	print(excel_resume)		
	pyperclip.copy(excel_resume)
	

def mAP_callback(model, train_params, class_names):

	def get_mAP(epoch, logs):
		
		print('***', 'Evaluating mAP')
		
		t = time.time()
		
		weights_path = train_params['path_model'] + '/weights/' + \
					sorted([ f for f in os.listdir(train_params['path_model'] + '/weights/') if f.startswith('ep') ])[-1]
		print("weights_path", weights_path)
		
		preds_filename_epoch = '{}preds_{}_{}_score{}_iou{}.json'.format(train_params['path_model'], 
					   'stage2', train_params['path_annotations'][1].split('/')[-1][:-4], 
					   train_params['eval_val_score'], 0.5)
		eval_stats_epoch = preds_filename_epoch.replace('preds', 'stats')
		if os.path.isfile(preds_filename_epoch): os.remove(preds_filename_epoch)
		if os.path.isfile(eval_stats_epoch): os.remove(eval_stats_epoch)
		
		_, preds_filename_epoch = prediction_utils.predict_and_store_from_annotations(train_params['path_model'], 
											train_params, train_params['path_annotations'][1], train_params['path_model'], 
											train_params['input_shape'], 
											score=train_params['eval_val_score'], nms_iou=0.5,
											best_weights=None, raw_eval='def',
											model_path=weights_path)
		eval_stats_epoch, videos = evaluate_model.get_full_evaluation(train_params['path_annotations'][1], preds_filename_epoch, 
																 class_names, full=False)
		
		os.remove(preds_filename_epoch); os.remove(preds_filename_epoch.replace('preds', 'stats'))
		epoch_mAP = eval_stats_epoch['total'][1]
		
		os.rename(weights_path, weights_path.replace('.h5', '-val_mAP{:.4f}.h5'.format(epoch_mAP)))
		
		logs.update({'val_mAP': epoch_mAP})
		print('***', 'mAP Evaluated', epoch_mAP)
		
		print('*** Elapsed mAP Evaluation time: {:.2f}'.format((time.time()-t)/60))
		
	return get_mAP


def initialize_model(path_classes, path_anchors, path_model, input_shape, freeze_body, 
				  path_weights, path_annotations, eval_val_score, td_len=None, mode=None, 
				  spp=False, loss_percs={}, mAP_lr=False, **kwargs):

	class_names = ktrain.get_classes(path_classes)
	num_classes = len(class_names)
	anchors = ktrain.get_anchors(path_anchors)
	
	# Create model
	title = 'Create Keras model'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	model = create_model(input_shape, anchors, num_classes, 
						 freeze_body=freeze_body,
						 weights_path=path_weights, td_len=td_len, mode=mode, 
						 spp=spp, loss_percs=loss_percs)
	
	# Store model architecture
	model_architecture = model.to_json()
	with open(path_model + 'architecture.json', 'w') as f:
		json.dump(model_architecture, f)
	print("Model architecture stored as json")
	print('='*print_line)
	
#	if mAP_lr: mc_filename = 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-val_mAP{val_mAP:.4f}.h5'
#	else: mc_filename = 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
	mc_filename = 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
	monitor = 'val_mAP' if mAP_lr else 'val_loss'
	mode = 'max' if mAP_lr else 'min'
	print('***', monitor)
	
	# Training callbacks
	callbacks = {
			'logging': TensorBoard(log_dir = path_model),
			'checkpoint': ModelCheckpoint(path_model + 'weights/' + \
										 mc_filename, 	#'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5',
										 monitor='val_loss', 
										 save_weights_only=True, 
										 save_best_only=False,
										 period=1),
			'reduce_lr_1': ReduceLROnPlateau(monitor='loss', min_delta=0.5, factor=0.1, patience=3, verbose=1, mode=mode),
			'reduce_lr_2': ReduceLROnPlateau(monitor=monitor, min_delta=0, factor=0.1, patience=3, verbose=1, mode=mode),
			'early_stopping': EarlyStopping(monitor=monitor, min_delta=0, patience=8, verbose=1, mode=mode)
		}

	log('MODEL CREATED')

	return model, callbacks, anchors, num_classes, class_names


# Train with frozen layers first, to get a stable loss.
def train_frozen_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, 
					   path_model, num_train, num_val, input_shape, batch_size_frozen,
					   frozen_epochs, multi_scale, init_epoch=0, init_lr=1e-3, **kwargs):
	
	title = 'Train first stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	log('TRAIN STAGE 1')
	if frozen_epochs == 0: return
	
	optimizer = Adam(lr=init_lr)
	model.compile(optimizer = optimizer,
				  loss = {'yolo_loss': loss},		# use custom yolo_loss Lambda layer.
				  metrics = [
							 xy_loss, wh_loss, confidence_loss, 
							 confidence_loss_obj, confidence_loss_noobj, class_loss,
							 precision, recall, f1,
							 train_utils.get_lr_metric(optimizer),
							 ]	
				  )

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_frozen))
	model.fit_generator(
			data_generator_wrapper_custom(lines_train, batch_size_frozen, input_shape, 
								 anchors, num_classes, random=True, multi_scale=multi_scale),
			steps_per_epoch = max(1, num_train//batch_size_frozen),
			validation_data = data_generator_wrapper_custom(lines_val, batch_size_frozen, 
								   input_shape, anchors, num_classes, random=False, multi_scale=False),
			validation_steps = max(1, num_val//batch_size_frozen),
			epochs = frozen_epochs,
			initial_epoch = init_epoch,
			callbacks=[callbacks['checkpoint'], callbacks['mAP_callback'], callbacks['logging']],
			use_multiprocessing = False, workers = 8, max_queue_size=10
			)
	model.save_weights(path_model + 'weights/trained_weights_stage_1.h5')
	print('='*print_line)
	

# Unfreeze and continue training, to fine-tune.
def train_final_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, 
					   path_model, num_train, num_val, input_shape, batch_size_unfrozen,
					   frozen_epochs, multi_scale, init_epoch=None, init_lr=1e-4, **kwargs):
	
	if init_epoch is None: init_epoch = frozen_epochs
	
	title = 'Train second stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	log('TRAIN STAGE 2')
	# Unfreeze layers
	for i in range(len(model.layers)):
		model.layers[i].trainable = True
	print('Unfreeze all of the layers.')

	optimizer = Adam(lr=min(init_lr,1e-4))
	model.compile(optimizer = optimizer,
				  loss = {'yolo_loss': loss},		# use custom yolo_loss Lambda layer.
				  metrics = [
							 xy_loss, wh_loss, confidence_loss, 
							 confidence_loss_obj, confidence_loss_noobj, class_loss,
							 precision, recall, f1,
							 train_utils.get_lr_metric(optimizer),
							 ]	
				  )

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfrozen))
	model.fit_generator(
		data_generator_wrapper_custom(lines_train, batch_size_unfrozen, input_shape, 
								anchors, num_classes, random=True, multi_scale=multi_scale),
		steps_per_epoch = max(1, num_train//batch_size_unfrozen),
		validation_data = data_generator_wrapper_custom(lines_val, batch_size_unfrozen, 
								input_shape, anchors, num_classes, random=False, multi_scale=False),
		validation_steps = max(1, num_val//batch_size_unfrozen),
		epochs = 500,
		initial_epoch = init_epoch,
		callbacks = [callbacks['checkpoint'], callbacks['mAP_callback'], callbacks['logging'], 
					   callbacks['reduce_lr_2'], callbacks['early_stopping']],
		use_multiprocessing = False, workers = 8, max_queue_size=10
		)
	model.save_weights(path_model + 'weights/trained_weights_final.h5')
	print('='*print_line)
	

# Evaluate trained model
# If best_weights is True, evaluates with the model weights that get the lower mAP
# If best_weights is False, evaluates with the last stored model weights
# If best_weights is None, evaluates with the kind of weights that get the highest mAP
# The reference metric is mAP@50
# score_train and score_val specify the minimum score to filter out predictions before evaluation
# 	increase this value for large datasets
def evaluate_training(train_params, best_weights, score_train=evaluate_model.MIN_SCORE, 
					  score_val=0, iou=0.5, **kwargs):
	log('EVALUATING')
	
	evaluate_model.main_v2(train_params['path_results'], train_params['dataset_name'], train_params['model_num'], full=True)


# %%

def main(train_params):
	
	lines_train, lines_val, tp = load_data_and_initialize_training(**train_params)
	train_params.update(tp)
	store_train_params(train_params)
	
	model, callbacks, anchors, num_classes, class_names = initialize_model(**train_params)
	
	if train_params['mAP_metric'] or train_params['mAP_lr']:
		callbacks['mAP_callback'] = LambdaCallback(on_epoch_end=mAP_callback(model, train_params, class_names))
	
	train_frozen_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, **train_params)
	train_final_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, **train_params)

#	train_utils.remove_worst_weights(train_params['path_model'])
	
	evaluate_training(train_params, best_weights=None, 
				   score_train=train_params['eval_train_score'], 
				   score_val=train_params['eval_val_score'])


# %%

if __name__ == "__main__":
	train_params = get_train_params_from_args()
	main(train_params)






