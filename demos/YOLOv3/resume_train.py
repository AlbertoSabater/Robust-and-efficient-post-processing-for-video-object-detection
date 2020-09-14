#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:59:22 2019

@author: asabater
"""

import train
import train_utils
import json
import os
import argparse
from tensorboard.backend.event_processing import event_accumulator
from keras.callbacks import LambdaCallback

# =============================================================================
# Resume training from the last model checkpoint
# =============================================================================


def get_args():
	description = '''
			Script to resume a training of a YOLO v3 model.
		'''
	
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("path_results", help="path where the store the training results")
	parser.add_argument("dataset_name", help="subfolder where to store the training results")
	parser.add_argument("model_num", help="number of the model stored in path_results/dataset_name/")
	args = parser.parse_args()

	path_results = args.path_results
	dataset_name = args.dataset_name
	model_num = args.model_num
	
	return path_results, dataset_name, model_num


def get_init_lr(model_folder, last_epoch):
	tb_files = [ model_folder + f for f in os.listdir(model_folder) if f.startswith('events.out.tfevents') ]
	lrs = []
	for tbf in tb_files:
		try:
			ea = event_accumulator.EventAccumulator(tbf).Reload()
			lrs += [ (e.step, e.value) for e in ea.Scalars('lr') ]
		except: continue
	
#	init_lr = [ v for step, v in lrs if step==last_epoch-1 ][0]
	init_lr = sorted(lrs, key= lambda x: x[0], reverse=False)[-1][1]
	
	return init_lr

	
def resume_training(path_results, dataset_name, model_num):
	
	train.log('RESUMING TRAINING MODEL {}'.format(model_num))
	
	# %%
	
	print('RESUMING TRAINING MODEL {}'.format(model_num))
	model_folder = train.train_utils.get_model_path(path_results, dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	print('*** Train params', train_params)
	train.log('TRAIN PARAMS: {}'.format(train_params))
	
	lines_train, lines_val, _ = train.load_data_and_initialize_training(resume_training = True, **train_params)
	
	last_weights = sorted([ f for f in os.listdir(model_folder + 'weights/') if 'trained' not in f ])[-1]
	last_epoch = int([ s for s in last_weights.split('-') if s.startswith('ep') ][0][2:])
	path_weigths = model_folder + 'weights/' + last_weights
	freeze_body = train_params['freeze_body'] if last_epoch < train_params['frozen_epochs'] else 0
	
	
	init_lr = get_init_lr(model_folder, last_epoch)
#	init_lr = 1e-8
	
	print(' *  Restoring training in epoch {} with lr {}'.format(last_epoch, init_lr))
	print(' *  weights:', path_weigths)
	train.log('RESUMING ON EPOCH {} WITH LR {:.3f}'.format(last_epoch, init_lr))

	
	model, callbacks, anchors, num_classes, class_names = train.initialize_model(train_params['path_classes'], 
					  train_params['path_anchors'], train_params['path_model'], 
					  train_params['input_shape'], freeze_body, 
					  path_weigths, train_params['path_annotations'], train_params['eval_val_score'], 
					  train_params['td_len'], train_params['mode'], 
					  train_params['spp'], train_params['loss_percs'],
					  mAP_lr=train_params['mAP_lr'])
#	model, callbacks, anchors, num_classes, class_names = train.initialize_model(**train_params)

	if train_params['mAP_metric'] or train_params['mAP_lr']:
		callbacks['mAP_callback'] = LambdaCallback(on_epoch_end=train.mAP_callback(model, train_params, class_names))

	init_epoch = last_epoch
	
	if init_epoch < train_params['frozen_epochs']:
		train.train_frozen_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, 
								   init_epoch=init_epoch, init_lr=init_lr, **train_params)
	train.train_final_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, 
								   init_epoch=init_epoch, init_lr=init_lr, **train_params)
	
#	train_utils.remove_worst_weights(train_params['path_model'])
	
	train.evaluate_training(train_params, best_weights=None, 
				   score_train=train_params['eval_train_score'], 
				   score_val=train_params['eval_val_score'])

	return True


if __name__ == "__main__":
	path_results, dataset_name, model_num = get_args()
	resume_training(path_results, dataset_name, model_num)
