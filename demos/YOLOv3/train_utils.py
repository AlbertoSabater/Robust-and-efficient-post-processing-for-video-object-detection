#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:37:08 2019

@author: asabater
"""

import os
import datetime
import shutil


def create_folder_if_not_exists(folder_path):
	if not os.path.isdir(folder_path): os.makedirs(folder_path)
	

def create_model_folder(path_results, dataset_name):
	
#	path_log = path_results + dataset_name + '/logs/'
	path_model = path_results + dataset_name
	
	# Create logs and model main folder
#	create_folder_if_not_exists(path_log)
	create_folder_if_not_exists(path_model)

	model_name = get_model_name(path_results + dataset_name)
#	path_log += model_name; 
	path_model += model_name
#	create_folder_if_not_exists(path_log)
	create_folder_if_not_exists(path_model)
	create_folder_if_not_exists(path_model + 'weights/')
	
#	return path_log, path_model
	return path_model


def get_model_name(model_dir):
	folder_num = len(os.listdir(model_dir))
	return '/{}_model_{}/'.format(datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)


# Remove folders whose training has not finished the frozen stage
def remove_null_trainings(path_results, dataset_name):
	folder_models = path_results + dataset_name + '/'
	models = os.listdir(folder_models)
	
	models_to_remove = []
	for model in models:
		if 'weights' not in os.listdir(folder_models + model) or \
			('trained_weights_stage_1.h5' not in os.listdir(folder_models + model + '/weights/') and\
			'trained_weights_final.h5' not in os.listdir(folder_models + model + '/weights/')):
				models_to_remove.append(model)
			
	for mtr in models_to_remove:
#		for f in os.listdir(folder_models + mtr): 
#			print('remove', folder_models + mtr + '/' + f)
#			os.remove(folder_models + mtr + '/' + f)
		print('remove', folder_models + mtr)
#		os.rmdir(folder_models + mtr)
		shutil.rmtree(folder_models + mtr) 
		
	
	count = 0
	for i, model in enumerate(sorted(os.listdir(folder_models))):
		if int(model.split('_')[-1]) != count:
			print('rename {} -> {}'.format(model, '_'.join(model.split('_')[:-1]) + '_' + str(count)))
			os.rename(folder_models + model, folder_models + '_'.join(model.split('_')[:-1]) + '_' + str(count))
		count += 1


def get_best_weights(model_folder):
	files = [ f for f in os.listdir(model_folder + '/weights/') if f.startswith('ep') ]
	if len(files) == 0: return model_folder + 'weights/weights.h5'
	val_losses = [ [ float(v[8:]) for v in f[:-3].split('-') if v.startswith('val_loss') ][0] for f in files ]
	model_weights = model_folder + 'weights/' + files[val_losses.index(min(val_losses))]
	return model_weights

def get_model_path(path_results, dataset_name, model_num):
	return '{}{}/{}/'.format(path_results, dataset_name, 
				[ f for f in os.listdir(path_results + dataset_name) if f.endswith('model_{}'.format(model_num)) ][0])


# Remove the worst weights of a model
def remove_worst_weights(path_model):
	files_model = [ '/weights/' + f for f in os.listdir(path_model + '/weights') if f.endswith('.h5') and not f.startswith('trained_weights') ]
	files_model = sorted(files_model, 
						 key=lambda x: [ float(i[8:]) for i in x[:-3].split('/')[-1].split('-') if i.startswith('val_loss') ][0], 
						 reverse=False)
	
	for fm in files_model[1:]:
		print('Removing', path_model + fm)
		os.remove(path_model + fm)


# Print learning_rate on each epoch
def get_lr_metric(optimizer):
	def lr(y_true, y_pred):
		return optimizer.lr
	return lr



