#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:55:40 2020

@author: asabater
"""


import os

from emodel import darknet_body
from keras.layers import Input, Dense, Flatten, concatenate, UpSampling2D, Conv2DTranspose, Lambda, GlobalAveragePooling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Reshape
from keras.models import model_from_json
import numpy as np
from ROI_pooling_frcnn import RoiPoolingConv
from keras import backend as K
from tqdm import tqdm
import json




# Creates and initializes the YOLOv3 backbone for the feature maps extraction
def get_backbone(path_weights=None, downsample_rate=32):

	dummy_img = Input(shape=(None,None,3), name='dummy_img')
	backbone = darknet_body(dummy_img)
	backbone = Model(dummy_img, backbone, name='backbone')
	
	if downsample_rate == 32: backbone = backbone.layers[:] 			# 13x1054
	elif downsample_rate == 16: backbone = backbone.layers[:153] 		# 26x512
	elif downsample_rate == 8: backbone = backbone.layers[:93]			# 52x256
	elif downsample_rate == 4: backbone = backbone.layers[:33]			# 104x128
	elif downsample_rate == 2: backbone = backbone.layers[:15]			# 208x64
	elif downsample_rate == 1: backbone = backbone.layers[:4]			# 416x32
	elif downsample_rate == 0: backbone = backbone.layers[:1]			# 413x3
	else: raise ValueError('donwsample_rate not valid')
	
	backbone = Model(dummy_img, backbone[-1].output, name='backbone')
	print('Last layer {}: {}'.format(len(backbone.layers), backbone.layers[-1]))
	
	if path_weights is not None:
		print('** Pretraining backbone **')
		backbone.load_weights(path_weights, by_name=True, skip_mismatch=False)
	else:
		print('** No pretraining **')
		
	for i in range(len(backbone.layers)): 
		backbone.layers[i].trainable = False

	return backbone


# Creates the appearance embedding model generator
# Unused?
def get_branch_body(pool_size, downsample_rate, emb_len, use_roi_layer):
	
	if use_roi_layer:
		backbone_output = Input(shape=(None,None,1024//(32//downsample_rate)))
		dummy_roi = Input(shape=(1, 4))
		branch_body = RoiPoolingConv(pool_size=pool_size, num_rois=1)([backbone_output, dummy_roi]) 	# (None, 1, 7, 7, 512)
		branch_body = Reshape((pool_size, pool_size, 1024//(32//downsample_rate)))(branch_body)
		
	else:
		roi_input = Input(shape=(pool_size, pool_size, 1024//(32//downsample_rate)))
		branch_body = roi_input
		branch_body = Lambda(lambda x: x)(branch_body)
		branch_body = GlobalAveragePooling2D()(branch_body)
		branch_body = Dense(emb_len)(branch_body)
		
	branch_body = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')(branch_body)
	
	if use_roi_layer:
		branch_body = Model([backbone_output, dummy_roi], branch_body, name='branch_body')
	else:
		branch_body = Model(roi_input, branch_body, name='branch_body')

	branch_body.summary()
	
	return branch_body



# Unused?
def get_embs(backbone, branch_body, downsample_rate, use_roi_layer, pool_size, name):
	inp_roi = Input(shape=(1, 4), name='roi_{}'.format(name))
	
	if backbone is None:
		if use_roi_layer:
			inp_img = Input(shape=(None,None,1024//(32//downsample_rate)))
			branch = inp_img
		else:
			inp_img = Input(shape=(pool_size, pool_size, 1024//(32//downsample_rate)))
			branch = inp_img
	else:
		inp_img = Input(shape=(None,None,3), name='img_{}'.format(name))
		branch = backbone(inp_img)
		
	if use_roi_layer:
		branch = branch_body([branch, inp_roi])
		return branch, inp_img, inp_roi
	else:
		branch = branch_body(branch)
		return branch, inp_img


# Unused?
def get_triplet_model(path_weights, downsample_rate, pool_size, emb_len, gap, use_backbone, use_roi_layer, branch_type, **kwargs):
	if use_backbone: backbone = get_backbone(path_weights, downsample_rate)
	else: backbone = None
	branch_body = get_branch_body(pool_size, downsample_rate, emb_len, gap, use_roi_layer, branch_type)
	
	branchs = [ get_embs(backbone, branch_body, downsample_rate, use_roi_layer, pool_size, name) for name in ['A', 'P', 'N']]
	output = concatenate([ b[0] for b in branchs], axis=-1, name='list_of_embds')
	
	if use_roi_layer:
		tripletModel = Model(inputs=[ img for _,img,_ in branchs] + [ roi for _,_,roi in branchs], outputs=output)
	else:
		tripletModel = Model(inputs=[ b[1] for b in branchs], outputs=output)
	
	return tripletModel



def triplet_loss(y_true, y_pred, alpha = .2):
	
	total_lenght = y_pred.shape.as_list()[-1]
	print(int(total_lenght*1/3), int(total_lenght*2/3), int(total_lenght*3/3))
	
	anchor = y_pred[:,0:int(total_lenght*1/3)]
	positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
	negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]
	
	print('anchor', anchor, anchor.shape, positive.shape, negative.shape)
	# distance between the anchor and the positive
	pos_dist = K.sum(K.square(anchor-positive),axis=1)

	# distance between the anchor and the negative
	neg_dist = K.sum(K.square(anchor-negative),axis=1)

	# compute loss
	basic_loss = pos_dist-neg_dist+alpha
	loss = K.maximum(basic_loss,0.0)
 
	return loss



def load_branch_body(model_folder_roi):
	
	# Load architecture
	model_arch = json.load(open(model_folder_roi + '/architecture.json', 'r'))
	branch_body = model_from_json(model_arch, {'RoiPoolingConv': RoiPoolingConv})
	model_params = json.load(open(model_folder_roi+'train_params.json', 'r'))
	downsample_rate = model_params['downsample_rate']

	
	# Load weights
	branch_body_weights = sorted(os.listdir(model_folder_roi + '/weights'), key=lambda x: float(x[:-3].split('-')[-1][8:]))
	if len(branch_body_weights) == 0: print(' **  No weights found for branch_body {}'.format(model_folder_roi))
	else: 
		branch_body_weights = model_folder_roi + '/weights/' +  branch_body_weights[0]
		branch_body.load_weights(branch_body_weights, by_name=True, skip_mismatch=False)

	# Create branch
	branch_body = branch_body.get_layer('branch_body')	
	branch_body.name = branch_body.name + '_model_{}'.format(model_folder_roi)
	
	
	if type(branch_body.layers[1]) != RoiPoolingConv:
		branch_tp = json.load(open(model_folder_roi + '/train_params.json', 'r'))
		inp_bck = Input(shape=(None,None,32*downsample_rate))
		inp_roi = Input(shape=(1, 4))
		roi_layer = RoiPoolingConv(pool_size=branch_tp['pool_size'], num_rois=1)([inp_bck, inp_roi])
		roi_layer = Reshape((branch_tp['pool_size'], branch_tp['pool_size'], 32*downsample_rate))(roi_layer)
		branch_body = branch_body(roi_layer)
		branch_body = Model([inp_bck, inp_roi], branch_body)
	
	return branch_body


# Unused?
def load_set_of_branches(path_roi_models):
	
	inp_bck = Input(shape=(None,None,32*downsample_rate))
	inp_roi = Input(shape=(1, 4))
	
	branches = []
	for prm in tqdm(path_roi_models, file=sys.stdout):
		branch_body = load_branch_body(prm)
		branch_body = branch_body([inp_bck, inp_roi])
		branches.append(branch_body)
	
	branch_set_model = Model([inp_bck, inp_roi], branches)
	
	return branch_set_model

