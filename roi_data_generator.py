#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:53:02 2020

@author: asabater
"""

from PIL import Image
import numpy as np
import cv2

MIN_PERC_AREA = 0.65

np.random.seed(0)


def preprocess_img_roi(img, roi, input_shape, downsample_rate, random):
	img = cv2.imread(img)
	roi = np.array(list(map(float, roi.split(','))))
	
	# height, width, channel
	ih, iw, _ = img.shape
	h, w = input_shape

	# resize image
	scale = min(w/iw, h/ih)
	nw = int(iw*scale)
	nh = int(ih*scale)
	dx = (w-nw)//2
	dy = (h-nh)//2

	# resize
	new_image, image_data = None, None
	image = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
	# convert into PIL Image object
	image = Image.fromarray(image[:, :, ::-1])
	new_image = Image.new('RGB', (w,h), (128,128,128))
	new_image.paste(image, (dx, dy))
	# convert into numpy array: RGB, 0-1
	image_data = np.array(new_image)/255.

	# correct boxes
	roi[[0,2]] = roi[[0,2]]*scale + dx
	roi[[1,3]] = roi[[1,3]]*scale + dy
	
	# bbox to roi
	roi = roi/downsample_rate
	
	# Min width must be 1. to be processed by RoiPoolingLayer
	roi[[2]] = np.maximum(roi[[2]] - roi[[0]],1.) 		# Transform x2 to w
	roi[[3]] = np.maximum(roi[[3]] - roi[[1]],1.) 		# Transform y2 to h

	return image_data, np.expand_dims(roi, axis=0)



def rand(a=0, b=1):
	return np.random.rand()*(b-a) + a

def preprocess_img_roi_rand(imgs, rois, path_dataset, input_shape, downsample_rate, random=True, fix_coords=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
	'''random preprocessing for real-time data augmentation'''

	# numpy array: BGR, 0-255
	images = [ cv2.imread(path_dataset + i) for i in imgs ]
	# height, width, channel
	ih, iw, _ = images[0].shape
	h, w = input_shape
	rois = [ np.array(list(map(float, roi.split(',')))) for roi in rois ]

	if not random:
		# resize image
		scale = min(w/iw, h/ih)
		nw = int(iw*scale)
		nh = int(ih*scale)
		dx = (w-nw)//2
		dy = (h-nh)//2
	
		# resize
		for i in range(len(images)):
			new_image = None
			img = images[i]
			image = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
			# convert into PIL Image object
			image = Image.fromarray(image[:, :, ::-1])
			new_image = Image.new('RGB', (w,h), (128,128,128))
			new_image.paste(image, (dx, dy))
			# convert into numpy array: RGB, 0-1
			images[i] = np.array(new_image)/255.
	
		for i in range(len(rois)):
			roi = rois[i]
			# correct boxes
			roi[[0,2]] = roi[[0,2]]*scale + dx
			roi[[1,3]] = roi[[1,3]]*scale + dy
			
			# bbox to roi
			roi = roi/downsample_rate
			
			# Min width must be 1. to be processed by RoiPoolingLayer
			if fix_coords:
				roi[[2]] = np.maximum(roi[[2]] - roi[[0]],1.) 		# Transform x2 to w
				roi[[3]] = np.maximum(roi[[3]] - roi[[1]],1.) 		# Transform y2 to hs_data)
			else:
				roi[[2]] = roi[[2]] - roi[[0]] 		# Transform x2 to w
				roi[[3]] = roi[[3]] - roi[[1]] 		# Transform y2 to hs_data)
			
			rois[i] = np.expand_dims(roi, axis=0)
		return images, rois

	# resize image
	new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
	scale = rand(.6, 2)
	if new_ar < 1:
		nh = int(scale*h)
		nw = int(nh*new_ar)
	else:
		nw = int(scale*w)
		nh = int(nw/new_ar)

	# resize
	images = [ cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) for image in images ]
	images = [ Image.fromarray(image[:, :, ::-1]) for image in images ]

	# place image
	dx = int(rand(0, w-nw))
	dy = int(rand(0, h-nh))
	new_images = [ Image.new('RGB', (w,h), (128,128,128)) for i in range(len(images)) ]
	for i in range(len(images)): new_images[i].paste(images[i], (dx, dy))
	# convert into numpy array: BGR, 0-255
	images = [ np.asarray(new_image)[:, :, ::-1] for new_image in new_images ]

	# horizontal flip (faster than cv2.flip())
	h_flip = rand() < 0.5
	if h_flip:
		images = [ image[:, ::-1] for image in images ]

	# distort image
	hue = rand(-hue, hue) * 179
	sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
	val = rand(1, val) if rand()<.5 else 1/rand(1, val)

	images_data = []
	for i in range(len(images)):
		img_hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
		H = img_hsv[:, :, 0].astype(np.float32)
		S = img_hsv[:, :, 1].astype(np.float32)
		V = img_hsv[:, :, 2].astype(np.float32)
	
		H += hue
		np.clip(H, a_min=0, a_max=179, out=H)
	
		S *= sat
		np.clip(S, a_min=0, a_max=255, out=S)
	
		V *= val
		np.clip(V, a_min=0, a_max=255, out=V)
	
		img_hsv[:, :, 0] = H.astype(np.uint8)
		img_hsv[:, :, 1] = S.astype(np.uint8)
		img_hsv[:, :, 2] = V.astype(np.uint8)

		# convert into numpy array: RGB, 0-1
		images_data.append(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB) / 255.0)

	# correct boxes
	for i in range(len(rois)):
		rois[i][[0,2]] = rois[i][[0,2]]*nw/iw + dx
		rois[i][[1,3]] = rois[i][[1,3]]*nh/ih + dy
		
		if rois[i][0] > w or rois[i][1] > h or rois[i][2] < 0 or rois[i][3] < 0:
			return None, None
		
		
		if h_flip:
			rois[i][[0,2]] = w - rois[i][[2,0]]
		init_area = (rois[i][2]-rois[i][0])*(rois[i][3]-rois[i][1])
		rois[i][0:2][rois[i][0:2]<0] = 0
		rois[i][2:3][rois[i][2:3]>w] = w
		rois[i][3:4][rois[i][3:4]>h] = h
		end_area = (rois[i][2]-rois[i][0])*(rois[i][3]-rois[i][1])
		perc_area = end_area/init_area
		if perc_area < MIN_PERC_AREA: return None, None
		

		rois[i] = rois[i]/downsample_rate
		
		if fix_coords:
			rois[i][[2]] = np.maximum(rois[i][[2]] - rois[i][[0]],1.) 		# Transform x2 to w
			rois[i][[3]] = np.maximum(rois[i][[3]] - rois[i][[1]],1.) 		# Transform y2 to h
		else:
			rois[i][[2]] = rois[i][[2]] - rois[i][[0]] 		# Transform x2 to w
			rois[i][[3]] = rois[i][[3]] - rois[i][[1]] 		# Transform y2 to h
		
		rois[i] = np.expand_dims(rois[i], axis=0)

#	images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
	return images_data, rois




# Perform data augmentation over an annotation line
# An annotation line can contain more than one frame that will be processed
# 	with the same augmentation parameters
# Return augmented image with its augmented ROI coordinates
def get_random_data_cv2(annotation_line, path_dataset, input_shape, downsample_rate, random=False, fix_coords=True):
	'''random preprocessing for real-time data augmentation'''
	line = annotation_line.split()
	videos = [ '/'.join(img.split('/')[:-1]) for img in line[:3] ]

	# Apply same transformations if anchor and negative belong to the same video
	if videos[0] == videos[2]:
		images, rois = None, None
		while images is None:
			images, rois = preprocess_img_roi_rand(line[:3], line[3:], path_dataset, input_shape, downsample_rate, random, fix_coords)
	else:
		images, rois = None, None
		while images is None:
			images, rois = preprocess_img_roi_rand(line[:2], line[3:5], path_dataset, input_shape, downsample_rate, random, fix_coords)
			
		images_N, rois_N = None, None
		while images_N is None:
			images_N, rois_N = preprocess_img_roi_rand([line[2]], [line[5]], path_dataset, input_shape, downsample_rate, random, fix_coords)
		images += images_N; rois += rois_N
			
	return images + rois



def data_generator(annotation_lines, batch_size, input_shape, downsample_rate, random, shuffle, fix_coords=True):
	'''data generator for fit_generator'''
	n = len(annotation_lines)
	i = 0
	while True:
		batch_data = []
		for b in range(batch_size):
			
			if shuffle and i==0:
				np.random.shuffle(annotation_lines)
			data = get_random_data_cv2(annotation_lines[i], path_dataset, input_shape, downsample_rate, random=random, fix_coords=fix_coords)
			batch_data.append(data)
			i = (i+1) % n
		
		data = [ np.stack([ bd[i] for bd in batch_data ]) for i in range(len(data)) ]
		yield  data, np.zeros((batch_size,64*3))


def data_generator_wrapper(annotation_lines, batch_size, input_shape, 
								  downsample_rate, random, shuffle=True, fix_coords=True, **kwargs):
	n = len(annotation_lines)
	if n==0 or batch_size<=0: return None
	return data_generator(annotation_lines, path_dataset, batch_size, input_shape, downsample_rate, random, shuffle, fix_coords)

