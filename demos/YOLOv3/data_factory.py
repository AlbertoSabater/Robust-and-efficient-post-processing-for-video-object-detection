#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:20:48 2019

@author: asabater
"""

from PIL import Image
import numpy as np

from yolo3.model import preprocess_true_boxes


def rand(a=0, b=1):
	return np.random.rand()*(b-a) + a


import cv2

# Perform data augmentation over an annotation line
# An annotation line can contain more than one frame that will be processed
# 	with the same augmentation parameters
# Images processed with RGB format
def get_random_data_cv2(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
	'''random preprocessing for real-time data augmentation'''
	line = annotation_line.split()

	# numpy array: BGR, 0-255
	images = [ cv2.imread(i) for i in line[0].split(',') ]
	# height, width, channel
	ih, iw, _ = images[0].shape
	h, w = input_shape
	box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

	if not random:
		# resize image
		scale = min(w/iw, h/ih)
		nw = int(iw*scale)
		nh = int(ih*scale)
		dx = (w-nw)//2
		dy = (h-nh)//2
		if proc_img:
			# resize
			new_images, images_data = [None]*len(images), [None]*len(images)
			for i in range(len(images)):
				images[i] = cv2.resize(images[i], (nw, nh), interpolation=cv2.INTER_AREA)
				# convert into PIL Image object
				images[i] = Image.fromarray(images[i][:, :, ::-1])
				new_images[i] = Image.new('RGB', (w,h), (128,128,128))
				new_images[i].paste(images[i], (dx, dy))
				# convert into numpy array: RGB, 0-1
				images_data[i] = np.array(new_images[i])/255.

		# correct boxes
		box_data = np.zeros((max_boxes,5))
		if len(box)>0:
			np.random.shuffle(box)
			if len(box)>max_boxes: box = box[:max_boxes]
			box[:, [0,2]] = box[:, [0,2]]*scale + dx
			box[:, [1,3]] = box[:, [1,3]]*scale + dy
			box_data[:len(box)] = box

		images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
		return images_data, box_data

	# resize image
	new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
	scale = rand(.25, 2)
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
	box_data = np.zeros((max_boxes,5))
	if len(box)>0:
		np.random.shuffle(box)
		box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
		box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
		if h_flip:
			box[:, [0,2]] = w - box[:, [2,0]]

		box[:, 0:2][box[:, 0:2]<0] = 0
		box[:, 2][box[:, 2]>w] = w
		box[:, 3][box[:, 3]>h] = h
		box_w = box[:, 2] - box[:, 0]
		box_h = box[:, 3] - box[:, 1]
		box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
		if len(box)>max_boxes: box = box[:max_boxes]
		box_data[:len(box)] = box

	images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
	return images_data, box_data


def data_generator_custom(annotation_lines, batch_size, input_shape, anchors, 
						  num_classes, random, multi_scale):
	'''data generator for fit_generator'''
	n = len(annotation_lines)
	i = 0
#	valid_img_sizes = [ 32*i for i in range(12,22) ]
	valid_img_sizes = np.arange(320, 608+1, 32)
	print(valid_img_sizes)
	while True:
		image_data = []
		box_data = []
		if multi_scale:
			size = np.random.choice(valid_img_sizes)
			input_shape = [size,size]
			
		for b in range(batch_size):

			if i==0:
				np.random.shuffle(annotation_lines)
			images, box = get_random_data_cv2(annotation_lines[i], input_shape, random=random)
			image_data.append(images)
			box_data.append(box)
			i = (i+1) % n
		image_data = np.array(image_data)
		box_data = np.array(box_data)
		y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
		yield [image_data, *y_true], np.zeros(batch_size)


import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)

    def next(self):     # Py2
        with self.lock:
            return self.it.next()
		
		
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator	
def data_generator_wrapper_custom(annotation_lines, batch_size, input_shape, 
								  anchors, num_classes, random, multi_scale=False):
	n = len(annotation_lines)
	if n==0 or batch_size<=0: return None
	return data_generator_custom(annotation_lines, batch_size, input_shape, 
							  anchors, num_classes, random, multi_scale)


