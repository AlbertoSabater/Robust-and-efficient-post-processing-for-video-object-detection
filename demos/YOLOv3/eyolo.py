#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:59:56 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.models import Model

from yolo3.utils import letterbox_image
from yolo import YOLO

from emodel import yolo_body, r_yolo_body, yolo_eval, yolo_eval_raw, yolo_eval_raw_scores, yolo_eval_raw_scores_feats
from yolo3.model import tiny_yolo_body

import cv2
import os
import random
import colorsys

import time
import json



# Load a YOLO model and its weights from a model folder
# Set downsample_rate to extract correspondings feature maps from YOLO
def load_yolo_model_raw(model_folder, path_weights, image_size, scores_vector,
						downsample_rate, score=0.005, iou_thr=0.5, max_boxes=20):

	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	
	if scores_vector:
        # Detection scores as given as a vector of class confidences
		raw_eval = 'raw_scores'
		if downsample_rate is not None: raw_eval += '_{}'.format(downsample_rate)
	else:
        # Each detection has a single score and category
		raw_eval = 'def'
		
	
	model = EYOLO(
			model_image_size = image_size,
			model_path = path_weights,
			classes_path = train_params['path_classes'],
			anchors_path = train_params['path_anchors'],
			score = score,
			iou = iou_thr,
			spp = train_params.get('spp', False),
			raw_eval = raw_eval,
			max_boxes = max_boxes
		)
	
	return model, train_params



class EYOLO(YOLO):
	
	def generate(self):
		model_path = os.path.expanduser(self.model_path)
		assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

		# Load model, or construct model and load weights.
		num_anchors = len(self.anchors)
		num_classes = len(self.class_names)
		is_tiny_version = num_anchors==6 # default setting
		try:
			self.yolo_model = load_model(model_path, compile=False)
		except:
			if is_tiny_version:
				self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
			else:
				self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes, self.spp)
			self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
		else:
			assert self.yolo_model.layers[-1].output_shape[-1] == \
				num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
				'Mismatch between model and given anchor and class sizes'

		print('{} model, anchors, and classes loaded.'.format(model_path))
		
		
		# Add feature map outptus if specified
		if self.raw_eval.startswith('raw_scores_'):
			downsample_rates = list(map(int, self.raw_eval.split('_')[2:]))
			print(downsample_rates)
			extra_outputs = []
			for downsample_rate in downsample_rates:
				if downsample_rate == 32: extra_outputs.append(self.yolo_model.layers[184].output) 			# 13x1054
				elif downsample_rate == 16: extra_outputs.append(self.yolo_model.layers[152].output) 		# 26x512
				elif downsample_rate == 8: extra_outputs.append(self.yolo_model.layers[92].output)			# 52x256
				elif downsample_rate == 4: extra_outputs.append(self.yolo_model.layers[32].output)			# 104x128
				elif downsample_rate == 2: extra_outputs.append(self.yolo_model.layers[14].output)			# 208x64
				elif downsample_rate == 1: extra_outputs.append(self.yolo_model.layers[3].output)			# 416x32
				elif downsample_rate == 0: extra_outputs.append(self.yolo_model.layers[0].output)			# 413x3			
				else: raise ValueError('downsample_rate not valid: {}'.format(downsample_rate))
				
			print('Extra outputs:')
			for eo in extra_outputs: print(' -', eo)
			self.yolo_model = Model(self.yolo_model.inputs, self.yolo_model.outputs + extra_outputs)
		

		# Generate colors for drawing bounding boxes.
		hsv_tuples = [(x / len(self.class_names), 1., 1.)
					  for x in range(len(self.class_names))]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		self.colors = list(
			map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
				self.colors))
		np.random.seed(10101)  # Fixed seed for consistent colors across runs.
		np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
		np.random.seed(None)  # Reset seed to default.

		# Generate output tensor targets for filtered bounding boxes.
		self.input_image_shape = K.placeholder(shape=(2, ))
		if self.gpu_num>=2:
			self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
			

		if not hasattr(self, 'raw_eval') or self.raw_eval == 'def':   # Each detection has a single score and category
			return yolo_eval(self.yolo_model.output, self.anchors,
					len(self.class_names), self.input_image_shape,
					score_threshold=self.score, iou_threshold=self.iou)
		elif self.raw_eval == 'raw':
			return yolo_eval_raw(self.yolo_model.output, self.anchors,
					len(self.class_names), self.input_image_shape,
					score_threshold=self.score)
# 		elif self.raw_eval == 'raw_scores':
# 			return yolo_eval_raw_scores(self.yolo_model.output, self.anchors,
# 					len(self.class_names), self.input_image_shape,
# 					score_threshold=self.score)
# 		elif 'raw_scores_' in self.raw_eval: 		# ej. raw_scores_16_32
		elif 'raw_scores' in self.raw_eval: 		# ej. raw_scores_16_32  # Detection scores as given as a vector of class confidences
			return yolo_eval_raw_scores_feats(self.yolo_model.output, self.anchors,
					len(self.class_names), self.input_image_shape,
					list(map(int, self.raw_eval.split('_')[2:])),
					max_boxes = self.max_boxes,
					score_threshold=self.score, 
					iou_threshold=self.iou)
		else: raise ValueError('Unrecognized eval error')
		
#		return boxes, scores, classes
	
	
	def get_prediction(self, image):
		if self.model_image_size != (None, None):
			assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
			assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
			if type(image) is list:
				if len(image) > 1:
					img_size = image[0].size
					boxed_image = np.stack([ letterbox_image(image, tuple(reversed(self.model_image_size))) for image in image  ])
				else:
					img_size = image[0].size
					boxed_image = letterbox_image(image[0], tuple(reversed(self.model_image_size)))
			else:
				img_size = image.size
				boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
		else:
			new_image_size = (image.width - (image.width % 32),
							  image.height - (image.height % 32))
			boxed_image = letterbox_image(image, new_image_size)
		image_data = np.array(boxed_image, dtype='float32')

#		print('nn_input', image_data.shape)
		image_data /= 255.
		image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
		
		if hasattr(self, 'raw_eval') and self.raw_eval == 'raw_scores':
			return self.sess.run(
				[self.boxes, self.scores],
				feed_dict={
					self.yolo_model.input: image_data,
					self.input_image_shape: [img_size[1], img_size[0]],
					K.learning_phase(): 0
				})
		else:
			return self.sess.run(
				[self.boxes, self.scores, self.classes],
				feed_dict={
					self.yolo_model.input: image_data,
					self.input_image_shape: [img_size[1], img_size[0]],
					K.learning_phase(): 0
				})
	
#		return out_boxes, out_scores, out_classes
	
	
	def detect_image(self, image):
		boxes, scores, classes = self.get_prediction(image)
		image = self.print_boxes(image, boxes, classes, scores, color=(0,0,255))
		return image
	
	
	def print_boxes(self, image, boxes, classes, scores=None, color=None, label_size=0.5):
		for i, c in reversed(list(enumerate(classes))):
			predicted_class = self.class_names[c]
			box = boxes[i]
#			score = '' if scores is None else scores[i]
			color_c = self.colors[c] if color is None else color
			
			if scores is None:
				label = '{}'.format(predicted_class)
			else:
				label = '{} {:.2f}'.format(predicted_class, scores[i])
			
			image = print_box(image, box, label, color_c, label_size)		
			
		return image
	
	
def print_box(image, box, label, color, label_size=0.5):
	
	font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
				size=np.floor(3e-2 * image.size[1] + label_size).astype('int32'))
	
	draw = ImageDraw.Draw(image)
	label_size = draw.textsize(label, font)

	top, left, bottom, right = box
	top = max(0, np.floor(top + 0.5).astype('int32'))
	left = max(0, np.floor(left + 0.5).astype('int32'))
	bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
	right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#			print(label, (left, top), (right, bottom))

	if top - label_size[1] >= 0:
		text_origin = np.array([left, top - label_size[1]])
	else:
		text_origin = np.array([left, top + 1])

	# My kingdom for a good redistributable image drawing library.
	for i in range((image.size[0] + image.size[1]) // 300):
		draw.rectangle(
			[left + i, top + i, right - i, bottom - i],
			outline=color)
	draw.rectangle(
		[tuple(text_origin), tuple(text_origin + label_size)],
		fill=color)
	draw.text(text_origin, label, fill=(0, 0, 0), font=font)
	del draw
	
	return image


def detect_video(yolo, video_path, output_path="", close_session=True):
	vid = cv2.VideoCapture(video_path)
	if not vid.isOpened():
		raise IOError("Couldn't open webcam or video")
	video_FourCC	= int(vid.get(cv2.CAP_PROP_FOURCC))
	video_fps	   = vid.get(cv2.CAP_PROP_FPS)
	video_size	  = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
						int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	isOutput = True if output_path != "" else False
	if isOutput:
		from skvideo.io import FFmpegWriter
#		print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
#		out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
		out = FFmpegWriter(output_path, inputdict={'-r': str(video_fps)}, outputdict={'-r': str(video_fps)})
		
	accum_time = 0
	curr_fps = 0
	fps = "FPS: ??"
	prev_time = timer()
	while True:
		return_value, frame = vid.read()
		
		if frame is None: 
			break
	
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(frame)
		image = yolo.detect_image(image)
		result = np.asarray(image)
		result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
		
		curr_time = timer()
		exec_time = curr_time - prev_time
		prev_time = curr_time
		accum_time = accum_time + exec_time
		curr_fps = curr_fps + 1
		if accum_time > 1:
			accum_time = accum_time - 1
			fps = "FPS: " + str(curr_fps)
			curr_fps = 0
		cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=0.50, color=(255, 0, 0), thickness=2)
		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", result)
		if isOutput:
#			print(type(result), result.shape)
#			out.writeFrame(image)
			out.writeFrame(image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	if close_session: yolo.close_session()
	
	

def detect_video_folder(yolo, video_folder, wk=1):
	frames = os.listdir(video_folder)
	prev_time = timer()

	accum_time = 0
	curr_fps = 0
	fps = "FPS: ??"
	
	for fr in frames:
		
#		image = cv2.imread(video_folder + fr)
#		image = Image.fromarray(image)
		image = Image.open(video_folder + fr)
		image, boxes, scores, classes = yolo.detect_image(image)
		image = yolo.print_boxes(image, boxes, classes, scores)
		result = np.asarray(image)
		
		curr_time = timer()
		exec_time = curr_time - prev_time
		prev_time = curr_time
		accum_time = accum_time + exec_time
		curr_fps = curr_fps + 1
		if accum_time > 1:
			accum_time = accum_time - 1
			fps = "FPS: " + str(curr_fps)
			curr_fps = 0
		cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=0.50, color=(255, 0, 0), thickness=2)
		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", result)
		if cv2.waitKey(wk) & 0xFF == ord('q'):
			break

	return result
	

def predict_annotations(model, annotations, path_base, wk):
	with open(annotations, 'r') as f: annotations = f.read().splitlines()
	annotations = annotations[random.randint(0, len(annotations)):]
	
	prev_time = timer()
	accum_time = 0
	curr_fps = 0
	fps = "FPS: ??"
	
	print(annotations[1])
	
	for l in annotations:
		t = time.time()
		
		l = l.split()
		img = l[0]
		boxes = [ [int(b) for b in bb.split(',') ] for bb in l[1:] ]
		classes = [ bb[-1] for bb in boxes ]
		boxes = [ bb[:-1] for bb in boxes ]
		boxes = [ [bb[1],bb[0],bb[3],bb[2]] for bb in boxes ]
		
#		image = cv2.imread(path_base + img)
#		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#		image = Image.fromarray(image)	
		images = [ Image.open(path_base + img) for img in img.split(',') ]
		image = model.print_boxes(images[len(images)//2], boxes, classes, color=(0,255,0))

		boxes, scores, classes = model.get_prediction(images)
		image = model.print_boxes(image, boxes, classes, scores, color=(0,0,255))
	
		result = np.asarray(image)
		result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	
		curr_time = timer()
		exec_time = curr_time - prev_time
		prev_time = curr_time
		accum_time = accum_time + exec_time
		curr_fps = curr_fps + 1
		if accum_time > 1:
			accum_time = accum_time - 1
			fps = "FPS: " + str(curr_fps)
			curr_fps = 0
		
		cv2.putText(result, text=fps + " | {}".format(img.split('/')[-2]), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=0.75, color=(255, 0, 0), thickness=2)

		result = cv2.resize(result, (result.shape[1]//2, result.shape[0]//2))
#		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", result)
		if cv2.waitKey(max(wk - int(((time.time() - t) * 1000)), 1)) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	cv2.destroyAllWindows()
	
	