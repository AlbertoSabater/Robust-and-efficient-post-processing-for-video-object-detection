#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:12:38 2019

@author: asabater
"""

import sys
import argparse
from eyolo import EYOLO, detect_video
from PIL import Image

import os

def detect_img(yolo, img, output_path=""):
	print('***', img)
	try:
		image = Image.open(img)
	except:
		print('Open Error! Try again!')
	else:
		image = yolo.detect_image(image)
		if output_path != "": image.save(output_path)
		image.show()
	yolo.close_session()

FLAGS = None

if __name__ == '__main__':
	# class YOLO defines the default value, so suppress any default here
	parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
	parser.add_argument('--model', type=str, required=True, help='path to model weight file')
	parser.add_argument('--anchors', type=str, required=True, help='path to anchor definitions')
	parser.add_argument('--classes', type=str, required=True, help='path to class definitions')
	parser.add_argument("--input", type=str, required=True, help = "Video/image input path")
	parser.add_argument('--image', default=False, action="store_true", help='Image detection mode')
	parser.add_argument('--spp', default=False, action="store_true", help='use this option if the model uses SPP')
	parser.add_argument('--output_path', type=str, default="", help='output file where to store the result')

	FLAGS = parser.parse_args()
	
	model = EYOLO(
				model_image_size = (416, 416),
				model_path = FLAGS.model,
				anchors_path = FLAGS.anchors,
				classes_path = FLAGS.classes,
				score = 0.4,
				iou = 0.5,
				spp = FLAGS.spp
			)

	if FLAGS.image:
		"""
		Image detection mode, disregard any remaining command line arguments
		"""
		print("Image detection mode")
		detect_img(model, FLAGS.input, FLAGS.output_path)
	elif "input" in FLAGS:
		detect_video(model, FLAGS.input, output_path=FLAGS.output_path)
	else:
		print("Must specify at least video_input_path.  See usage with --help.")


