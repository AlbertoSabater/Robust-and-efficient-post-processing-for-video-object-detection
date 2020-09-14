#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:32:41 2019

@author: asabater
"""

import os
from PIL import Image
from tqdm import tqdm
import sys




path_dataset = '/home/asabater/projects/ADL_dataset/'
annotations_file = 'adl/annotations_adl_train_v2_27.txt'
with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
annotations = [ path_dataset + l for l in annotations ]



new_annotations = []
for ann in tqdm(annotations, total=len(annotations), file=sys.stdout):
	ann = ann.split()
	img = ann[0]
	new_annotations.append(img)
	
	img_labels_file = img.replace('.' + img.split('.')[-1], '.txt')
	img = Image.open(img)
	img_width, img_height= img.size
	
	with open(img_labels_file, 'w') as f:
		boxes = []
		for box in ann[1:]:
			x_min, y_min, x_max, y_max, cat = [ float(b) for b in box.split(',') ]
			width = x_max - x_min
			height = y_max - y_min
			x_center = x_max - width/2
			y_center = y_max - height/2
			x_center, y_center, width, height = x_center/img_width, y_center/img_height, width/img_width, height/img_height
			
			f.write('{:.0f} {} {} {} {}\n'.format(cat, x_center, y_center, width, height))
#			print('{:.0f} {} {} {} {}'.format(cat, x_center, y_center, width, height))		


output_file = annotations_file.replace('.txt', '_darknet.txt')
with open(output_file, 'w') as f:
	for ann in new_annotations:
		f.write(ann + '\n')

print(output_file, 'stored')
