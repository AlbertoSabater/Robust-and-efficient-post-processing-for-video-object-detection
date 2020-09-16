#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:57:25 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

from kmeans import YOLO_Kmeans
import numpy as np
from tqdm import tqdm
import cv2
import argparse


class EYOLO_Kmeans(YOLO_Kmeans):

	def __init__(self, cluster_number, filename, path_dataset, output_shape):
		self.cluster_number = cluster_number
		self.filename = filename
		self.path_dataset = path_dataset
		self.output_shape = output_shape
		
		
	def txt2boxes(self):
		f = open(self.filename, 'r')
		dataSet = []
		lines = f.read().splitlines()
		print('Loading annotations...')
		for line in tqdm(lines, total=len(lines), file=sys.stdout):
			infos = line.split()
			length = len(infos)
			
			img  = cv2.imread(self.path_dataset + infos[0])
			scale = max(self.output_shape) / max(img.shape[:2])
			
			for i in range(1, length):
				width = (int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])) * scale
				height = (int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])) * scale
				dataSet.append([width, height])
				
		result = np.array(dataSet)
		f.close()
		return result
	
	def get_best_anchors(self):
		all_boxes = self.txt2boxes()
		num_steps = 20
		best_iou, best_result = 0, None
		print('Calculating optimal anchors')
		for i in tqdm(range(num_steps), total=num_steps, file=sys.stdout):
			result = self.kmeans(all_boxes, k=self.cluster_number)
			result = result[np.lexsort(result.T[0, None])]
			iou = self.avg_iou(all_boxes, result) * 100
			if iou > best_iou:
				best_iou = iou
				best_result = result
				
		print("Avg. IOU: {:.2f}%".format(self.avg_iou(all_boxes, best_result) * 100))
		return best_result		
		
	def result2txt(self, anchors, output_filename):
		anchors_txt = ', '.join([ '{:.2f},{:.2f}'.format(*an) for an in anchors ])
		with open(output_filename, 'w') as f:
			f.write(anchors_txt)
		
		print(output_filename, 'stored')
		print(anchors_txt)
		

def main():
	
	description = "Script to calculate custom anchors from an annotations file"
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("path_annotations", help="annotations file")
	parser.add_argument("store_path", help="anchors output filename")
	parser.add_argument("--path_dataset", help="path to each training image if not specified in annotations file", default='', type=str)
	parser.add_argument("--num_clusters", help="number of clusters to obtain", default=9, type=int)
	parser.add_argument("--output_shape", help="size of the final image input shape", default=416, type=int)
	args = parser.parse_args()
	
	kmeans = EYOLO_Kmeans(args.num_clusters, args.path_annotations, args.path_dataset, (args.output_shape, args.output_shape))
	anchors = kmeans.get_best_anchors()
	kmeans.result2txt(anchors, args.store_path)


if __name__ == "__main__":
	main()



