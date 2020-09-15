#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:01:33 2020

@author: asabater
"""

import os
from xml.dom import minidom
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import pandas as pd
import numpy as np
import argparse

from data_annotations.imagenet_vid_classes import classes_map

np.random.seed(0)


# =============================================================================
# Creates a set of triplet annotations (Anchor, Positive, Negative) from
# ILSVRC original annotations
# These annotations are used to train the Logistic Regression REPP linking model
# and optionally the appearance embedding descriptor model
# =============================================================================


N_PERC_CHANGE_TO_VIDEO = 0.1
N_PERC_CHANGE_TO_TRACK = 0.1


# Reads and parses Imagenet VID metadata
def get_video_bndbox(annotations_dir, video, frame, mode):
	doc =  minidom.parse(annotations_dir + '/' + video + '/' + frame)
	
	filefolder = doc.getElementsByTagName('folder')[0].firstChild.data
	filename = doc.getElementsByTagName('filename')[0].firstChild.data
	
	size = doc.getElementsByTagName('size')[0]
	width = size.getElementsByTagName('width')[0].firstChild.data
	height = size.getElementsByTagName('height')[0].firstChild.data
	
	objs = doc.getElementsByTagName("object")
	
	bboxes = []
	for o in objs:
		
		name = o.getElementsByTagName('name')[0].firstChild.data
		track_id = o.getElementsByTagName('trackid')[0].firstChild.data
		
		bndbox = o.getElementsByTagName('bndbox')[0]
		xmax = bndbox.getElementsByTagName('xmax')[0].firstChild.data
		xmin = bndbox.getElementsByTagName('xmin')[0].firstChild.data
		ymax = bndbox.getElementsByTagName('ymax')[0].firstChild.data
		ymin = bndbox.getElementsByTagName('ymin')[0].firstChild.data

		bboxes.append({
					'img': mode + '/' + filefolder + '/' + filename + '.JPEG',
					'num_frame': int(filename),
					'xmax': xmax,
					'xmin': xmin,
					'ymax': ymax,
					'ymin': ymin,		 
					'class': classes_map.index(name),
					'track_id': filefolder + '/' + track_id,
					'video': filefolder,
					'width': width,
					'height': height,
				})
			
	return bboxes


def get_folder_bndbox(annotations_dir, video, mode):
	frames = os.listdir(annotations_dir + '/' + video)
	return sum([ get_video_bndbox(annotations_dir, video, frame, mode) for frame in frames ], [])


def get_bboxes_val(annotations_dir):
	bboxes = []
	videos = os.listdir(annotations_dir)
	bboxes = sum(Parallel(n_jobs=8)(delayed(get_folder_bndbox)(annotations_dir, video, 'val') 
						for video in tqdm(videos, total=len(videos), file=sys.stdout)), [])
	return bboxes

def get_bboxes_train(annotations_dir):
	sets = os.listdir(annotations_dir)
	
	bboxes = []
	for s in sets:
	
		videos = os.listdir(annotations_dir + s)
		
		bboxes += sum(Parallel(n_jobs=8)(delayed(get_folder_bndbox)(annotations_dir + '/' + s, video, 'train') 
						for video in tqdm(videos, total=len(videos), file=sys.stdout)), [])
		
	return bboxes
	


# Generates random triplet annotations
def get_annotations(bboxes, path_dataset, mode, num_samples, max_frame_dist):
	bboxes_df = pd.DataFrame(bboxes)

	# Each triplet sample contains:
	# 	Anchor from a random track_id
	#	Positive from the same track_id
	#	Negative
	#	 	From the Anchor frame
	#	 	From a track_id
	# 	 	Frrom a random video
	
	# Annotations format -> imgA, imgN, imgP, bboxA, bboxN, bboxP
	
	samples = []
	track_ids = bboxes_df.track_id.drop_duplicates().tolist()
	empty_P, empty_N = [], []
	
	def get_samples():
		track_A = np.random.choice(track_ids)
		
		# random track_id
		sample_A = bboxes_df[bboxes_df.track_id == track_A].sample(1)
		
		img_A = sample_A.img.iloc[0]
		num_frame_A = sample_A.num_frame.iloc[0]
		
		# Positive: same track, close frame to A, different from A
		opts_P = bboxes_df[(bboxes_df.track_id == track_A) \
					 & (bboxes_df.num_frame > num_frame_A-max_frame_dist)\
					 & (bboxes_df.num_frame < num_frame_A+max_frame_dist)\
					 & (bboxes_df.img != img_A) ]
		if len(opts_P) == 0: 
			empty_P.append(track_A)
			return None
		else: sample_P = opts_P.sample(1)
		
		if np.random.rand() > N_PERC_CHANGE_TO_VIDEO:
			# Negative: same frame, different track
			opts_N = bboxes_df[(bboxes_df.img == img_A) & (bboxes_df.track_id != track_A)]
		else: opts_N = []
		if len(opts_N) == 0:
			
			if np.random.rand() > N_PERC_CHANGE_TO_TRACK:
				# Negative from same video, different track
				video_A = sample_A.video.iloc[0]
				opts_N = bboxes_df[(bboxes_df.video == video_A) & (bboxes_df.track_id != track_A)]
			else: opts_N = []
			
			if len(opts_N) == 0:
				# Negative: different track
				empty_N.append(img_A)
				while True:
					new_track = np.random.choice(track_ids)
					if new_track != track_A: break
				opts_N = bboxes_df[(bboxes_df.track_id == new_track)]
				
		sample_N = opts_N.sample(1)
		
		return (sample_A, sample_P, sample_N)
	
	
	samples = []
	count = 0
	pbar = tqdm(total=num_samples)
	while count < num_samples:
		sample = get_samples()
		if sample is not None: 
			samples.append(sample); count += 1; pbar.update(1)
		else:
			pass


	anns = []
	for sample in samples:
		if sample is None: continue
		ann = " ".join([ path_dataset + 'Data/VID/' + s.img.iloc[0] for s in sample ])
		ann += " " + " ".join([ "{},{},{},{}".format(s.xmin.iloc[0], s.ymin.iloc[0], s.xmax.iloc[0], s.ymax.iloc[0]) for s in sample ])	
		anns.append(ann)


	filename = './data_annotations/triplet_annotations/triplet_annotations_{}.txt'.format(mode)
	with open(filename, 'w') as f:
		for ann in anns: f.write(ann + '\n')
	
	print('saved:', filename)	
	
	return bboxes_df, anns




def main():
	parser = argparse.ArgumentParser(description='Creates the set of triplet annotations.')
	parser.add_argument('--path_dataset', help='path of the Imagenet VID dataset', type=str)
	parser.add_argument('--num_samples_train', help='number of train samples', default=50000, type=int)
	parser.add_argument('--num_samples_val', help='number of validation samples', default=8000, type=int)
	parser.add_argument('--max_frame_dist', help='frame distance between anchor and positive samples', default=25, type=int)
    
	args = parser.parse_args()
		
		
	
	annotations_dir = args.path_dataset + '/Annotations/VID/{}/'
	mode = 'val'
	print('Reading validation data')
	bboxes_val = get_bboxes_val(annotations_dir.format(mode))
	print('='*80)
	mode = 'train'
	print('Reading train data')
	bboxes_train = get_bboxes_train(annotations_dir.format(mode))
	print('='*80)
	
	
	np.random.seed(0)
	
	mode = 'train'
	print('Generating train triplets annotations')
	bboxes_df_train, anns_train = get_annotations(bboxes_train, path_dataset='', 
												  mode=mode, num_samples=args.num_samples_train, 
												  max_frame_dist=args.max_frame_dist)
	
	mode = 'val'
	print('Generating validation triplets annotations')
	bboxes_df_val, anns_val = get_annotations(bboxes_val, path_dataset='', 
											  mode=mode, num_samples=args.num_samples_val, 
											  max_frame_dist=args.max_frame_dist)


if __name__ == '__main__':
	main()

