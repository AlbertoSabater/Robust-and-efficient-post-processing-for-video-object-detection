#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:59:56 2020

@author: asabater
"""

import pickle
from tqdm import tqdm
import json


#preds_filename = '/mnt/hdd/egocentric_results/ilsvrc/FGFA/predictions_fgfa.pckl'
#imageset_filename = '/home/asabater/projects/Egocentric_object_detection/dataset_scripts/ilsvrc/skms/annotations_val_skms-1_mvl2_image_set.txt'
#res = pickle.load(open(preds_filename, 'rb'), encoding='bytes')
path_dataset = '/home/asabater/projects/ILSVRC2015/Data/VID/'



def transform_selsa_results(preds_filename, res, path_dataset):
    preds_frame = res[0][0][0]
    image_ids = res[0][1]
    
    #imageset_filename = './data/ILSVRC/ImageSets/VID_val_frames.txt'
    with open(path_dataset + '../../ImageSets/VID_val_videos_eval.txt', 'r') as f: frame_data = f.read().splitlines()
    
    
    # =============================================================================
    # Store predictions by video along with scores vector to post-processing
    # =============================================================================
    
    imageset_video_frame = {}
    
    for frame in frame_data:
    	frame, i = frame.split()
    	video = '/'.join(frame.split('/')[:-1])
    	if video not in imageset_video_frame: imageset_video_frame[video] = {}
    	
    	imageset_video_frame[video][i] = frame
    
    
    import pandas as pd
    
    imageset = {}
    for frame in frame_data:
    	frame, i = frame.split()
    	imageset[i] = frame
    
    frame_image_ids = [ imageset[str(i)] for i in image_ids ]
    video_image_ids = [ '/'.join(imageset[str(i)].split('/')[:-1]) for i in image_ids ]
    imageset = pd.DataFrame({'pred_id': image_ids, 'frame': frame_image_ids, 
    						 'video': video_image_ids, 'preds': preds_frame})
    
    
    from PIL import Image
    
    #store_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/preds_raw_scores_aug_score{}.pckl'.format(min_score)
    store_filename = preds_filename.replace('.pckl', '_repp.pckl').replace('_raw_', '_').replace('/res', '/preds')
    store_file = open(store_filename, 'wb')
    
    for vid, g in tqdm(imageset.groupby('video')):
    	
    	preds_video_frame = {}
    	for ind, r in g.iterrows():
    		
    		preds = r.preds
    #		preds = preds_frame[r.pred_id-1]
    		image_id = r.frame
    		frame_num = image_id.split('/')[-1]
    	
    		image = Image.open(path_dataset + 'val/' + image_id + '.JPEG')
    		img_size = image.size
    		ih, iw = img_size[::-1]
    		width_diff = max(0, (ih-iw)//2)
    		height_diff = max(0, (iw-ih)//2)
    		
    		for pred in preds:
    			bbox = pred[:4]
    			scores = pred[4:]
    #			x1,y1,x2,y2 = bbox		
    
    #			y_min, x_min, y_max, x_max = bbox
    			x_min, y_min, x_max, y_max = bbox
    			y_min, x_min = max(0, y_min), max(0, x_min)
    			y_max, x_max = min(img_size[1], y_max), min(img_size[0], x_max)
    			width = x_max - x_min
    			height = y_max - y_min
    			if width <= 0 or height <= 0: continue
    		
    			bbox_center = [ (x_min + width_diff + width/2)/max(iw,ih),
    							  (y_min + height_diff + height/2)/max(iw,ih)]
    			
    #			if scores.max() < min_score: continue
    		
    			pred = {
    					'image_id': 'val/' + image_id,
    					'bbox': [ x_min, y_min, width, height ],
    					'scores': scores,
    					'bbox_center': bbox_center
    					}
    
    			if frame_num in preds_video_frame: preds_video_frame[frame_num].append(pred)
    			else: preds_video_frame[frame_num] = [pred]
    #			print(bbox)
    
    	preds_video_frame = {k: preds_video_frame[k] for k in sorted(preds_video_frame)}
    	pickle.dump(('val/' + vid, preds_video_frame), store_file)
    
    store_file.close()			
    print('Stream predictions stored:', store_filename)
