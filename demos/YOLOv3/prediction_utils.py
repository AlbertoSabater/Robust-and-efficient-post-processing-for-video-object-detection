#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:24:20 2019

@author: asabater
"""

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = "1";  

from eyolo import EYOLO
from PIL import Image
from test_post_processing import seq_nms_fgfa

from tqdm import tqdm
import os
import train_utils
import time
import json
import sys
import pandas as pd
import numpy as np
import pickle
import keras



# Stores an json file with the predictions calculated by the given annotations_file
# Uses best weights or last calculated weights depending on best_weights
def predict_and_store_from_annotations(model_folder, train_params, annotations_file, 
									   output_dir, image_size, score, nms_iou, best_weights=True,
									   raw_eval='def', load=False, model_path=None):
	
	if raw_eval == 'raw_scores':
		preds_filename = '{}preds_{}_{}_is{}_score{}_raw_score.json'.format(output_dir, 
					   'bw' if best_weights else 'stage2',
					   annotations_file.split('/')[-1][:-4], 
					   image_size[0], score, nms_iou)
	elif raw_eval == 'raw':
		preds_filename = '{}preds_{}_{}_is{}_score{}_raw.json'.format(output_dir, 
					   'bw' if best_weights else 'stage2',
					   annotations_file.split('/')[-1][:-4], 
					   image_size[0], score)
	elif 'def':
		preds_filename = '{}preds_{}_{}_is{}_score{}_iou{}.json'.format(output_dir, 
					   'bw' if best_weights else 'stage2',
					   annotations_file.split('/')[-1][:-4], 
					   image_size[0], score, nms_iou)
	else: raise ValueError('Unrecognized eval error')
		
	
	if os.path.isfile(preds_filename): 
		print(' * Loading:', preds_filename)
		if load: return json.load(open(preds_filename, 'r')), preds_filename
		else: return None, preds_filename
	
	print(' * Predicting:', preds_filename)
	
	if model_path is None:
		if best_weights:
			model_path = train_utils.get_best_weights(model_folder)
		else:
			model_path = model_folder + 'weights/trained_weights_final.h5'
	
	model = EYOLO(
					model_image_size = image_size,
					model_path = model_path,
					classes_path = train_params['path_classes'],
					anchors_path = train_params['path_anchors'],
					score = score,
					iou = nms_iou,	  # 0.5
					td_len = train_params.get('td_len', None),
					mode = train_params.get('mode', None),
					spp = train_params.get('spp', False),
					raw_eval = raw_eval
				)
	
	
	# Create pred file
	with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
	preds = []
	
	prediction_time = 0
	total = len(annotations)
	total_prediction_time = time.time()
	for ann in tqdm(annotations[:total], total=total, file=sys.stdout):
		img = ann.split()[0]
		
#		image = cv2.imread(train_params['path_dataset'] + img)
#		image = Image.fromarray(image)
		images = [ Image.open(train_params['path_dataset'] + img) for img in img.split(',') ]
		img_size = images[0].size
		max_size = max(img_size[0], img_size[1])
#		images = images[0] if len(images) == 1 else np.stack(images)
		t = time.time()
		boxes, scores, classes = model.get_prediction(images)
		prediction_time += time.time() - t
		
		for i in range(len(boxes)):
			left, bottom, right, top = [ min(int(b), max_size)  for b in boxes[i].tolist() ]
			left, bottom = max(0, left), max(0, bottom)
			right, top = min(img_size[1], right), min(img_size[0], top)
			width = top-bottom
			height = right-left
			preds.append({
							'image_id': '.'.join(img.split('.')[:-1]),
							'category_id': int(classes[i]),
#							'bbox': [ x_min, y_min, width, height ],
							'bbox': [ bottom, left, width, height ],
							'score': float(scores[i]),
						})
	
	print('Total prediction time: ', time.time()-total_prediction_time)
	print('Prediction time: {} secs || {:.2f}'.format(prediction_time, prediction_time/len(annotations)))
	
#	keras.backend.clear_session()
	model.yolo_model = None
	del model.yolo_model
	model = None
	del model
	
#	fps = len(annotations)/(time.time()-t)
	json.dump(preds, open(preds_filename, 'w'))
	
	if load: return preds, preds_filename
	else: return None, preds_filename


def filter_predictions_by_score(preds_filename, filter_out_score, load=False, force=False):

	out_filename = preds_filename.replace(
			[ s for s in preds_filename.split('_') if s.startswith('score') ][0], 'score{}'.format(filter_out_score))

	if force and os.path.isfile(out_filename): os.remove(out_filename)

	if os.path.isfile(out_filename): 
		print(' * Loading:', preds_filename)
		if load: return json.load(open(preds_filename, 'r')), out_filename
		else: return None, out_filename

	preds = json.load(open(preds_filename, 'r'))
	preds = [ p for p in preds if p['score'] >= filter_out_score ]
	
	json.dump(preds, open(out_filename, 'w'))

	if load: return preds, out_filename
	else: return None, out_filename
	

# def get_model_seq_nms_predictions(annotations_file_raw, class_names, preds_filename_raw, max_images, nms_score, load=False):

# 	preds_filename_post = preds_filename_raw.replace('.json', '_seq_nms_mi{}_s{}.json'.format(max_images, nms_score))
# 	if os.path.isfile(preds_filename_post):
# 		print(preds_filename_post, 'already stored')
# 		if load: return json.load(open(preds_filename_post, 'r')), preds_filename_post
# 		else: return None, preds_filename_post
# 	
# 	else:

# 		print(' * Processing:', preds_filename_post)

# 		
# 		preds_raw = json.load(open(preds_filename_raw, 'r'))
# 		preds_df = pd.DataFrame(preds_raw)

# 		if annotations_file_raw != '':
# 			with open(annotations_file_raw, 'r') as f: annotations = f.read().splitlines()
# 			annotations = pd.DataFrame([ '.'.join(ann.split()[0].split('.')[:-1]) for ann in annotations ], columns=['image_id'])
# 			preds_df = preds_df.merge(annotations, how='outer')
# 	
# 		preds_df['video'] = preds_df['image_id'].apply(lambda x: '/'.join(x.split('/')[:-1]))
# 		preds_df = preds_df.sort_values(by=['image_id'])
# 		videos = preds_df['video'].drop_duplicates().tolist()
# 	
# 		time_seq_nms = 0	
# 		total_preds = []
# 		
# 		#video = videos[1]
# 		for video in tqdm(videos, total=len(videos), file=sys.stdout):
# 		
# 			video_dets_all = preds_df[preds_df['video'] == video]
# 			video_image_ids_all = video_dets_all.image_id.sort_values().drop_duplicates().tolist()
# 			if max_images <= 0: max_images = len(video_image_ids_all)
# 			
# #			print('num_frames:', len(video_image_ids_all))
# #			for row_ind in tqdm(range(0,len(video_dets_all), max_images), total=len(video_dets_all)//max_images):
# 			for ids_init in range(0, len(video_image_ids_all), max_images):	
# #			for ids_init in tqdm(range(0, len(video_image_ids_all), max_images), total=len(video_image_ids_all)//max_images, file=sys.stdout):	
# 				
# 				if max_images == -1:
# 					video_dets = video_dets_all[video_dets_all.image_id.isin(video_image_ids_all)]
# 				else:
# 					video_dets = video_dets_all[video_dets_all.image_id.isin(video_image_ids_all[ids_init: ids_init+max_images])]
# #				video_dets = video_dets_all.iloc[row_ind:row_ind+max_images, :]
# 			
# 				video_image_ids = video_dets.image_id.sort_values().drop_duplicates().tolist()
# 				frame_num = video_dets.image_id.drop_duplicates().size
# 				
# 				
# 				dets_all = [ [ [] for _ in range(frame_num) ] for _ in class_names ]
# 				
# 				for i,r in video_dets.iterrows():
# 					if r.hasnans: continue
# 					else:
# 						x,y,w,h = r.bbox
# 						dets_all[int(r.category_id)][video_image_ids.index(r.image_id)].append([x,y,x+w,y+h, r.score])
# 				
# 				t = time.time()
# 				dets_all = [ [ np.array(dets_all[c][f]) if len(dets_all[c][f])>0 else np.zeros((0,4)) for f in range(frame_num) ] for c in range(len(class_names)) ]
# 				links = seq_nms_fgfa.createLinks(dets_all, len(class_names))
# 				dets = seq_nms_fgfa.maxPath(dets_all, links, nms_score)
# 				
# 				for c in range(len(class_names)):
# 					for f in range(frame_num):
# 						boxes = dets[c][f]
# 						for i in range(boxes.shape[0]):
# 							x1,y1,x2,y2,s = boxes[i,:]
# 							total_preds.append({
# 													'image_id': video_image_ids[f],
# 													'category_id': c,
# 				#									'bbox': [ bottom, left, width, height ],
# 													'bbox': [ x1, y1, x2-x1, y2-y1 ],
# 													'score': s,
# 												})
# 				time_seq_nms += time.time() -t
# 		
# 		print('Seq-NMS time: {} secs || {:.2f} fps'.format(time_seq_nms, len(annotations)//time_seq_nms))
# 	
# 		json.dump(total_preds, open(preds_filename_post, 'w'))
# 		print(preds_filename_post, 'stored')
# 	
# 	
# #	eval_filename_post = preds_filename_post.replace('preds', 'stats')
# #	eval_stats_post, videos = evaluate_model.get_full_evaluation(annotations_file, preds_filename_post, 
# #														eval_filename_post, class_names, full=False)
# 		
# 		if load: total_preds, preds_filename_post
# 		else: return None, preds_filename_post

# # TODO: 
# def filter_out_seq_nms_predictions(preds_seq_nms_filename, annotations_file, load=False):
# 	
# 	preds_filename_output = preds_seq_nms_filename.replace('.json', '_short.json')
# 	if os.path.isfile(preds_filename_output):
# 		print(' * Loading:', preds_filename_output, 'stored')
# 		if load: return json.load(open(preds_filename_output, 'r')), preds_filename_output
# 		return None, preds_filename_output
# 	
# 	with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
# 	image_ids = [ '.'.join(ann.split()[0].split('.')[:-1]) for ann in annotations ]
# 	
# 	preds = json.load(open(preds_seq_nms_filename, 'r'))
# 	preds_short = [ p for p in preds if p['image_id'] in image_ids ]
# 	
# 	json.dump(preds_short, open(preds_filename_output, 'w'))
# 	print(' *', preds_filename_output, 'stored')
# 	
# 	if load: preds_short, preds_filename_output
# 	else: return None, preds_filename_output
# 	

# # %%
# 	
# # =============================================================================
# # Seq-Bbox Matching
# # =============================================================================

# def nms(dets, scores, thresh):
# 	"""Pure Python NMS baseline."""
# 	x1 = dets[:, 0]
# 	y1 = dets[:, 1]
# 	x2 = dets[:, 2]
# 	y2 = dets[:, 3]
# #	scores = dets[:, 4]
# 	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
# 	order = scores.argsort()[::-1]
# 	keep = []
# 	while order.size > 0:
# 		i = order[0]
# 		keep.append(i)
# 		xx1 = np.maximum(x1[i], x1[order[1:]])
# 		yy1 = np.maximum(y1[i], y1[order[1:]])
# 		xx2 = np.minimum(x2[i], x2[order[1:]])
# 		yy2 = np.minimum(y2[i], y2[order[1:]])
# 		w = np.maximum(0.0, xx2 - xx1 + 1)
# 		h = np.maximum(0.0, yy2 - yy1 + 1)
# 		inter = w * h
# 		ovr = inter / (areas[i] + areas[order[1:]] - inter)
# 		inds = np.where(ovr <= thresh)[0]
# 		order = order[inds + 1]
# 	return keep

# #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
# def get_iou(bb1, bb2):
#     """
#     Calculate the Intersection over Union (IoU) of two bounding boxes.

#     Parameters
#     ----------
#     bb1 : dict
#         Keys: {'x1', 'x2', 'y1', 'y2'}
#         The (x1, y1) position is at the top left corner,
#         the (x2, y2) position is at the bottom right corner
#     bb2 : dict
#         Keys: {'x1', 'x2', 'y1', 'y2'}
#         The (x, y) position is at the top left corner,
#         the (x2, y2) position is at the bottom right corner

#     Returns
#     -------
#     float
#         in [0, 1]
#     """
#     assert bb1[0] < bb1[2]
#     assert bb1[1] < bb1[3]
#     assert bb2[0] < bb2[2]
#     assert bb2[1] < bb2[3]

#     # determine the coordinates of the intersection rectangle
#     x_left = max(bb1[0], bb2[0])
#     y_top = max(bb1[1], bb2[1])
#     x_right = min(bb1[2], bb2[2])
#     y_bottom = min(bb1[3], bb2[3])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # compute the area of both AABBs
#     bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
#     bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
#     assert iou >= 0.0
#     assert iou <= 1.0
#     return iou

# def distance_def(box_a, box_b, scores_a, scores_b):
# 	iou = get_iou(box_a, box_b)
# 	score = np.dot(scores_a, scores_b)
# 	div = iou * score
# 	if div == 0: return np.inf
# 	return 1 / div

# def get_pairs_matching(boxes_a, boxes_b, scores_a, scores_b, distance_func, pair_iou):

# 	num_boxes_a, num_boxes_b = len(boxes_a), len(boxes_b)
# 	
# 	if num_boxes_a == 0 or num_boxes_b == 0: return [], []
# 	
# 	distances = np.zeros((num_boxes_a, num_boxes_b))
# 	
# 	for a in range(num_boxes_a):
# 		for b in range(num_boxes_b):
# 			if distance_func == 'def':
# 				distances[a,b] = distance_def(boxes_a[a], boxes_b[b], scores_a[a], scores_b[b])
# 			elif distance_func == 'def_iou':
# 				distances[a,b] = np.inf if get_iou(boxes_a[a], boxes_b[b])<pair_iou \
# 										else distance_def(boxes_a[a], boxes_b[b], scores_a[a], scores_b[b])
# 			else: raise ValueError('Distance function not recognized')
# 			
# 	
# 	pairs = []
# 	while distances.min() != np.inf:
# 		try:
# 			inds = np.where(distances == distances.min())
# 			a,b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
# 			a,b = int(a), int(b)
# 			pairs.append((a, b))
# 			distances[a,:] = np.inf
# 			distances[:,b] = np.inf
# 		except Exception as e:
# 			print(e)
# 			print(distances.min())
# 			print(np.where(distances == distances.min()))
# 		
# 	unmatched_pairs = [ i for i in range(num_boxes_a) if i not in [ p[0] for p in pairs] ]
# 	
# 	return pairs, unmatched_pairs

# # Pair -> (ind of box in the current frame, ind of box in the next frame)
# def get_pairs(video_boxes, video_scores, num_frames, distance_func, pair_iou):
# 	pairs, unmatched_pairs = [], []
# 	for i in range(num_frames - 1):
# 		pairs_i, unmatched_pairs_i = get_pairs_matching(video_boxes[i], video_boxes[i+1], 
# 												  video_scores[i], video_scores[i+1],
# 												  distance_func, pair_iou)
# 		pairs.append(pairs_i); unmatched_pairs.append(unmatched_pairs_i)
# 		
# 	return pairs, unmatched_pairs

# # Tubelet -> [ (ind of frame, bbox, scores), (...), ... ]
# def create_tubelets(pairs, video_boxes, video_scores, num_frames):

# 	tubelets = []
# 	tubelet_count = 0
# #	iter_num = 0
# 	first_frame = 0
# 	
# #	while sum([ len(p) for p in pairs ] ) > 0:
# 	while first_frame != num_frames-1:
# 		
# 		ind = None # Reset tubelet
# 		
# #		for i in range(num_frames-1):
# 		for i in range(first_frame, num_frames-1):
# 			
# 			if ind is not None:
# 				# Continue tubelet
# 				pair = [ p for p in pairs[i] if p[0] == ind ]
# 				if len(pair) == 0:
# 					# Tubelet ended
# 					tubelets[tubelet_count].append([i, video_boxes[i][ind], video_scores[i][ind]]) # 1
# 					tubelet_count += 1
# 					ind = None
# 					break
# 				else:
# 					# Continue tubelet
# 					pair = pair[0]; del pairs[i][pairs[i].index(pair)]
# 					tubelets[tubelet_count].append([i, video_boxes[i][ind], video_scores[i][ind]]) # 2
# 					ind = pair[1]
# 			
# 			else:
# 				# Looking for a new tubelet
# 				if len(pairs[i]) == 0: 
# 					# Keep searching
# 					first_frame = i+1
# 					continue
# 				else:
# 					# Beginning of tubelet found
# 					pair = pairs[i][0]; del pairs[i][0]
# 					tubelets.append([[i, video_boxes[i][pair[0]], video_scores[i][pair[0]]]]) # 3
# 					ind = pair[1]
# 		
# 		if ind != None:
# 			tubelets[tubelet_count].append([i+1, video_boxes[i+1][ind], video_scores[i+1][ind]]) # 4
# 			tubelet_count += 1
# 			ind = None		
# 		
# #		iter_num += 1
# 		
# 	return tubelets


# def rescore_tubelets(tubelets, rescore_func):	
# 	# Rescoring
# 	for t in range(len(tubelets)):
# 		t_scores = [ t[2] for t in tubelets[t] ]
# 		if rescore_func == 'mean': new_score = np.mean(t_scores, axis=0)
# 		elif rescore_func == 'median': new_score = np.median(t_scores, axis=0)
# 		elif rescore_func == 'max': new_score = np.max(t_scores, axis=0)
# 		else: raise ValueError('Rescore function not recognized')
# 		for i in range(len(tubelets[t])): tubelets[t][i][2] = new_score
# 		
# 	return tubelets


# def add_unmatched_pairs_as_single_tubelet(unmatched_pairs, video_boxes, video_scores):
# 	new_tubelets = []
# 	for num_frame, um in enumerate(unmatched_pairs):
# 		for num_box in um:
# 			new_tubelets.append([[num_frame, video_boxes[num_frame][num_box], video_scores[num_frame][num_box]]])
# 	return new_tubelets



# # Tubelet -> [(frame_num, bbox, score), ...]
# def tubelet_linking(tubelets, k, distance_func, link_iou):

# 	extrems = [ (t[0], t[-1]) for t in tubelets ]
# 	
# 	tubelets_to_join = []
# 	matched_tubelets = []
# 	for i in range(len(extrems)):
# 		_, (end_frame, end_box, end_score) = extrems[i]
# 	
# 		distances, d_inds = [], []
# 		for j in range(i+1, len(extrems)):
# 			if j in matched_tubelets: continue
# 			(start_frame, start_box, start_score), _ = extrems[j]
# 			diff = start_frame - end_frame
# 			
# 			if diff <= k and diff > 0:
# 				if get_iou(end_box, start_box) > link_iou:
# 					if distance_func in ['def', 'def_iou']:
# 						distances.append(distance_def(end_box,  start_box, end_score, start_score))
# #					elif distance_func == 'euc':
# #						distances.append(distance_euclidean(end_box,  start_box, end_score, start_score))
# #					elif distance_func == 'cos':
# #						distances.append(distance_cosine(end_box,  start_box, end_score, start_score))
# 					else: raise ValueError('Distance function not recognized')
# 				d_inds.append(j)
# 			elif diff > k: break
# 				
# 		if len(distances) > 0: 
# 			link_ind = d_inds[distances.index(min(distances))]
# 			matched_tubelets.append(link_ind)
# 			tubelets_to_join.append((i,link_ind))
# 	
# 	
# 	
# 	new_tubelets = []
# 	for start, end in tubelets_to_join:
# 		(start_frame, start_box, start_scores), start_len = tubelets[start][-1], len(tubelets[start])
# 		(end_frame, end_box, end_scores), end_len = tubelets[end][0], len(tubelets[end])
# 		
# 		num_frames = end_frame - start_frame - 1 
# 				
# 		new_boxes = np.vstack([np.linspace(start_box[0], end_box[0], num_frames+2, endpoint=True),
# 								np.linspace(start_box[1], end_box[1], num_frames+2, endpoint=True),
# 								np.linspace(start_box[2], end_box[2], num_frames+2, endpoint=True),
# 								np.linspace(start_box[3], end_box[3], num_frames+2, endpoint=True)]).transpose()[1:-1]
# 		
# #		new_scores = np.mean([start_scores, end_scores], axis=0)
# 		new_scores = (start_len/(start_len + end_len))*start_scores + (end_len/(start_len + end_len))*end_scores
# 		
# 		new_tubelet = [ (start_frame + i + 1, new_boxes[i], new_scores) for i in range(num_frames) ]
# 		new_tubelets.append(new_tubelet)

# 	return tubelets + new_tubelets


# def get_predictions(tubelets, video_annotations):
# 	preds = []
# 	track_id = 0
# 	for t in range(len(tubelets)):
# 		for frame_num, bbox, scores in tubelets[t]:
# 			ann = video_annotations[frame_num]
# 			img = ann.split()[0]
# 			
# 			left, bottom, right, top = [ int(b) for b in bbox.tolist() ]
# 			left, bottom = max(0, left), max(0, bottom)
# 	#		right, top = min(img_size[1], right), min(img_size[0], top)
# 			width = top-bottom
# 			height = right-left
# 				
# 			preds.append({
# 					'image_id': '.'.join(img.split('.')[:-1]),
# 					'category_id': int(np.where(scores == scores.max())[0]),
# 					'bbox': [ bottom, left, width, height ],
# 					'score': float(scores.max()),
# 					'track_id': track_id
# 				})
# 		track_id += 1
# 		
# 	return preds


# def get_predictions_per_class(annotations_file, tubelets, video_annotations, score=0.):
# 	if annotations_file != '': 
# 		with open(annotations_file, 'r') as f: annotations = [ ann.split()[0] for ann in f.read().splitlines() ]
# 	preds = []
# 	track_id = 0
# #	for t in tqdm(range(len(tubelets))):
# 	for t in range(len(tubelets)):
# 		for frame_num, bbox, scores in tubelets[t]:
# 			ann = video_annotations[frame_num]
# 			img = ann.split()[0]
# 			if annotations_file != '' and img not in annotations: continue
# 			
# 			left, bottom, right, top = [ int(b) for b in bbox.tolist() ]
# 			left, bottom = max(0, left), max(0, bottom)
# 	#		right, top = min(img_size[1], right), min(img_size[0], top)
# 			width = top-bottom
# 			height = right-left
# 				
# 			for i in range(len(scores)):
# 				if scores[i] > score:
# 					preds.append({
# 							'image_id': '.'.join(img.split('.')[:-1]),
# 							'category_id': i,
# 							'bbox': [ bottom, left, width, height ],
# 							'score': float(scores[i]),
# 							'track_id': track_id
# 						})
# 		track_id += 1
# 		
# 	return preds




# #def get_raw_score_predictions(model_folder, train_params, annotations_file, 
# #									   output_dir, score, nms_iou, best_weights=True,
# #									   raw_eval='def', load=False):
# #	
# #		model = EYOLO(
# #					model_image_size = tuple(train_params['input_shape']),
# #					model_path = train_utils.get_best_weights(model_folder),
# #					classes_path = train_params['path_classes'],
# #					anchors_path = train_params['path_anchors'],
# #					score = score,
# #					iou = None,	  # 0.5
# #					td_len = train_params.get('td_len', None),
# #					mode = train_params.get('mode', None),
# #					spp = train_params.get('spp', False),
# #					raw_eval = 'raw_scores'
# #				)
# #	
# #	
# #	
# #	from PIL import Image
# #	
# #	with open(annotations_video_frames_file, 'r') as f: annotations = f.read().splitlines()
# #	annotations = [ ann for ann in annotations[::skip_frames] ]
# #	with open(annotations_file, 'r') as f: annotations += [ ann.split()[0] for ann in f.read().splitlines() ]
# #	annotations = sorted(list(set(annotations)))
# #	
# #	total_boxes, total_scores = [], []
# #	total = len(annotations)
# #	for ann in tqdm(annotations[:total], total=total, file=sys.stdout):
# #	
# #		img = ann.split()[0]
# #		images = [ Image.open(train_params['path_dataset'] + img) for img in img.split(',') ]
# #		
# #		boxes, scores = model.get_prediction(images)
# #		
# #		total_boxes.append(boxes); total_scores.append(scores)
# #		
# #	
# #	pickle.dump([annotations, total_boxes, total_scores], open(raw_score_file, 'wb'))




# # TODO: get raw_score desde esta funcion -> nombre de predicciones post basado en el de raw_eval
# def seq_bbox_matching(annotations_file, raw_score_file, nms_iou, distance_func, pair_iou, rescore_func, k, add_unmatched, link_iou, 
# 				 per_class, verbose=False, load=False, force=False, min_score=0., **kargs):
# 	
# 	total_annotations, total_boxes, total_scores = pickle.load(open(raw_score_file, 'rb'))

# 	preds_filename = raw_score_file.replace('.pckl', 
# 			 '_iou{}_dist-{}_resc-{}_k{}_add-unm-{}_link-iou{}_perclass-{}_ms{}.json'.format(
# 				nms_iou, 
# 				distance_func if distance_func != 'def_iou' else 'def-iou{}'.format(pair_iou), 
# 				rescore_func, k,
# 				't' if add_unmatched else 'f',
# 				'-f' if k <2 else link_iou,
# 				't' if per_class else 'f',
# 				min_score
# 			))
# 	

# 	if force and os.path.isfile(preds_filename): os.remove(preds_filename)


# 	if os.path.isfile(preds_filename): 
# 		print(' * Loading:', preds_filename)
# 		if load: return json.load(open(preds_filename, 'r')), preds_filename
# 		else: return None, preds_filename
# 		
# 		
# 	print(' * Processing:', preds_filename)

# 	if len(total_annotations[0].split('/')) == 1: videos = ['']
# 	else: videos = list(set([ s.split('/')[1] for s in total_annotations ]))
# 	preds = []
# 	for video in tqdm(videos, file=sys.stdout):
# 	#	video = '00007041'
# 		video_boxes, video_scores, video_annotations = [], [], []
# 		if verbose: print('\nApplying IoU'); t = time.time()
# 		for i, ann in enumerate(total_annotations):
# 			if video in ann:
# 				frame_boxes = total_boxes[i]
# 				frame_scores = total_scores[i]
# 				
# 				inds = nms(frame_boxes, frame_scores.max(axis=1), nms_iou)
# 				
# 				video_boxes.append(frame_boxes[inds])
# 				video_scores.append(frame_scores[inds])
# 				video_annotations.append(ann)
# 				
# 		
# 		num_frames = len(video_annotations)
# 		
# 		if verbose: print(num_frames, 'Frames')
# 		if verbose: print(' - {:.2f}m'.format((time.time()-t)/60)); print('Getting pairs'); t = time.time()
# 		pairs, unmatched_pairs = get_pairs(video_boxes, video_scores, num_frames, distance_func, pair_iou)
# 		if verbose: print(' - {:.2f}m'.format((time.time()-t)/60)); print('Creating tubelets'); t = time.time()
# 		tubelets = create_tubelets(pairs, video_boxes, video_scores, num_frames)
# 		if verbose: print(' - {:.2f}m'.format((time.time()-t)/60)); print('Rescoring'); t = time.time()
# 		tubelets = rescore_tubelets(tubelets, rescore_func)
# 		if add_unmatched: tubelets += add_unmatched_pairs_as_single_tubelet(unmatched_pairs, video_boxes, video_scores)
# 		
# 		if verbose: print(' - {:.2f}m'.format((time.time()-t)/60)); print('Tubelet linking'); t = time.time()
# 		if k>0: 
# 			tubelets = tubelet_linking(tubelets, k, distance_func, link_iou)
# #			tubelets = tubelet_linking_v2(tubelets, k, distance_func, link_iou, pair_iou, num_frames)
# 		
# 		if verbose: print(' - {:.2f}m'.format((time.time()-t)/60)); print('Getting predictions'); t = time.time()
# 		if per_class: preds += get_predictions_per_class(annotations_file, tubelets, video_annotations, score=min_score)
# 		else: preds += get_predictions(tubelets, video_annotations)
# 		if verbose: print(' - {:.2f}m'.format((time.time()-t)/60)); 
# 	
# #	preds_filename = 'test_post_processing/preds_sbm_ilsvrc_val_s0.005_sk3_prenmsscore{}_tbltlink{}.json'.format(iou, k)
# 	json.dump(preds, open(preds_filename, 'w'))
# 	print(preds_filename, 'stored')

# 	if load: return preds, preds_filename
# 	else: return None, preds_filename


# # %%

# if False:
# 	# %%
# 	sys.path.append('keras_yolo3/')
# 	sys.path.append('keras_yolo3/yolo3/')
# 	import keras_yolo3.train as ktrain
# 	import evaluate_model
# 	
# 	# ilsvrc m3 - default - 	 	55.159
# 	# ilsvrc m3 - sk2 - s0.005 - 	0.581
# 	# ilsvrc m3 - sk2 - s0.0005 - 	0.605
# 	# ilsvrc m3 - sk2 - s0.00005 - 	0.614
# 	
# 	# adl m66 - default -  	 	 	39.694
# 	# adl m66 - sk2 - s0.005 -
# 	# adl m66 - sk2 - s0.00005 -
# 	

# # ilsvrc m3 - score 0.00005 - sk2 - mi -1
# #score = 5e-05, skip_frames = 2, max_images = -1
# #score 5e-05 | seq-nms: 61.044 | reg: 56.280 | diff 4.765
# #score 0.2 | seq-nms: 53.908 | reg: 48.188 | diff 5.720
# #score 0.4 | seq-nms: 47.131 | reg: 44.341 | diff 2.790

# # ilsvrc m3 - score 0.00005 - sk2 - mi 25
# #score 5e-05 | seq-nms: 58.221 | reg: 56.280 | diff 1.941
# #score 0.2 | seq-nms: 50.938 | reg: 48.188 | diff 2.750
# #score 0.4 | seq-nms: 45.405 | reg: 44.341 | diff 1.064

# # ilsvrc m3 - score 0.00005 - sk2 - mi 50
# # ilsvrc m3 - score 0.00005 - sk2 - mi 100
# # ilsvrc m3 - score 0.00005 - sk2 - mi 250

# #############################################################################

# # adl m66 - score 0.05 - sk3 - mi -1
# #score 0.05 | seq-nms: 36.631 | reg: 35.591 | diff 1.040
# #score 0.2 | seq-nms: 33.662 | reg: 31.222 | diff 2.440
# #score 0.4 | seq-nms: 29.309 | reg: 26.996 | diff 2.313
# 	
# # adl m66 - score 0.0005 - sk3 - mi 25
# #score = 0.0005, skip_frames = 3, max_images = 25
# #score 0.0005 | seq-nms: 40.502 | reg: 39.488 | diff 1.014
# #score 0.2 | seq-nms: 32.713 | reg: 31.222 | diff 1.492
# #score 0.4 | seq-nms: 27.023 | reg: 26.996 | diff 0.027

# # adl m66 - score 0.0005 - sk3 - mi 50

# # adl m66 - score 0.0005 - sk3 - mi 100

# # adl m66 - score 0.0005 - sk3 - mi 250
# #score = 0.0005, skip_frames = 3, max_images = 250
# #score 0.0005 | seq-nms: 41.042 | reg: 39.488 | diff 1.555
# #score 0.2 | seq-nms: 33.369 | reg: 31.222 | diff 2.147
# #score 0.4 | seq-nms: 27.377 | reg: 26.996 | diff 0.381

# # adl m66 - score 0.0005 - sk3 - mi 1000
# #t = 114
# #score = 0.0005, skip_frames = 3, max_images = 1000
# #score 0.0005 | seq-nms: 40.921 | reg: 39.488 | diff 1.434
# #score 0.2 | seq-nms: 33.308 | reg: 31.222 | diff 2.086
# #score 0.4 | seq-nms: 27.389 | reg: 26.996 | diff 0.393
# 	

# 	path_results = '/mnt/hdd/egocentric_results/'
# #	dataset_name, model_num, skip_frames, annotations_file = 'ilsvrc', 3, 2, 'dataset_scripts/ilsvrc/annotations_val_sk20.txt'
# 	dataset_name, model_num, skip_frames, annotations_file = 'adl', 66, 3, 'dataset_scripts/adl/annotations_adl_val_v2_27.txt'
# 	score = 0.0005
# 	max_images = 1000
# 		
# 	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
# 	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
# 	class_names = ktrain.get_classes(train_params['path_classes'])
# 	
# 	
# #	annotations_file = 'dataset_scripts/ilsvrc/annotations_val_sk20.txt'
# 	annotations_file_seq_nms = annotations_file.replace('.txt', '_seq_nms_sk{}.txt'.format(skip_frames))
# 	
# 	preds_raw, preds_raw_filename = predict_and_store_from_annotations(model_folder, train_params, annotations_file_seq_nms, 
# 										   model_folder, score, nms_iou=None, best_weights=False,
# 										   raw_eval=True)
# 	
# 	t = time.time()
# 	preds_seq_nms, preds_seq_nms_filename = get_model_seq_nms_predictions(annotations_file_seq_nms, class_names, preds_raw_filename, max_images)
# 	if (time.time()-t)/60 > 5: os.system("echo {} > {}".format((time.time()-t)/60, preds_seq_nms_filename.replace('.json', '_time.txt')))
# 	
# 	preds_seq_nms_short, preds_seq_nms_short_filename = filter_out_seq_nms_predictions(preds_seq_nms_filename, annotations_file)
# 	
# 	
# 	eval_stats_val_post, videos = evaluate_model.get_full_evaluation(annotations_file, preds_seq_nms_short_filename, class_names, full=False)
# 	
# 	print('='*80)
# 	print('score {} | seq-nms: {:.3f}'.format(score, eval_stats_val_post['total'][1]*100))
# 	print('='*80)
# 	
# 	
# 	# Regular prediction
# 	preds_reg, preds_reg_filename = predict_and_store_from_annotations(model_folder, train_params, annotations_file, 
# 										   model_folder, score, nms_iou=0.5, best_weights=False,
# 										   raw_eval=False)
# 	eval_stats_val_reg, videos = evaluate_model.get_full_evaluation(annotations_file, preds_reg_filename, class_names, full=False)

# 	print('='*80)
# 	print('score {} | seq-nms: {:.3f} | reg: {:.3f} | diff {:.3f}'.format(score, 
# 				   eval_stats_val_post['total'][1]*100, eval_stats_val_reg['total'][1]*100,
# 				   eval_stats_val_post['total'][1]*100 - eval_stats_val_reg['total'][1]*100))
# 	print('='*80)
# 	
# 	
# 	filter_out_scores = [0.2, 0.4]
# 	eval_files = {}
# 	for fos in filter_out_scores:
# 		preds_seq_nms_short_2 = filter_predictions_by_score(preds_seq_nms_short_filename, fos)
# 		preds_reg_2 = filter_predictions_by_score(preds_reg_filename, fos)
# 		eval_stats_val_post_fos, videos = evaluate_model.get_full_evaluation(annotations_file, preds_seq_nms_short_2, class_names, full=False)
# 		eval_stats_val_reg_fos, videos = evaluate_model.get_full_evaluation(annotations_file, preds_reg_2, class_names, full=False)
# 		
# 		eval_files['post_{}'.format(fos)] = eval_stats_val_post_fos
# 		eval_files['reg_{}'.format(fos)] = eval_stats_val_reg_fos
# 		
# 	print('='*80)
# 	print('score = {}, skip_frames = {}, max_images = {}'.format(score, skip_frames, max_images))
# 	print('score {} | seq-nms: {:.3f} | reg: {:.3f} | diff {:.3f}'.format(score, 
# 				   eval_stats_val_post['total'][1]*100, eval_stats_val_reg['total'][1]*100,
# 				   eval_stats_val_post['total'][1]*100 - eval_stats_val_reg['total'][1]*100))
# 	for fos in filter_out_scores:
# 		print('score {} | seq-nms: {:.3f} | reg: {:.3f} | diff {:.3f}'.format(fos,
# 			eval_files['post_{}'.format(fos)]['total'][1]*100, 
# 			eval_files['reg_{}'.format(fos)]['total'][1]*100,
# 			eval_files['post_{}'.format(fos)]['total'][1]*100 - eval_files['reg_{}'.format(fos)]['total'][1]*100))
# 	print('='*80)
# 	
# 		
# 	# %%
# 	
# 	new_score = 0.005
# 	preds_raw_short = [ p for p in preds_raw if p['score'] >= score ]
# 	preds_short_filename = preds_raw_filename.replace('score{}'.format(score), 'score{}'.format(score))
# 	json.dump(preds_raw_short, open(preds_short_filename, 'w'))
# 	print(preds_short_filename, 'stored')


