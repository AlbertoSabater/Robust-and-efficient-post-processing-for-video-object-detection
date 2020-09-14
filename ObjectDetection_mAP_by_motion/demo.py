# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import sys
sys.path.append('object_detection_mAP_by_motion')
from imagenet_vid_eval_motion import get_motion_mAP
import motion_utils

from imagenet_vid_eval_motion import vid_eval_motion, classname_map
import os
# import pickle
import numpy as np
# import scipy.io as sio
import motion_utils
from tqdm import tqdm 



# %%

annotations_filename = '../dataset_scripts/ilsvrc/sfps/annotations_val_sfps-1_mvl2.txt'
#preds_filename_imdb = 'data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-logregv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms5e-05.json'
#preds_filename = 'data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-logregv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms5e-05_imdb.txt'
#preds_filename = 'data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-lgbv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms5e-05_imdb.txt'
#preds_filename = 'data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-logregv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms5e-05._coco.json'
preds_filename = 'data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-logregv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms0.005_coco.json'
stats = motion_utils.get_motion_mAP(annotations_filename, preds_filename)
print(stats)

# %%

annotations_filename = 'dataset_scripts/ilsvrc/sfps/annotations_val_sfps-1_mvl2.txt'
image_set_dest_filename = 'dataset_scripts/ilsvrc/sfps/annotations_val_sfps-1_mvl2_image_set.txt'
motion_iou_dest_filename = 'dataset_scripts/ilsvrc/sfps/annotations_val_sfps-1_mvl2_motion_iou.mat'
#preds_filename = 'object_detection_mAP_by_motion/data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-logregv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms0.005_coco.json'
preds_filename = 'object_detection_mAP_by_motion/data_test/preds_bm8_sbms_nms_embs_bw_annotations_val_sfps-1_mvl2_is512_score5e-05_iou0.5_mb25|tub_ms0.005_df-clf-logregv2.1-dot-probT-clf_thr0.0_auF_bhF_k0|preds_pcT_ms0.005_imdb.txt'

#annotations_filename = 'dataset_scripts/ilsvrc/annotations_val_sk15_fgfa.txt'
#image_set_dest_filename = 'dataset_scripts/ilsvrc/annotations_val_sk15_fgfa_image_set.txt'
#motion_iou_dest_filename = 'dataset_scripts/ilsvrc/sfps/annotations_val_sk15_fgfa_motion_iou.mat'
#preds_filename = 'object_detection_mAP_by_motion/data_test/preds_bw_annotations_val_sk15_fgfa_is512_score5e-05_iou0.5_imdb.txt'

motion_iou_file_orig = 'object_detection_mAP_by_motion/imagenet_vid_groundtruth_motion_iou.mat'
imageset_filename_orig = '/mnt/hdd/datasets/imagenet_vid/ILSVRC2015/ImageSets/VID/val.txt'

# %%


# Split ImageSet into videos
with open(image_set_dest_filename) as f: image_set = f.read().splitlines()
videos = list(set([ imgs.split('/')[-2] for imgs in image_set ]))
image_set_videos = { vid:[] for vid in videos }
for imgs in tqdm(image_set):
	vid = imgs.split('/')[-2]
	image_set_videos[vid].append(imgs)
	
# Split annotations into filename
with open(annotations_filename, 'r') as f: annotations = f.read().splitlines()
annotations_video = { vid:[] for vid in videos }
for ann in annotations:
	vid = ann.split()[0].split('/')[-2]
	annotations_video[vid].append(ann)

for vid in videos:
	annotations_filename_video = annotations_filename.replace('.txt', '_{}.txt'.format(vid))
	with open(annotations_filename_video, 'w') as f:
		for ann in annotations_video[vid]: f.write(ann + '\n')
	print('Stored:', annotations_filename_video)
	
	
for vid in tqdm(videos):
	# Store imageset
	imageset_vid_filename = image_set_dest_filename.replace('.txt', '_{}.txt'.format(vid))
	with open(imageset_vid_filename, 'w') as f:
		for imgs in image_set_videos[vid]:
			f.write(imgs + '\n')
	print('Stored:', imageset_vid_filename)
	
	# Split motion_file into videos
	motion_iou_dest_filename = imageset_vid_filename.replace('.txt', '_motion_iou.mat')
	motion_iou_dest_filename = motion_utils.image_set_to_motion_file(motion_iou_file_orig, 
						imageset_filename_orig, imageset_vid_filename, motion_iou_dest_filename=motion_iou_dest_filename)

# %%
	
# Split preds into videos

with open(image_set_dest_filename) as f: image_set = f.read().splitlines()
image_set = dict([ imgs.split()[::-1] for imgs in image_set ])
with open(preds_filename, 'r') as f: preds = f.read().splitlines()

preds_video = { vid:[] for vid in videos }
for pred in tqdm(preds):
	vid = image_set[pred.split()[0]].split('/')[0]
	preds_video[vid].append(pred)
	
for vid in videos:
	preds_vid_filename = preds_filename.replace('.txt', '_{}.txt'.format(vid))
	with open(preds_vid_filename, 'w') as f:
		for pred in preds_video[vid]:
			f.write(pred + '\n')
	print('Stored:', preds_vid_filename)
	

# %%
	
# TODO: test multifiles
# Compute mAP by video and average

annopath = os.path.join('/home/asabater/projects/ILSVRC2015', 'Annotations', '{0!s}.xml')
#	ap_data = vid_eval_motion(multifiles, preds_filename_imdb, annopath, imageset_dest_filename, classname_map, 
#                motion_iou_dest_filename, remove_cache=remove_cache)
ap_video = {}
for vid in tqdm(videos):
	annotations_filename_video = annotations_filename.replace('.txt', '_{}.txt'.format(vid))
	preds_vid_filename = preds_filename.replace('.txt', '_{}.txt'.format(vid))
	imageset_vid_filename = image_set_dest_filename.replace('.txt', '_{}.txt'.format(vid))
	motion_iou_dest_filename_vid = imageset_vid_filename.replace('.txt', '_motion_iou.mat')

	ap_data = vid_eval_motion(False, preds_vid_filename, annopath, 
						   imageset_vid_filename, classname_map, 
                motion_iou_dest_filename_vid, remove_cache=True)
	stats = motion_utils.parse_ap_data(ap_data)
	ap_video[vid] = stats

# %%

video_weights = { vid: len(anns)/len(annotations) for vid,anns in annotations_video.items() }
# 83.345, 84.569 vs. 74,3026
print('mAP_weight_mean:', sum([ ap_video[vid]['mAP_total']*video_weights[vid] for vid in videos ]))
print('mAP_mean:', np.mean([ ap_video[vid]['mAP_total'] for vid in videos ]))

# %%

annotations_name = 'annotations_val_sk15_fgfa'
#annotations_name = 'annotations_val_sk3'

# Create IMDB ImageSet
annotations_filename = '/home/asabater/projects/Egocentric_object_detection/dataset_scripts/ilsvrc/{}.txt'.format(annotations_name)
imageset_dest_filename = utils.annotations_to_imageset(annotations_filename, store_filename=None)

# Convert COCO predictions to IMDB format
preds_filename = 'data_test/preds_bw_{}_is512_score5e-05_iou0.5.json'.format(annotations_name)
# imageset_filename = 'data_test/annotations_val_sk15_fgfa_image_set.txt'
imdb_preds = utils.coco_preds_to_imdb(preds_filename, imageset_dest_filename, store_filename=None)

# Create motion file to fit new ImageSet
motion_iou_file_orig = 'imagenet_vid_groundtruth_motion_iou.mat'
imageset_filename_orig = '/mnt/hdd/datasets/imagenet_vid/ILSVRC2015/ImageSets/VID/val.txt'
motion_iou_dest_filename = utils.image_set_to_motion_file(motion_iou_file_orig, 
						imageset_filename_orig, imageset_dest_filename)



# %%

remove_cache = True
multifiles = False
annopath = os.path.join('/home/asabater/projects/ILSVRC2015', 'Annotations', '{0!s}.xml')
ap_data = vid_eval_motion(multifiles, imdb_preds, annopath, imageset_dest_filename, classname_map, 
                motion_iou_dest_filename, remove_cache=remove_cache)

utils.print_mAP(ap_data)

stats_motion = utils.parse_ap_data(ap_data)

                

