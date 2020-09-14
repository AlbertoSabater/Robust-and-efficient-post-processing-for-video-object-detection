#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:47:03 2020

@author: asabater
"""

import pickle
from tqdm import tqdm
import json

#min_score = 0.005
min_score = 1e-3

"""
min_score = 0.005
{'mAP_total': 0.8223109218516467, 'mAP_slow': 0.8765187455817982, 'mAP_medium': 0.8066908533419901, 'mAP_fast': 0.6667568835312807}
min_score = 0.001
{'mAP_total': 0.826237741329942, 'mAP_slow': 0.8794652252704362, 'mAP_medium': 0.8110391589581412, 'mAP_fast': 0.6737941895654274}


min_score = 0.001
====================
distance = def
model_59
*{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.5', 'recoordinate': None}
*{'mAP_total': 0.8277318281698552, 'mAP_slow': 0.8760032999573327, 'mAP_medium': 0.8180597559607303, 'mAP_fast': 0.6838119144212477, 'time_matching': 412.50490951538086, 'time_stats': 117.47354292869568}
{'distance_func': 'clf-logregv2.0-max-probT-clf_thr0.5', 'recoordinate': None}
{'mAP_total': 0.8274140535602081, 'mAP_slow': 0.8762354129005356, 'mAP_medium': 0.817470474934055, 'mAP_fast': 0.6819491899930161, 'time_matching': 410.62167501449585, 'time_stats': 115.6283950805664}
*{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
*{'mAP_total': 0.8361610731727602, 'mAP_slow': 0.8802059747758826, 'mAP_medium': 0.8279515731612056, 'mAP_fast': 0.7008499768203557, 'time_matching': 404.71059584617615, 'time_stats': 126.51208424568176}
{'distance_func': 'clf-logregv2.0-max-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8358521025144217, 'mAP_slow': 0.8804191051268339, 'mAP_medium': 0.8271198572021945, 'mAP_fast': 0.6991637397576442, 'time_matching': 408.44320797920227, 'time_stats': 125.32793092727661}
model_55
{'distance_func': 'clf-logregv3.0-dot-probT-clf_thr0.5', 'recoordinating': None}
{'mAP_total': 0.827773427707364, 'mAP_slow': 0.8764852031766311, 'mAP_medium': 0.8176743057960928, 'mAP_fast': 0.6831421212530322, 'time_matching': 413.2671592235565, 'time_stats': 124.12069416046143}
{'distance_func': 'clf-logregv3.0-max-probT-clf_thr0.5', 'recoordinating': None}
{'mAP_total': 0.8270203908399213, 'mAP_slow': 0.8757846269265221, 'mAP_medium': 0.8164735056500021, 'mAP_fast': 0.6828720715722016, 'time_matching': 414.6664354801178, 'time_stats': 119.84029817581177}
{'distance_func': 'clf-logregv3.0-dot-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8360412530531651, 'mAP_slow': 0.8806726682055551, 'mAP_medium': 0.8273228964966267, 'mAP_fast': 0.6998325306942424, 'time_matching': 406.63107442855835, 'time_stats': 105.61772131919861}
{'distance_func': 'clf-logregv3.0-max-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8353508248571555, 'mAP_slow': 0.8800102462310514, 'mAP_medium': 0.8262209981479204, 'mAP_fast': 0.6994128992200506, 'time_matching': 421.10778522491455, 'time_stats': 115.08711385726929}

{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8373851250370803, 'mAP_slow': 0.8808975847399901, 'mAP_medium': 0.8290372882869482, 'mAP_fast': 0.6977458728072782, 'time_matching': 1649.065645456314, 'time_stats': 231.73603916168213}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.6', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8396473639189209, 'mAP_slow': 0.8838401374240364, 'mAP_medium': 0.8314851409632228, 'mAP_fast': 0.6974624487828405, 'time_matching': 1578.291211605072, 'time_stats': 228.0507526397705}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.7', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8406129560326748, 'mAP_slow': 0.8852198285857917, 'mAP_medium': 0.8314438455256767, 'mAP_fast': 0.7037383324358752, 'time_matching': 1565.749320268631, 'time_stats': 230.34095168113708}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.75', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8414208639944099, 'mAP_slow': 0.8867436990698736, 'mAP_medium': 0.8319977041049966, 'mAP_fast': 0.7083104265979561, 'time_matching': 1649.3702945709229, 'time_stats': 221.91299986839294}
*{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.80', 'recoordinate': 'wl0-std0.4'}
*{'mAP_total': 0.8421329795837483, 'mAP_slow': 0.8871784038276325, 'mAP_medium': 0.8332090469178383, 'mAP_fast': 0.7109387713303483, 'time_matching': 1575.168268918991, 'time_stats': 215.30632185935974}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.85', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.841984449994227, 'mAP_slow': 0.8915213297454255, 'mAP_medium': 0.832399211613932, 'mAP_fast': 0.7101066323290554, 'time_matching': 1559.9357306957245, 'time_stats': 203.24736332893372}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.9', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.839300917848328, 'mAP_slow': 0.8922590018678835, 'mAP_medium': 0.830903393289129, 'mAP_fast': 0.7032602501141554, 'time_matching': 1566.8819501399994, 'time_stats': 187.6267101764679}

{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.75', 'recoordinate': None}
{'mAP_total': 0.8328311775049769, 'mAP_slow': 0.8829589919268162, 'mAP_medium': 0.822285458574357, 'mAP_fast': 0.6913804100511354, 'time_matching': 1631.9658751487732, 'time_stats': 220.43813467025757}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.8', 'recoordinate': None}
{'mAP_total': 0.8339783788496403, 'mAP_slow': 0.8835131502699032, 'mAP_medium': 0.8242631252791502, 'mAP_fast': 0.6938319263599876, 'time_matching': 1560.4138586521149, 'time_stats': 215.77097630500793}
*{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.85', 'recoordinate': None}
*{'mAP_total': 0.8345693872528499, 'mAP_slow': 0.8881195396316492, 'mAP_medium': 0.8244228807603479, 'mAP_fast': 0.6954015412699265, 'time_matching': 1540.91313123703, 'time_stats': 214.18531918525696}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.9', 'recoordinate': None}
{'mAP_total': 0.8333484039706467, 'mAP_slow': 0.8894707123501975, 'mAP_medium': 0.8243320267988664, 'mAP_fast': 0.6912003653547191, 'time_matching': 1604.989217042923, 'time_stats': 212.33284711837769}


min_score = 0.005
====================
distance = def
{'mAP_total': 0.8252151782330073, 'mAP_slow': 0.8746162916945994, 'mAP_medium': 0.8138978025955995, 'mAP_fast': 0.6769627409228527}
model_59
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.5'}
{'mAP_total': 0.8269199312334806, 'mAP_slow': 0.8762292027629291, 'mAP_medium': 0.8165547249174703, 'mAP_fast': 0.682426694691781, 'time_matching': 488.3729259967804, 'time_stats': 102.86060380935669}
{'distance_func': 'clf-logregv2.0-max-probT-clf_thr0.5'}
{'mAP_total': 0.8268878656854046, 'mAP_slow': 0.8766601561842184, 'mAP_medium': 0.8156401956256173, 'mAP_fast': 0.6807969424615429, 'time_matching': 880.0275883674622, 'time_stats': 99.41381311416626}
{'distance_func': 'clf-logregv2.0-dot-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8352097195386575, 'mAP_slow': 0.880610596133767, 'mAP_medium': 0.8254734184818999, 'mAP_fast': 0.7005971895930796, 'time_matching': 404.954407453537, 'time_stats': 103.26727342605591}
{'distance_func': 'clf-logregv2.0-max-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8351736290935955, 'mAP_slow': 0.881005697966332, 'mAP_medium': 0.8245702996430081, 'mAP_fast': 0.6992631987646637, 'time_matching': 409.72991132736206, 'time_stats': 107.19990086555481}
model_55
{'distance_func': 'clf-logregv3.0-dot-probT-clf_thr0.5'}
{'mAP_total': 0.8264019171620797, 'mAP_slow': 0.875581785180845, 'mAP_medium': 0.8159036390282233, 'mAP_fast': 0.6810342068942468, 'time_matching': 395.4631383419037, 'time_stats': 101.24738335609436}
{'distance_func': 'clf-logregv3.0-max-probT-clf_thr0.5'}
{'mAP_total': 0.8262820562149745, 'mAP_slow': 0.8755878395147426, 'mAP_medium': 0.8149185521609158, 'mAP_fast': 0.6794699996157623, 'time_matching': 398.6341013908386, 'time_stats': 102.48417019844055}
{'distance_func': 'clf-logregv3.0-dot-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8344925500820443, 'mAP_slow': 0.8798231093589407, 'mAP_medium': 0.8247365944577479, 'mAP_fast': 0.6992225760576484, 'time_matching': 404.4858775138855, 'time_stats': 105.52545309066772}
{'distance_func': 'clf-logregv3.0-max-probT-clf_thr0.5', 'recoordinate': 'wl0-std0.4'}
{'mAP_total': 0.8343551460143794, 'mAP_slow': 0.8799604776640887, 'mAP_medium': 0.8235265884732772, 'mAP_fast': 0.6973084218495232, 'time_matching': 404.08143162727356, 'time_stats': 102.69867944717407}



"""


preds_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/new_res_VID_val_videos_resnet_v1_101_rcnn_selsa_aug_raw_aug_score0.001_opFalse.pckl'
#preds_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/res_raw_aug.pckl'
#preds_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/res_raw_aug_score0.001.pckl'
#preds_filename = '/mnt/hdd/egocentric_results/ilsvrc/FGFA/predictions_fgfa.pckl'
res = pickle.load(open(preds_filename, 'rb'), encoding='bytes')
preds_frame = res[0][0][0]
image_ids = res[0][1]

#imageset_filename = './data/ILSVRC/ImageSets/VID_val_frames.txt'
imageset_filename = '/home/asabater/projects/Egocentric_object_detection/dataset_scripts/ilsvrc/skms/annotations_val_skms-1_mvl2_image_set.txt'
with open(imageset_filename, 'r') as f: frame_data = f.read().splitlines()

# %%

# =============================================================================
# Store standard predictions per class. Pickle Stream
# =============================================================================


imageset = {}
for frame in frame_data:
	frame, i = frame.split()
	imageset[i] = frame
		
		
total_preds = []
for i in tqdm(range(len(imageset))):
	preds = preds_frame[i]
	frame_id = image_ids[i]
	image_id = imageset[str(frame_id)]
	
	for pred in preds:
		bbox = pred[:4]
		scores = pred[4:]
		x1,y1,x2,y2 = bbox
		
		for c,s in enumerate(scores):
			if s < min_score: continue
			total_preds.append({
									'image_id': image_id,
									'category_id': c,
									'bbox': [ x1, y1, x2-x1, y2-y1 ],
									'score': s,
								})
		
	
#store_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/preds_aug_score{}.json'.format(min_score)
store_filename = preds_filename.replace('.pckl', '_fs{}.json'.format(min_score)).replace('/res', '/preds')
json.dump(total_preds, open(store_filename, 'w'))
print('Standard predictions stored:', store_filename)


# %%

# =============================================================================
# Evaluate standard predicitons
# =============================================================================

import os
import json
import sys
sys.path.append('/home/asabater/projects/Egocentric_object_detection/object_detection_mAP_by_motion/')

#from imagenet_vid_eval_motion import get_motion_mAP
import imagenet_vid_eval_motion
import motion_utils

#min_score = 0.005
annotations_filename = '/home/asabater/projects/Egocentric_object_detection/dataset_scripts/ilsvrc/skms/annotations_val_skms-1_mvl2.txt'
imageset_filename = '/home/asabater/projects/Egocentric_object_detection/dataset_scripts/ilsvrc/skms/annotations_val_skms-1_mvl2_image_set.txt'

#preds_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/res_raw_aug_score0.001.pckl'
preds_standard_filename = preds_filename.replace('.pckl', '_fs{}.json'.format(min_score)).replace('/res', '/preds')
preds_imdb_filename = motion_utils.coco_preds_to_imdb(preds_standard_filename, imageset_filename, store_filename=None)
stats_filename = preds_imdb_filename.replace('preds', 'stats').replace('.txt', '.json')
print('***', stats_filename)

if os.path.isfile(stats_filename):
	stats = json.load(open(stats_filename, 'r'))
else:
	annocache = imageset_filename.replace('.txt', '_cache.pckl')
	annocache = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/' + annocache.split('/')[-1]
	
	#stats = get_motion_mAP(annotations_filename, preds_imdb_filename, stats_filename, annocache=annocache, remove_cache=False)
	
	imageset_dest_filename = motion_utils.annotations_to_imageset(annotations_filename, store_filename=None)	
	
	# Create motion file to fit new ImageSet
	motion_iou_file_orig = 'object_detection_mAP_by_motion/imagenet_vid_groundtruth_motion_iou.mat'
	imageset_filename_orig = '/mnt/hdd/datasets/imagenet_vid/ILSVRC2015/ImageSets/VID/val.txt'
	motion_iou_dest_filename = motion_utils.image_set_to_motion_file(motion_iou_file_orig, 
								imageset_filename_orig, imageset_dest_filename)
		
	multifiles = False
	annopath = os.path.join('/home/asabater/projects/ILSVRC2015', 'Annotations', '{0!s}.xml')
	print('Calculating mAP by motion speed')
	ap_data = imagenet_vid_eval_motion.vid_eval_motion(multifiles, preds_imdb_filename, annopath, 
													   imageset_dest_filename, imagenet_vid_eval_motion.classname_map, 
				   motion_iou_dest_filename, remove_cache=False, annocache=annocache)
		
	stats = motion_utils.parse_ap_data(ap_data)
	json.dump(stats, open(stats_filename, 'w'))
	
print(stats)
	

# %%
# %%
# %%
# %%

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

path_dataset = '/home/asabater/projects/ILSVRC2015/Data/VID/'
#store_filename = '/mnt/hdd/egocentric_results/ilsvrc/SELSA/preds_raw_scores_aug_score{}.pckl'.format(min_score)
store_filename = preds_filename.replace('.pckl', '_fs{}.pckl'.format(min_score)).replace('_raw_', '_raw_scores_').replace('/res', '/preds')
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
			
			if scores.max() < min_score: continue
		
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


# %%

store_file = open(store_filename, 'rb')
counts = []
while True:
	vid, preds = pickle.load(store_file)
	counts.append(max([ len(p) for _,p in preds_video_frame.items() ]))


