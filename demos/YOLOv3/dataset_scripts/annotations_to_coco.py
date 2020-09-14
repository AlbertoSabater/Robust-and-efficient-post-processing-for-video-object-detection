#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse


# python ./dataset_scripts/annotations_to_coco.py path_dataset path_classes

def annotations_to_coco(dataset_annotations_filename, dataset_classes_filename):
    
    with open(dataset_annotations_filename, 'r') as f: dataset_annotations = f.read().splitlines()
    with open(dataset_classes_filename, 'r') as f: dataset_classes = f.read().splitlines()

    annotations_coco = {
                        'info': {
                                    'description': dataset_annotations_filename,
                                },
                        'annotations': [],
                        'categories': [],
                        'images': [],
                        'licenses': []
            }

    # Add categories
    for i, c in enumerate(dataset_classes):
        annotations_coco['categories'].append({
                                        'supercategorie': '',
                                        'id': i,
                                        'name': c
                                    })

    # Add images and annotations
    count = 0
    for l in dataset_annotations:
        l = l.split()
        img = l[0]
        bboxes = l[1:]
        annotations_coco['images'].append({
                                'license': None,
                                'file_name': img,
                                'coco_url': None,
                                'height': None,
                                'width': None,
                                'date_captured': None,
                                'flickr_url': None,
                                'id': '.'.join(img.split('.')[:-1])
                    })
                            
        for bb in bboxes:
            bb = bb.split(',')
            cat = bb[-1]
            x_min, y_min, x_max, y_max = [ int(b) for b in bb[:-1] ]
            width = x_max-x_min
            height = y_max-y_min
            annotations_coco['annotations'].append({
                            'segmentations': [],
                            'area': width * height,
                            'iscrowd': 0,
                            'image_id': '.'.join(img.split('.')[:-1]),
                            'bbox': [ x_min, y_min, width, height ],
                            'category_id': int(cat),
                            'id': count
                    })
            count += 1
        
    # Store annotations_coco
    annotations_coco_filename = dataset_annotations_filename[:-4] + '_coco.json'
    json.dump(annotations_coco, open(annotations_coco_filename, 'w+'))
    print('Saving:', annotations_coco_filename)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("path_annotations", help="path to the annotations file to convert")
	parser.add_argument("path_classes", help="path to the dataset classes filename")
	args = parser.parse_args()
	annotations_to_coco(args.path_annotations, args.path_classes)


if __name__ == '__main__':
	main()
    



