#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:57:38 2020

@author: asabater
"""

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse


# =============================================================================
# Trains the Logistic Regression model for the detections linking
# =============================================================================


def main():
	
	parser = argparse.ArgumentParser(description='Trains a matching LogisticRegression classifier')
	parser.add_argument('--path_dataset_train', help='path of the matching train dataset', default='./data_annotations/triplet_annotations/matching_models_dataset_train_appearance.pckl', type=str)
	parser.add_argument('--path_dataset_val', help='path of the matching val dataset', default='./data_annotations/triplet_annotations/matching_models_dataset_val_appearance.pckl', type=str)
	parser.add_argument('--add_appearance', help='true add appearance features', action='store_true')

	args = parser.parse_args()
	
	
	print('Loading datasets')
	X_train, Y_train = pickle.load(open(args.path_dataset_train, 'rb'))
	X_val, Y_val = pickle.load(open(args.path_dataset_val, 'rb'))
	
	if args.add_appearance and 'descriptor_dist' not in X_train.columns:
		raise ValueError('Classification Dataset does not contain appearance pair-wise features')
		
	if args.add_appearance:
		X_train = X_train[['center_distances_corrected', 'descriptor_dist', 'height_rel', 'iou', 'width_rel']]
		X_val = X_val[['center_distances_corrected', 'descriptor_dist', 'height_rel', 'iou', 'width_rel']]
	else:
		X_train = X_train[['center_distances_corrected', 'height_rel', 'iou', 'width_rel']]
		X_val = X_val[['center_distances_corrected', 'height_rel', 'iou', 'width_rel']]
		
	
	print('Training classifier')
	clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
	print('Evaluating classifier')
	score_train = clf.score(X_train, Y_train)
	score_val = clf.score(X_val, Y_val)
	print('Train/test scores:', score_train, score_val)
	val_pred_class = clf.predict(X_val)
	print('Accuracy:', accuracy_score(Y_val, val_pred_class))
	
	
	model_filename = './REPP_models/matching_model_logreg{}_new.pckl'.format('_appearance' if args.add_appearance else '')
	pickle.dump([clf, X_train.columns.tolist()], open(model_filename, 'wb'))
	print('Model stored in:', model_filename)	
	
	
if __name__ == '__main__': main()

