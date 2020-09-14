#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:38:30 2019

@author: asabater
"""

classes = [
#                '__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']
classes_map = [
#                    '__background__',  # always index 0
                'n02691156', 'n02419796', 'n02131653', 'n02834778',
                'n01503061', 'n02924116', 'n02958343', 'n02402425',
                'n02084071', 'n02121808', 'n02503517', 'n02118333',
                'n02510455', 'n02342885', 'n02374451', 'n02129165',
                'n01674464', 'n02484322', 'n03790512', 'n02324045',
                'n02509815', 'n02411705', 'n01726692', 'n02355227',
                'n02129604', 'n04468005', 'n01662784', 'n04530566',
                'n02062744', 'n02391049']


def main():
    # write classes to file
    with open('imagenet_vid_classes.txt', 'w+') as f:
        for c in classes:
            f.write(c + '\n')
            
            
if __name__ == '__main__':
    main()

