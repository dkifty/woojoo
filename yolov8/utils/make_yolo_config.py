#!/usr/bin/env python

import argparse
import math
import os
import shutil, glob
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline

with open('labels.txt', 'r') as label:
    labels = label.readlines()
label_list = []
for a in labels:
    label_list.append(a.rstrip())
label_list.sort()
label_name = label_list

label_list_check = label_list
label_list_check_ = []
for label_list_check__ in  label_list:
    if label_list_check__ == '__ignore__' or label_list_check__ == '_background_':
        pass
    else:
        label_list_check_.append(label_list_check__)
label_list_check_.sort()


# yolo v8
def yolov8_check():
    try:
        import ultralytics
    except ImportError:
        os.system('pip install ultralytics==8.0.221')

    import ultralytics
    
    if ultralytics.__version__ == '8.0.221':
        pass
    else:
        os.system('pip install ultralytics==8.0.221')

def yolov8_config(size=False, FOLDERS_COCO=['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']):
    if not os.path.exists('./yolo_configs'):
        os.mkdir('./yolo_configs')
    if not os.path.exists('./yolo_configs/data'):
        os.mkdir('./yolo_configs/data')
#    for a in FOLDERS_COCO:
#        globals()['{}_img_list'.format(a.split('_')[-1])] = glob.glob(os.path.join(a, 'images/*.jpg'))
#        globals()['{}_img_list'.format(a.split('_')[-1])].sort()
#        with open('./yolo_configs/data/'+a.split('_')[-1]+'.txt', 'w') as b:
#            b.write('\n'.join(globals()['{}_img_list'.format(a.split('_')[-1])]) + '\n')
#    print('train, valid, test txt file.... created')
    DIR = os.getcwd()
    with open('./yolo_configs/data/custom.yaml', 'w') as c:
            c.write('train : ' + str(os.path.join(DIR,FOLDERS_COCO[0])) + '/images' + '\n')
            c.write('val : ' + str(os.path.join(DIR,FOLDERS_COCO[1])) + '/images' + '\n')
            c.write('test : ' + str(os.path.join(DIR,FOLDERS_COCO[2])) + '/images' + '\n')
            c.write('\n')
            c.write(f'nc : {len(label_list_check_)}'+'\n')
            c.write(f'names : {label_list_check_}')
    print('costom.yaml file.... created')



    #import site
    #ultralytics_package_utils = os.path.join(site.getsitepackages()[0], 'ultralytics/data/utils.py')
    #with open(ultralytics_package_utils, 'r') as f:
    #    lines = f.readlines()
    #lines[33] = "    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels_seg{os.sep}'  # /images/, /labels/ substrings\n"
    #with open(ultralytics_package_utils, 'w') as a:
#	for line in lines:
#            a.write(line)

def make_yolo_config():
    yolov8_check()
    yolov8_config()
