# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torch.optim import Adam

import glob
import cv2
import os
import time
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
from generic_pose.losses.distance_utils import evaluateDataset 
from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.eval.plot_accuracy import plotAccuracy
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--image_set', type=str, default='train_split')
    parser.add_argument('--benchmark_dir', type=str, 
        default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/')
   
    args = parser.parse_args()
   
    dataset = YCBDataset(data_dir=args.benchmark_dir,
                         image_set=args.image_set,
                         img_size=(224,224))

    dataset.generateObjectImageSet()

