# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
from model_renderer.pose_renderer import BpyRenderer

import os
import time
import numpy as np
from torch.utils.data import DataLoader

from quat_math import random_quaternion, quaternion_about_axis

from generic_pose.datasets.tensor_dataset import TensorDataset
from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.utils.image_preprocessing import unprocessImages

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/base_renders/002_master_chef_can/')
    parser.add_argument('--model_filename', type=str, default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/models/002_master_chef_can/textured.obj')
    parser.add_argument('--random_offset', dest='random_offset', action='store_true') 
    args = parser.parse_args()
    
    #dataset = TensorDataset(data_dir=args.benchmark_folder, 
    #                        img_size=(224, 224))
    #dataset.loop_truth = [1]
    #data_loader = DataLoader(dataset,
    #                         num_workers=0, 
    #                         batch_size=2, 
    #                         shuffle=False)

    #q = quaternion_about_axis(np.pi/2, [1,0,0])
    if(args.random_offset):
        offset_quat = random_quaternion()
        data_dir = os.path.join(args.data_dir, str(offset_quat)[1:-1].replace(' ','_'))
        dataset = TensorDataset(data_dir=data_dir, model_filename = args.model_filename, 
                                offset_quat=offset_quat, img_size=(224, 224))

    else:
        TensorDataset(data_dir=args.data_dir, img_size=(224, 224))
    import IPython; IPython.embed()

if __name__=='__main__':
    main()

