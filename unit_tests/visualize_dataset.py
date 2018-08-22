# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:15:38 2018

@author: bokorn
"""
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from generic_pose.training.binary_angle_utils import evaluateBinaryEstimate
from generic_pose.training.utils import to_np
from quat_math import quat2AxisAngle

import time

norm_mean = np.array([0.485, 0.456, 0.406])
norm_std = np.array([0.229, 0.224, 0.225])

def unprocessImages(imgs):
    imgs = np.transpose(to_np(imgs), (0,2,3,1))
    imgs = np.minimum(np.maximum(imgs*norm_std + norm_mean, 0.0), 1.0)*255
    return imgs

def visualize(data_set, 
              image_prefix,
              num_samples = 1, 
              seperate_classes = False):
    if(seperate_classes):
        pass
    else:
        for j in range(dataset):
            img = unprocessImages(dataset.getImage(j)) 
            print(quat2AxisAngle(trans[0][j])[1]*180/np.pi)
            cv2.imwrite(image_prefix + "{}_{}.png".format(models[0][j], j), img)
        if(n >= num_samples):
            break

def main():
    import os
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/test/')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset_type', type=str, default='linemod')
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--target_angle', type=float, default=np.pi/4)
    parser.add_argument('--max_orientation_iters', type=int, default=200)    
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--num_samples', type=int, default=1)
 
    args = parser.parse_args()

    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None

    if(args.dataset_type.lower() == 'numpy'):
        from generic_pose.datasets.numpy_dataset import NumpyImageDataset as Dataset
    elif(args.dataset_type.lower() == 'linemod'):
        from generic_pose.datasets.benchmark_dataset import LinemodDataset as Dataset
    elif(args.dataset_type.lower() == 'linemod_masked'):
        from functools import partial
        from generic_pose.datasets.benchmark_dataset import LinemodDataset
        Dataset = partial(LinemodDataset, use_mask = True)
    else:
        raise ValueError('Dataset type {} not implemented'.format(args.dataset_type))

    t = time.time()
    dataset = Dataset(data_folders=args.data_folder,
                      img_size = (224, 224),
                      max_orientation_offset = None,
                      max_orientation_iters = args.max_orientation_iters,
                      model_filenames=None,
                      background_filenames = background_filenames)
    data_loader.dataset.loop_truth = [1,0]
    print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    t = time.time()
    import IPython; IPython.embed()
    visualize(dataset, args.results_prefix, num_samples=args.num_samples)
   
if __name__=='__main__':
    main()
