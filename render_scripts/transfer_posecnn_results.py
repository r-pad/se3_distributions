# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import cv2
import os
import shutil
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform

def loadImageSetIndex(data_dir, image_set='train'):
    """
    Load the indexes listed in this dataset's image set file.
    """
    image_set_file = os.path.join(data_dir, 'image_sets', image_set + '.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
        image_index = [x.rstrip('\n') for x in f.readlines()]
    return image_index

def main():
    benchmark_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    #posecnn_folder = '/ssd0/bokorn/data/benchmarks/pose_cnn/output/lov/lov_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000/mat'
    #num_imgs = 113198
    #image_set_idxs = loadImageSetIndex(benchmark_folder)
    posecnn_folder = '/ssd0/bokorn/data/benchmarks/pose_cnn/output/lov/lov_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000/mat'
    num_imgs = 2949
    image_set_idxs = loadImageSetIndex(benchmark_folder, image_set='keyframe')
    assert(len(image_set_idxs) == num_imgs), \
        'Image Set Indices and Num Images Do Not Match {} != {}'.format(len(image_set_idxs), num_imgs)

    pbar = tqdm(range(num_imgs))
    for j in pbar:
        pbar.set_description(image_set_idxs[j])
        mat_filename = os.path.join(posecnn_folder, '{:06d}.mat'.format(j))
        mat_data = sio.loadmat(mat_filename)
        
        data_path = os.path.join(benchmark_folder, 'data', image_set_idxs[j]) 
        shutil.copy(mat_filename, data_path + '-posecnn.mat')
        cv2.imwrite(data_path + '-posecnn-seg.png', mat_data['labels']) 
    
if __name__=='__main__':
    main()

