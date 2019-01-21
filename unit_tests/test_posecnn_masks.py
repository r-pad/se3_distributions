# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import cv2
import os
import numpy as np
import scipy.io as sio

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

def loadPoseCNNOutput(mat_filename):
    data = sio.loadmat(mat_filename)
    labels = data['labels']
    poses = data['poses']
    rois = data['rois']
    return labels, poses,  
def main():
    benchmark_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    posecnn_folder = '/ssd0/bokorn/data/benchmarks/pose_cnn/output/lov/lov_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000/mat'
    test_dir = '/home/bokorn/results/test/output_posecnn/'
    num_imgs = 113198
    image_set_idxs = loadImageSetIndex(benchmark_folder)
    assert(len(image_set_idxs) == num_imgs), \
        'Image Set Indices and Num Images Do Not Match {} != {}'.format(len(image_set_idxs), num_imgs)
    for j in range(num_imgs):
        mat_filename = os.path.join(posecnn_folder, '{:06d}.mat'.format(j))
        img_path = os.path.join(benchmark_folder, 'data', image_set_idxs[j]) 
        break

    import IPython; IPython.embed()



if __name__=='__main__':
    main()

