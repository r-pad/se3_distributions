# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn with some code pulled from https://github.com/yuxng/PoseCNN/blob/master/lib/datasets/lov.py
"""

import os
import cv2
import torch
import numpy as np
import scipy.io as sio
import time
import sys

from se3_distributions.datasets.image_dataset import PoseImageDataset
from se3_distributions.utils import SingularArray
import se3_distributions.utils.transformations as tf_trans
from se3_distributions.utils.pose_processing import viewpoint2Pose
from se3_distributions.utils.image_preprocessing import cropAndPad
from transforms3d.quaternions import quat2mat, mat2quat

def ycbRenderTransform(q):
    trans_quat = q.copy()
    trans_quat = tf_trans.quaternion_multiply(trans_quat, tf_trans.quaternion_about_axis(-np.pi/2, [1,0,0]))
    return viewpoint2Pose(trans_quat)

def setYCBCamera(renderer, width=640, height=480):
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    renderer.setCameraMatrix(fx, fy, px, py, width, height)

class YCBDataset(PoseImageDataset):
    def __init__(self, data_dir, image_set, 
                 use_syn_data = False,
                 use_posecnn_masks = False,
                 *args, **kwargs):

        super(YCBDataset, self).__init__(*args, **kwargs)
        self.use_syn_data = use_syn_data
        self.data_dir = data_dir
        self.classes = ['__background__']
        with open(os.path.join(self.data_dir, 'image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])

        self.num_classes = len(self.classes)
        self.model_filenames = {}
        for j in range(1, self.num_classes):
            self.model_filenames[j] = os.path.join(self.data_dir, 'models', self.classes[j], 'textured.obj')

        self.image_set = image_set
        self.append_rendered = False
        self.use_posecnn_masks = use_posecnn_masks
	
    def loadImageSet(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self.data_dir, 'image_sets', self.image_set+'.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            self.data_filenames = [x.rstrip('\n') for x in f.readlines()]
        if(self.use_syn_data):
            for j in range(80000):
                self.data_filenames.extend('../data_syn/{:06d}'.format(j))

    def loadQuat(self, index):
        data = sio.loadmat(os.path.join(self.data_dir, 'data', self.data_filenames[index] + '-meta.mat'))
        pose_idx = np.where(data['cls_indexes'].flatten()==self.obj)[0][0]
        mat = np.eye(4)
        mat[:3,:3] = data['poses'][:3,:3,pose_idx]
        quat = tf_trans.quaternion_from_matrix(mat)
        return quat

    def getQuat(self, index):
        return self.loadQuat(index)

    def getImage(self, index, boarder_ratio=0.25, preprocess = True):

        image_prefix = os.path.join(self.data_dir, 'data', self.data_filenames[index])
        img = cv2.imread(image_prefix + '-color.png')
        if(self.use_posecnn_masks):
            mask = 255*(cv2.imread(image_prefix + '-posecnn-seg.png')[:,:,:1] == self.obj).astype('uint8')
        else:
            mask = 255*(cv2.imread(image_prefix + '-label.png')[:,:,:1] == self.obj).astype('uint8')
        if(np.sum(mask) == 0):
            #import IPython; IPython.embed()
            print('Index {} invalid for {} ({}:{})'.format(index, self.getObjectName(),
                    self.image_set, image_prefix))
            return None, None 
        img = np.concatenate([img, mask], axis=2)
        #if(preprocess):
        #    crop_img = self.preprocessImages(cropAndPad(img), normalize_tensor = True, augment_img = True)
        #else:
        crop_img = cropAndPad(img)
        
        if(self.append_rendered):
            #import IPython; IPython.embed();
            #rendered_img = self.preprocessImages(cv2.imread(image_prefix + '-color.png'), normalize_tensor = True)
            rendered_img = cv2.imread(image_prefix + '-{}-render.png'.format(self.obj), cv2.IMREAD_UNCHANGED)
            if(rendered_img is None):
                print(image_prefix + '-{}-render.png'.format(self.obj), 'Not Found')
                rendered_img = cropAndPad(img)

            #rendered_img = self.preprocessImages(rendered_img, normalize_tensor = True)
            #crop_img = torch.cat((crop_img, rendered_img), 0)
            #print('='*100)
            #print(image_prefix + '-{}-render.png'.format(self.obj))
            #print(crop_img.shape)
            #print(rendered_img.shape)
            #print('='*100)
            #crop_img = np.concatenate([crop_img, rendered_img], axis=2)
        else:
            rendered_img = None

        return crop_img, rendered_img

    def __len__(self):
        return len(self.data_filenames)
 
