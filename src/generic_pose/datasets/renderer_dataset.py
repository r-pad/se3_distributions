# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
import cv2
import numpy as np
import torch
from quat_math import euler_matrix, random_quaternion, quaternion_matrix
from model_renderer.pose_renderer import BpyRenderer
from generic_pose.datasets.image_dataset import PoseImageDataset
from generic_pose.utils.image_preprocessing import cropAndPad

class PoseRendererDataset(PoseImageDataset):
    def __init__(self, model_filename, 
                 epoch_size = 10000,
                 *args, **kwargs):

        super(PoseRendererDataset, self).__init__(*args, **kwargs)
        
        self.model_filename = model_filename
        self.epoch_size = epoch_size
        self.renderer = BpyRenderer()                                                            
        self.renderer.loadModel(model_filename)

        fx = 1066.778
        fy = 1067.487
        px = 312.9869
        py = 241.3109
        self.renderer.setCameraMatrix(fx, fy, px, py, 640, 480)
        
        self.ycb_mat = euler_matrix(-np.pi/2,0,0)
        self.obj_dist = 1.0

    def __getitem__(self, index):
        quat = random_quaternion()
        trans_mat = quaternion_matrix(quat)
        trans_mat = trans_mat.dot(self.ycb_mat)
        trans_mat[:3,3] = [0, 0, self.obj_dist] 
        img = self.renderer.renderTrans(trans_mat) 
        img = self.preprocessImages(cropAndPad(img), normalize_tensor = True, augment_img = True)
        model = 0
        model_file = self.model_filename
        return img, quat, model, model_file
 
    def __len__(self):
        return self.epoch_size
