# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import os
import cv2
import numpy as np
import scipy.io as sio

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
import quat_math

ycb_mat = quat_math.euler_matrix(-np.pi/2,0,0)

def setYCBCameraMatrix(renderer):
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    w = 640
    h = 480
    renderer.setCameraMatrix(fx, fy, px, py, w, h)
        
def getOcclusionPercentage(dataset, renderer, index):
    image_prefix = os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index])
    occluded_mask = (cv2.imread(image_prefix + '-label.png')[:,:,:1] == dataset.obj)[:,:,0]

    data = sio.loadmat(os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index] + '-meta.mat'))
    pose_idx = np.where(data['cls_indexes'].flatten()==dataset.obj)[0][0]
    trans_mat = np.vstack([data['poses'][:,:,pose_idx], [0,0,0,1]])
    trans_mat = trans_mat.dot(ycb_mat)
    
    render_img = renderer.renderTrans(trans_mat)
    full_mask = render_img[:,:,3] == 255
    return 1-occluded_mask[full_mask].mean() 

