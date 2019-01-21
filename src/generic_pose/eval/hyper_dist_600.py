# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np

from generic_pose.utils.image_preprocessing import preprocessImages
from generic_pose.utils import to_var, to_np

import os
root_folder = os.path.dirname(os.path.abspath(__file__))
vert600 = np.load(os.path.join(root_folder, 'ordered_600_cell.npy'))

class ExemplarDistPoseEstimator(object):
    def __init__(self, model_filename, dist_network,
                 img_size = (224,224),
                 use_bpy_renderer=False):
        
        self.img_size = img_size

        self.dist_network = dist_network
        self.dist_network.cuda()
        self.dist_network.eval()
        if(use_bpy_renderer):
            from model_renderer.pose_renderer import BpyRenderer
            self.renderer = BpyRenderer()
            self.renderer.loadModel(model_filename)
            self.renderPoses = self.renderer.renderPose
        else:
            from model_renderer.syscall_renderer import renderView
            from functools import partial
            self.renderPoses = partial(renderView, 
                                       model_filename,
                                       camera_dist=2,
                                       standard_lighting=True)

        self.rend600 = to_var(preprocessImages(self.renderPoses(vert600), 
                                               img_size = self.img_size,
                                               normalize_tensors = True).float())

    def estimate(self, img, preprocess=True):
        if(preprocess):
            img = preprocessImages([img],
                                   img_size = self.img_size,
                                   normalize_tensors = True)
        num_imgs = img.shape[0]
        img = to_var(img.repeat(60,1,1,1).float())
        dists = to_np(self.dist_network(self.rend600, img))
        return dists.flatten() 
        quats = []
        for j in range(num_imgs):
            idx = np.arange(j, 60*num_imgs, num_imgs)
            

