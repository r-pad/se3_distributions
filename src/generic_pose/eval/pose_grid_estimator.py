# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017
@author: bokorn
"""
#from model_renderer.pose_renderer import BpyRenderer

import numpy as np
import torch
import time

from generic_pose.utils.image_preprocessing import preprocessImages
from generic_pose.utils import to_var, to_np

from generic_pose.losses.distance_utils import getFeatures
from generic_pose.utils.pose_processing import getGaussianKernal

#from generic_pose.bbTrans.discretized4dSphere import S3Grid

import os
root_folder = os.path.dirname(os.path.abspath(__file__))

class PoseGridEstimator(object):
    MODE_MAX = 'max'
    MODE_MEANSHIFT = 'mean_shift'    

    def __init__(self, render_dir, distance_network, 
                 image_chunk_size = 500, 
                 kernal_sigma = None,
                 dist_sign = 1, 
                 mode_selection = None,
                 num_modes = 1):

        self.mode_selection = mode_selection if mode_selection else self.MODE_MAX
        self.num_modes = num_modes

        self.dist_estimator = distance_network
        self.dist_estimator.eval()
        self.dist_sign = 1
        assert os.path.exists(os.path.join(render_dir, 'renders.pt')), \
            'Render Dir {} does not contain renders.pt'.format(render_dir)
        
        self.grid_vertices = torch.load(os.path.join(render_dir, 'vertices.pt'))
        self.grid_renders = torch.load(os.path.join(render_dir, 'renders.pt'))
        
        with torch.no_grad():
            self.grid_features = getFeatures(self.dist_estimator, to_var(self.grid_renders), image_chunk_size)
            self.grid_size = self.grid_features.shape[0]
        
        if(kernal_sigma is not None):
            self.kernal = getGaussianKernal(self.grid_vertices, kernal_sigma)
        else:
            self.kernal = None

    def getDistances(self, img, preprocess = True):
        if(preprocess):
            img = preprocessImages([img], (224,224),
                                   normalize_tensors = True,
                                   background = None,
                                   background_filenames = None, 
                                   remove_mask = True, 
                                   vgg_normalize = False).cuda()

        query_features = self.dist_estimator.queryFeatures(img).repeat(self.grid_size,1)
        dist_est = self.dist_estimator.compare_network(self.grid_features,query_features)
        
        if(self.kernal is not None):
            dist_est = torch.mm(test, kernal_norm)
        return dist_est

    def dist2Poses(self, dists, num_modes = None, mode_selection = None):
        mode_selection = mode_selection if mode_selection else self.mode_selection
        num_modes = num_modes if num_modes else self.num_modes 

        assert num_modes > 0, 'Must have atleaset one mode: {} < 1'.format(num_modes)
        if(mode_selection == self.MODE_MAX):
            if(num_modes == 1):
                mode_idxs = torch.argmax(dists)
            else:
                mode_idxs = torch.argsort(dists, descending=True)[:num_modes]
        elif(mode_selection == self.MODE_MEANSHIFT):
            raise NotImplemented()
        else:
            raise ValueError('Invalid Mode Selection ({}),'.format(mode_selection) \
                    + ' Valid values [max, meanshift]')
        quats = self.grid_vertices[mode_idxs]
        return quats, mode_idxs

    def getPose(self, img, preprocess = True, 
                mode_selection = None, num_modes = None):
        dists = self.getDistances(img, preprocess=preprocess)
        return self.dist2Poses(dists, num_modes = num_modes, 
                               mode_selection = mode_selection)
