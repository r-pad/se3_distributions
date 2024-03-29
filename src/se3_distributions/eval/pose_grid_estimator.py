# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017
@author: bokorn
"""
#from model_renderer.pose_renderer import BpyRenderer

import numpy as np
import torch
import time

from se3_distributions.utils.image_preprocessing import preprocessImages
from se3_distributions.utils import to_var, to_np

from se3_distributions.losses.distance_utils import getFeatures
from se3_distributions.utils.pose_processing import getGaussianKernal, meanShift, tensorAngularAllDiffs

#from se3_distributions.bbTrans.discretized4dSphere import S3Grid

import os
root_folder = os.path.dirname(os.path.abspath(__file__))

class PoseGridEstimator(object):
    MODE_MAX = 'max'
    MODE_MEANSHIFT = 'mean_shift'    

    def __init__(self, render_dir, distance_network, 
                 image_chunk_size = 500, 
                 kernal_sigma = None,
                 mode_selection = None,
                 num_modes = 1):

        self.mode_selection = mode_selection if mode_selection else self.MODE_MEANSHIFT
        self.num_modes = num_modes

        self.dist_estimator = distance_network
        self.dist_estimator.eval()
        assert os.path.exists(os.path.join(render_dir, 'renders.pt')), \
            'Render Dir {} does not contain renders.pt'.format(render_dir)
        
        self.grid_vertices = torch.tensor(torch.load(os.path.join(render_dir, 'vertices.pt'))).float()
        self.grid_renders = torch.load(os.path.join(render_dir, 'renders.pt'))
        if torch.cuda.is_available():
            self.grid_vertices = self.grid_vertices.cuda()
            
        with torch.no_grad():
            self.grid_features = getFeatures(self.dist_estimator, to_var(self.grid_renders), image_chunk_size)
            self.grid_size = self.grid_features.shape[0]
        
        if(kernal_sigma is not None):
            self.kernal = getGaussianKernal(self.grid_vertices, kernal_sigma)
            if torch.cuda.is_available():
                self.kernal.cuda()
        else:
            self.kernal = None

    def getDistances(self, img, preprocess = True):
        if(preprocess):
            img = preprocessImages([img], (224,224),
                                   normalize_tensors = True,
                                   background = None,
                                   background_filenames = None, 
                                   remove_mask = True, 
                                   vgg_normalize = False).float()
        if torch.cuda.is_available():
            img = img.cuda()

        query_features = self.dist_estimator.queryFeatures(img).repeat(self.grid_size,1)
        dist_est = self.dist_estimator.compare_network(self.grid_features,query_features)
        
        if(self.kernal is not None):
            dist_est = torch.mm(test, kernal_norm)
        return dist_est

    def dist2Poses(self, dists, num_modes = None, mode_selection = None):
        mode_selection = mode_selection if mode_selection else self.mode_selection
        num_modes = num_modes if num_modes else self.num_modes 

        assert num_modes > 0, 'Must have atleaset one mode: {} < 1'.format(num_modes)
        if(mode_selection in (self.MODE_MAX, self.MODE_MEANSHIFT)):
            if(num_modes == 1):
                mode_idxs = torch.argmax(dists)
            else:
                mode_idxs = torch.argsort(dists, descending=True)[:num_modes]
            
            quats = self.grid_vertices[mode_idxs]
            if(len(quats.shape) == 1):
                quats = quats.unsqueeze(0)
            
            if(mode_selection == self.MODE_MEANSHIFT):
                quats = meanShift(quats, self.grid_vertices, dists)
                mode_idxs = torch.argmin(tensorAngularAllDiffs(quats, self.grid_vertices), dim=1) 
                  
        else:
            raise ValueError('Invalid Mode Selection ({}),'.format(mode_selection) \
                    + ' Valid values [max, meanshift]')
        return quats, mode_idxs

    def getPose(self, img, preprocess = True, 
                mode_selection = None, num_modes = None):
        dists = self.getDistances(img, preprocess=preprocess)
        return self.dist2Poses(dists, num_modes = num_modes, 
                               mode_selection = mode_selection)
