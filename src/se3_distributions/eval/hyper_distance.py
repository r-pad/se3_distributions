# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
import torch

from se3_distributions.utils.image_preprocessing import preprocessImages
from se3_distributions.utils import to_var, to_np
from se3_distributions.bbTrans.discretized4dSphere import S3Grid

import os
root_folder = os.path.dirname(os.path.abspath(__file__))
vert600 = np.load(os.path.join(root_folder, 'ordered_600_cell.npy'))

class ExemplarDistPoseEstimator(object):
    def __init__(self, model_filename, dist_network,
                 img_size = (224,224),
                 use_bpy_renderer=False,
                 base_level = 0,
                 model_scale = 1.0):
        
        self.img_size = img_size

        self.dist_network = dist_network
        self.dist_network.cuda()
        self.dist_network.eval()
        self.grid = S3Grid(base_level)
        if(use_bpy_renderer):
            from model_renderer.pose_renderer import BpyRenderer
            self.renderer = BpyRenderer()
            self.renderer.loadModel(model_filename, model_scale = model_scale, emit = 1.0)
            self.renderPoses = self.renderer.renderPose
        else:
            from model_renderer.syscall_renderer import renderView
            from functools import partial
            self.renderPoses = partial(renderView, 
                                       model_filename,
                                       camera_dist=2,
                                       model_scale = model_scale,
                                       standard_lighting=-1)

        self.base_vertices = np.unique(self.grid.vertices, axis = 0)
        self.base_size = self.base_vertices.shape[0]
        self.base_renders = to_var(preprocessImages(self.renderPoses(self.base_vertices), 
                                                    img_size = self.img_size,
                                                    normalize_tensors = True).float(), 
                                   requires_grad = True)
        if(self.base_size > 1500):
            self.base_features = []
            for j in range(self.base_size//1500):
                j_srt = j*1500
                j_end = (j+1)*1500
                self.base_features.append(to_np(
                    self.dist_network.originFeatures(self.base_renders[j_srt:j_end]).detach()))
            self.base_features.append(to_np(
                self.dist_network.originFeatures(self.base_renders[j_end:]).detach()))
            self.base_features = to_var(torch.from_numpy(np.vstack(self.base_features)), 
                                        requires_grad = False)
            torch.cuda.empty_cache()
        else:
            self.base_features = self.dist_network.features(self.base_renders)

    def estimate(self, img, preprocess=True):
        if(preprocess):
            img = preprocessImages([img],
                                   img_size = self.img_size,
                                   normalize_tensors = True)
        num_imgs = img.shape[0]
        #img = to_var(img.repeat(60,1,1,1).float())
        #dists = to_np(self.dist_network(self.base_renders, img))
        query_features = self.dist_network.queryFeatures(to_var(img.float(), requires_grad=False)).repeat(self.base_size,1)
        dists = self.dist_network.compare_network(self.base_features,
                                                  query_features)
        return dists.flatten() 
        quats = []
        for j in range(num_imgs):
            idx = np.arange(j, 60*num_imgs, num_imgs)
            

