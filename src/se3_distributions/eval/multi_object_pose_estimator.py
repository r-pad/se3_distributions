#!/usr/bin/env python
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

try:
    from model_renderer.pose_renderer import BpyRenderer
    from se3_distributions.datasets.ycb_dataset import ycbRenderTransform 
    BPY_IMPORTED = True
except ImportError:
    BPY_IMPORTED = False

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import glob
import cv2
import os
import time
import numpy as np
import scipy.io as sio

from se3_distributions.eval.pose_grid_estimator import PoseGridEstimator
from se3_distributions.models.pose_networks import gen_pose_net, load_state_dict
from se3_distributions.utils import to_np
from se3_distributions.utils.image_preprocessing import preprocessImages, unprocessImages
from se3_distributions.eval.template_pose_estimator import computeT 

class MultiObjectPoseEstimator(object):
    def __init__(self, weight_paths, render_paths, focal_length, image_center, 
                 object_models = None, render_distance = 0.2, render_f = 490):
        self.pose_estimators = []
        self.render_distance = render_distance
        self.render_f = render_f

        self.img_f = focal_length
        self.img_p = image_center
        self.img_size = (480, 640)
        self.pyr_scales = None

        if(object_models is not None):
            assert BPY_IMPORTED, 'Blender must be installed and a python module (bpy) to render model'
            self.renderer = BpyRenderer(transform_func = ycbRenderTransform)
            self.model_tags = []
            for obj in object_models:
                self.model_tags.append(self.renderer.loadModel(obj, emit = 0.5))
            self.renderer.hideAll()
        else:
            self.renderer = None


        for weight_file, render_folder in zip(weight_paths, render_paths):
            model = gen_pose_net('alexnet','sigmoid', output_dim = 1, 
                                 pretrained = True, siamese_features = False)
            load_state_dict(model, weight_file)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
            self.pose_estimators.append(PoseGridEstimator(render_folder, model))

    def __call__(self, img, class_idx, bbox_corner = (0,0)):
        poses, mode_idxs = self.pose_estimators[class_idx].getPose(img)
        if(self.renderer is not None):
            self.renderer.hideModel(self.model_tags[class_idx], hide=False)
            template = self.renderer.renderPose(poses,camera_dist=self.render_distance)[0]
            self.renderer.hideModel(self.model_tags[class_idx], hide=True)
        else:
            renders = self.pose_estimators[class_idx].grid_renders[mode_idxs]
            template = unprocessImages(renders)[0].astype(img.dtype)

        h,w = img.shape[:2]
        boarder_width = max(h,w)//2 - 20
        wide_crop = 255*np.ones((h + boarder_width*2,w + 2*boarder_width,3),
                                dtype=np.uint8)
        y0 = boarder_width
        x0 = boarder_width
        y1 = y0 + h
        x1 = x0 + w
        wide_crop[y0:y1, x0:x1,:] = img

        bbox_corner = (bbox_corner[0] - x0, bbox_corner[1] - y0)
        trans, t_box = computeT(wide_crop, template, 
                         f = self.img_f, p = self.img_p, 
                         f_t = self.render_f, 
                         bbox_corner = bbox_corner)
        
        y, x, wt, ht = t_box
        display_img = np.zeros(self.img_size + (3,))
        template_resized = cv2.resize(template, (ht, wt))
        y0, x0 = (y - int(np.floor(ht/2)), int(x - np.floor(wt/2)))
        y1, x1 = (y + int(np.ceil( ht/2)), int(x + np.ceil( wt/2)))
        display_img[x0:x1, y0:y1] = template_resized

        return trans, to_np(poses[0]), display_img

    def getDistances(self, img, class_idx):
        dists = self.pose_estimators[class_idx].getDistances(img)
        return dists


