# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn with some code pulled from https://github.com/yuxng/PoseCNN/blob/master/lib/datasets/lov.py
"""

import os
import torch
import numpy as np

from quat_math import quaternionBatchMultiply

from generic_pose.bbTrans.discretized4dSphere import S3Grid
from generic_pose.utils import SingularArray
from generic_pose.datasets.ycb_dataset import ycbRenderTransform
from generic_pose.utils.image_preprocessing import preprocessImages
from generic_pose.datasets.image_dataset import PoseImageDataset
import generic_pose.utils.transformations as tf_trans

class TensorDataset(PoseImageDataset):
    def __init__(self, data_dir, 
                 model_filename = None, 
                 transform_func = ycbRenderTransform,
                 offset_quat = None,
                 base_level = 2,
                 *args, **kwargs):

        super(TensorDataset, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.model_filenames = model_filename
        if(model_filename is None):
            model_filename = ''
        self.append_rendered = False
        self.data_models = SingularArray(1) 
        if(os.path.exists(os.path.join(data_dir, 'renders.pt'))):
            self.base_renders = torch.load(os.path.join(data_dir, 'renders.pt'))
            self.base_vertices = torch.load(os.path.join(data_dir, 'vertices.pt'))
        else:
            from model_renderer.pose_renderer import BpyRenderer
            grid = S3Grid(base_level)
            renderer = BpyRenderer(transform_func = transform_func)
            renderer.loadModel(model_filename, emit = 0.5)
            renderPoses = renderer.renderPose
            
            self.base_vertices = np.unique(grid.vertices, axis = 0)
            if(offset_quat is not None):
                offset_quat = np.tile(offset_quat, [self.base_vertices.shape[0], 1])
                self.base_vertices = quaternionBatchMultiply(offset_quat, self.base_vertices)

            self.base_renders = preprocessImages(renderPoses(self.base_vertices, camera_dist = 0.33),
                                                 img_size = self.img_size,
                                                 normalize_tensors = True).float()
            import pathlib
            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.base_renders, os.path.join(data_dir, 'renders.pt'))
            torch.save(self.base_vertices, os.path.join(data_dir, 'vertices.pt'))
 
    def getQuat(self, index):
        return self.base_vertices[index]

    def getImage(self, index):
        if(self.append_rendered):
            return torch.cat((self.base_renders[index], self.base_renders[index]), 0)
        return self.base_renders[index]

    def __len__(self):
        return self.base_vertices.shape[0]
 
