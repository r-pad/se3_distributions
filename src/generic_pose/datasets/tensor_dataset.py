# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn with some code pulled from https://github.com/yuxng/PoseCNN/blob/master/lib/datasets/lov.py
"""

import os
import torch
import numpy as np
import pickle

from quat_math import quaternionBatchMultiply, euler_matrix, quaternion_matrix, random_quaternion 

from generic_pose.bbTrans.discretized4dSphere import S3Grid
from generic_pose.utils import SingularArray
from generic_pose.datasets.ycb_dataset import ycbRenderTransform
from generic_pose.utils.image_preprocessing import preprocessImages
from generic_pose.datasets.image_dataset import PoseImageDataset
import generic_pose.utils.transformations as tf_trans
from generic_pose.utils.image_preprocessing import cropAndPad

import quat_math

class TensorDataset(PoseImageDataset):
    def __init__(self, data_dir, 
                 model_filename = None, 
                 obj = -1,
                 transform_func = ycbRenderTransform,
                 offset_quat = None,
                 camera_dist = 0.33,
                 base_level = 2,
                 *args, **kwargs):

        super(TensorDataset, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.model_filenames = model_filename
        if(model_filename is None):
            model_filename = ''
        self.append_rendered = False
        self.data_models = SingularArray(obj) 
        if(os.path.exists(os.path.join(data_dir, 'renders.pkl'))):
            print('Loading renders from {}'.format(data_dir))
            with open(os.path.join(data_dir, 'renders.pkl'), 'rb') as f:
                self.base_renders = pickle.load(f)
            #self.base_renders = torch.load(os.path.join(data_dir, 'renders.pt'))
            self.base_vertices = torch.load(os.path.join(data_dir, 'vertices.pt'))
        else:
            print('Rendering poses for {}'.format(model_filename))
            from model_renderer.pose_renderer import BpyRenderer
            grid = S3Grid(base_level)
            renderer = BpyRenderer(transform_func = transform_func)
            renderer.loadModel(model_filename, emit = 0.5)
            
            fx = 1066.778
            fy = 1067.487
            px = 312.9869
            py = 241.3109
            renderer.setCameraMatrix(fx, fy, px, py, 640, 480)

            self.base_vertices = np.unique(grid.vertices, axis = 0)
            if(offset_quat is not None):
                offset_quat = np.tile(offset_quat, [self.base_vertices.shape[0], 1])
                self.base_vertices = quaternionBatchMultiply(offset_quat, self.base_vertices)
            
            self.base_renders = []
            #self.base_renders = torch.zeros([self.base_vertices.shape[0], 3, 224, 224])

            for j, q in enumerate(self.base_vertices):
                trans_mat = quaternion_matrix(q)
                ycb_mat = euler_matrix(-np.pi/2,0,0)
                trans_mat = trans_mat.dot(ycb_mat)
                trans_mat[:3,3] = [0, 0, camera_dist] 
                img = renderer.renderTrans(trans_mat)
                rendered_img = cropAndPad(img)
                self.base_renders.append(rendered_img)
                #self.base_renders[j] = self.preprocessImages(rendered_img, normalize_tensor = True).float()
                #crop_img = torch.cat((crop_img, rendered_img), 0)

            import pathlib
            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(data_dir, 'renders.pkl'), 'wb') as f:
                pickle.dump(self.base_renders, f)
            #torch.save(self.base_renders, os.path.join(data_dir, 'renders.pt'))
            torch.save(self.base_vertices, os.path.join(data_dir, 'vertices.pt'))
            self.model_dir = None

    def getQuat(self, index):
        return self.base_vertices[index]

    def getImage(self, index):
        if(self.append_rendered):
            rendered_img = self.base_renders[index]
        else:
            rendered_img = None
        return self.base_renders[index], rendered_img

    def getObjectPoints(self):
        if(self.model_dir is None):
            return None
        point_file = os.path.join(self.model_dir, 'models', self.model_name, 'points.xyz')
        assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
        points = np.loadtxt(point_file)

        return points

    def getObjectName(self):
        return self.model_name

    def getTrans(self, index, use_gt):
        if(self.model_dir is None):
            return None
        if(use_gt):
            return quaternion_matrix(self.base_vertices[index])
        else:
            return quaternion_matrix(random_quaternion())


    def __len__(self):
        return self.base_vertices.shape[0]
 
