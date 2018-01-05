# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:44:12 2017

@author: bokorn
"""
import cv2
import numpy as np
#import os
import torch
from torch.utils.data import Dataset
import glob

#import quaternions as quat
from pose_renderer import camera2quat
import transformations as tf
from data_preprocessing import label2Probs, resizeAndPad

#filepath = os.path.realpath(__file__)
#filedir = os.path.abspath(os.path.join(filepath, os.pardir))
#project_dir = os.path.abspath(os.path.join(filedir, os.pardir))

class PoseFileDataSet(Dataset):
    def __init__(self, render_filenames, img_size, 
                 max_orientation_offset = None, 
                 max_orientation_iter = 1000):
        super(PoseFileDataSet, self).__init__()
        self.img_size = img_size
        self.filenames = []
        self.quats = []
        self.models = []
        self.model_idxs = {}
        self.euler_bins = 360
        
        if(max_orientation_offset is not None):
            self.max_orientation_offset = max_orientation_offset
        else:
            self.max_orientation_offset = float('inf')
            
        self.max_orientation_iter = max_orientation_iter
        
        idx = 0;
        
        for filename in render_filenames:                
            [model_class, model, azimuth, elevation, tilt, depth] = filename.split('/')[-1].split('.')[-2].split('_')
            azimuth = azimuth[1:]
            elevation = elevation[1:]
            tilt = tilt[1:]
            if(model in self.model_idxs):
                self.model_idxs[model].append(idx)
            else:
                self.model_idxs[model] = [idx]
                
            self.filenames.append(filename)
            q_blender = camera2quat(float(azimuth), 
                                    float(elevation), 
                                    float(tilt))
            self.quats.append(np.roll(q_blender,-1))
            self.models.append(model)
            idx += 1        

    def generateCluttered(self, filename, model, 
                          num_objects = 1, 
                          percent_occluded = 0.0):
        # Model image is PNG with alpha channel in 3
        model_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)        
        composite_img = np.zeros(self.img_size + (3,))
        
        for j in range(num_objects):
            while True:
                model_idx = np.random.randint(0, len(self.model_idxs))    
                if (model_idx != model):
                    break
            clutter_idx = self.model_idxs[model_idx][np.random.randint(0, len(self.model_idxs[model_idx]))]
            clutter_filename = self.filenames[clutter_idx]
            clutter_img = cv2.imread(clutter_filename, cv2.IMREAD_UNCHANGED)

    def __getitem__(self, index):
        origin_filename = self.filenames[index]
        origin_pose = self.quats[index]

        model = self.models[index]
        
        query_idx = self.model_idxs[model][np.random.randint(0, len(self.model_idxs[model]))]
        assert len(self.model_idxs[model]) > 1, "Model must have > 1 view (model: {})".format(model)

        orientation_diff = float('inf')
        
        orientation_iter = 0
        while orientation_diff > self.max_orientation_offset or orientation_iter == 0:
            diff_not_found = True
            while diff_not_found:
                #print(model, query_idx, index)
                query_idx = self.model_idxs[model][np.random.randint(0, len(self.model_idxs[model]))]    
                if(query_idx != index):
                    diff_not_found = False
                
            query_filename = self.filenames[query_idx]
            query_pose = self.quats[query_idx]
            
            origin_img = cv2.imread(origin_filename)
            query_img = cv2.imread(query_filename)
            
            if (len(origin_img.shape) == 2):
                origin_img = np.expand_dims(origin_img, axis=2)
                query_img = np.expand_dims(query_img, axis=2)
            
            origin_img = resizeAndPad(origin_img, self.img_size)
            query_img = resizeAndPad(query_img, self.img_size)
            
            origin_img = origin_img.astype('float32')
            query_img = query_img.astype('float32')
            origin_img = np.rollaxis(origin_img, 2)
            query_img = np.rollaxis(query_img, 2)
            
            d_quat = tf.quaternion_multiply(query_pose, 
                                            tf.quaternion_conjugate(origin_pose))
        
            orientation_diff = 2.0*np.arccos(d_quat[3])
            
            orientation_iter += 1
            if(orientation_iter > self.max_orientation_iter):
                raise AssertionError('Orientation search exceeded max iterations {}'.format(self.max_orientation_iter))
                
        conj_q = torch.from_numpy(tf.quaternion_conjugate(d_quat))
        angles = tf.euler_from_quaternion(d_quat)
        d_euler = np.round(np.array(angles)*self.euler_bins/(2.*np.pi))
        
        d_azim = torch.from_numpy(label2Probs(d_euler[0], self.euler_bins))
        d_elev = torch.from_numpy(label2Probs(d_euler[1], self.euler_bins))
        d_tilt = torch.from_numpy(label2Probs(d_euler[2], self.euler_bins))
           
        d_quat = torch.from_numpy(d_quat)

        return origin_img, query_img, conj_q, d_quat, d_azim, d_elev, d_tilt, d_euler

    def __len__(self):
        return len(self.filenames)


class PoseDirDataSet(PoseFileDataSet):
    def __init__(self, data_dir, img_size, 
                 max_orientation_offset = None, 
                 max_orientation_iter = 1000):
        self.data_dir = data_dir
        render_filenames = glob.glob(self.data_dir + '/**/*.png', recursive=True)
        super(PoseDirDataSet, self).__init__(self, render_filenames, 
                                             img_size=img_size, 
                                             max_orientation_offset = max_orientation_offset, 
                                             max_orientation_iter = max_orientation_iter)