# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:25:15 2018

@author: bokorn
"""

import glob
import numpy as np

#import quaternions as quat
from pose_renderer import camera2quat
import transformations as tf

#filepath = os.path.realpath(__file__)
#filedir = os.path.abspath(os.path.join(filepath, os.pardir))
#project_dir = os.path.abspath(os.path.join(filedir, os.pardir))

class PoseDataSetGenerator(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_dict = {}

        files = glob.glob(self.data_dir + '/**/*.png', recursive=True)

        for filename in files:                
            [model_class, model, azimuth, elevation, tilt, depth] = filename.split('/')[-1].split('.')[-2].split('_')
            azimuth = azimuth[1:]
            elevation = elevation[1:]
            tilt = tilt[1:]
            
            if(model_class not in self.data_dict):
                self.data_dict[model_class] = {}

            class_dict = self.data_dict[model_class]
            
            if(model not in class_dict):
                class_dict[model] = []
            
            model_renders = class_dict[model]

            q_blender = camera2quat(float(azimuth), 
                                    float(elevation), 
                                    float(tilt))
            quat = np.roll(q_blender,-1)
            model_renders.append({'filename':filename,
                               'quat':quat})
           
    def modelTrainValidSplit(self, model, train_ratio = 1.0, 
                             max_orientation_offset = float('inf')):
                                 
        for k, class_dict in self.data_dict.items():
            if model in class_dict: 
                model_renders = class_dict[model].copy()
                break
        
        pairs = []        

        num_renders = len(model_renders)

        if(not np.isinf(max_orientation_offset)):
            while num_renders > 1:
                idx1 = np.random.randint(0, num_renders)
                q1 = model_renders[idx1]['quat']
                fn1 = model_renders[idx1]['filename']
                del model_renders[idx1]
                
                idxs = np.arange(num_renders-1)
                np.random.shuffle(idxs)
                for idx2 in idxs:
                    #idx2 = (idx1 + np.random.randint(1, num_renders))%num_renders
                    q2 = model_renders[idx2]['quat']
                    fn2 = model_renders[idx2]['filename']
    
                    dq = tf.quaternion_multiply(q1, tf.quaternion_conjugate(q2))
                    orientation_diff = 2.0*np.arccos(dq[3])
                    
                    if(orientation_diff < max_orientation_offset):
                        pairs.append([fn1, fn2])
                        del model_renders[idx2]
                        break
                
                num_renders = len(model_renders)
        else:
            pairs = [[render['filename']] for render in model_renders]

        split_idx = np.ceil(train_ratio*len(pairs)).astype(int)
        
        train_pairs = pairs[:split_idx]
        valid_pairs = pairs[split_idx:]
        
        train_filenames = [item for sublist in train_pairs for item in sublist]
        valid_filenames = [item for sublist in valid_pairs for item in sublist]

        return train_filenames, valid_filenames

    def classTrainValidSplit(self, model_class, 
                             num_models = -1,
                             train_ratio = 1.0, 
                             max_orientation_offset = float('inf')):
        if(num_models < 1 or num_models > len(self.data_dict[model_class])):
            num_models = len(self.data_dict[model_class])
            
        model_names = np.array(list(self.data_dict[model_class].keys()))   
        np.random.shuffle(model_names)
         
        split_idx = np.ceil(train_ratio*num_models).astype(int)
         
        train_models = model_names[:split_idx]
        valid_models = model_names[split_idx:num_models]
         
        train_filenames = []
        for model in train_models:
            filenames, _ = self.modelTrainValidSplit(model, max_orientation_offset=max_orientation_offset)
            train_filenames += filenames
         
        valid_filenames = []
        for model in valid_models:
             filenames, _ = self.modelTrainValidSplit(model, max_orientation_offset=max_orientation_offset)
             valid_filenames += filenames

        return train_filenames, valid_filenames
        
    def globalTrainValidSplit(self, num_classes = -1, 
                              num_models = -1,
                              train_ratio = 1.0, 
                              max_orientation_offset = float('inf')):
        if(num_classes < 1 or num_classes > len(self.data_dict)):
            num_classes = len(self.data_dict)
        
        class_names = np.array(list(self.data_dict.keys()))   
        np.random.shuffle(class_names)
         
        split_idx = np.ceil(train_ratio*class_names).astype(int)
         
        train_classes = class_names[:split_idx]
        valid_classes = class_names[split_idx:num_classes]
         
        train_filenames = []
        for model_class in train_classes:
            filenames, _ = self.classTrainValidSplit(model_class, 
                                                     num_models=num_models, 
                                                     max_orientation_offset=max_orientation_offset)
            train_filenames += filenames
         
        valid_filenames = []
        for model_class in valid_classes:
            filenames, _ = self.classTrainValidSplit(model_class, 
                                                      num_models=num_models, 
                                                      max_orientation_offset=max_orientation_offset)
            valid_filenames += filenames

        return train_filenames, valid_filenames
        