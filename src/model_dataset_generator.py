# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:14:41 2018

@author: bokorn
"""
import glob
import numpy as np

class ModelDataSetGenerator(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_dict = {}

        files = glob.glob(self.data_dir + '/**/*.obj', recursive=True)
            
        for filename in files:
            [model_class, model, model_file] = filename.split('/')[-3:]

            if(model_class not in self.data_dict):
                self.data_dict[model_class] = {}

            class_dict = self.data_dict[model_class]
            class_dict[model] = filename
    
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
            train_filenames += [self.data_dict[model_class][model]]
         
        valid_filenames = []
        for model in valid_models:
             valid_filenames += [self.data_dict[model_class][model]]

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
        