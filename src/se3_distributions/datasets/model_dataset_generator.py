# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:14:41 2018

@author: bokorn
"""
import glob
import numpy as np

class ModelDataSetGenerator(object):
    def __init__(self, data_dir, obj_dir_depth = 0):
        self.data_dir = data_dir
        self.data_dict = {}

        files = glob.glob(self.data_dir + '/**/*.obj', recursive=True)
            
        for filename in files:
            [model_class, model] = filename.split('/')[(-3-obj_dir_depth):(-1-obj_dir_depth)]

            if(model_class not in self.data_dict):
                self.data_dict[model_class] = {}

            class_dict = self.data_dict[model_class]
            class_dict[model] = filename
    
    def classTrainValidTestSplit(self, model_class, 
                             num_models = -1,
                             valid_ratio = 0, 
                             test_ratio = 0, 
                             max_orientation_offset = float('inf')):
        if(num_models < 1 or num_models > len(self.data_dict[model_class])):
            num_models = len(self.data_dict[model_class])
            
        model_names = np.array(list(self.data_dict[model_class].keys()))   
        np.random.shuffle(model_names)
        
        num_valid = np.ceil(valid_ratio*num_models).astype(int)
        num_test = np.ceil(test_ratio*num_models).astype(int)
        num_train = num_models - num_valid - num_test

        train_models = model_names[:num_train]
        split_idx = num_train
        valid_models = model_names[split_idx:(split_idx+num_valid)]
        split_idx = split_idx+num_valid
        test_models = model_names[split_idx:(split_idx+num_test)]
         
        train_filenames = []
        for model in train_models:
            train_filenames += [self.data_dict[model_class][model]]
         
        valid_filenames = []
        for model in valid_models:
             valid_filenames += [self.data_dict[model_class][model]]
         
        test_filenames = []
        for model in test_models:
             test_filenames += [self.data_dict[model_class][model]]

        return train_filenames, valid_filenames, test_filenames
        
    def globalTrainValidTestSplit(self, num_classes = -1, 
                              num_models = -1,
                              valid_ratio = 0, 
                              test_ratio = 0, 
                              max_orientation_offset = float('inf')):
        if(num_classes < 1 or num_classes > len(self.data_dict)):
            num_classes = len(self.data_dict)
        
        class_names = np.array(list(self.data_dict.keys()))   
        np.random.shuffle(class_names)
         
        num_valid = np.ceil(valid_ratio*num_classes).astype(int)
        num_test = np.ceil(test_ratio*num_classes).astype(int)
        num_train = num_classes - num_valid - num_test

        train_classes = class_names[:num_train]
        split_idx = num_train
        valid_classes = class_names[split_idx:(split_idx+num_valid)]
        split_idx = split_idx+num_valid
        test_classes = class_names[split_idx:(split_idx+num_test)]

        train_filenames = []
        valid_model_filenames = []
        test_model_filenames = []
        for model_class in train_classes:
            trn_filenames, vld_filenames, tst_filenames = self.classTrainValidTestSplit(model_class, 
                                                                                    num_models=num_models,
                                                                                    valid_ratio = valid_ratio, 
                                                                                    test_ratio = test_ratio, 
                                                                                    max_orientation_offset=max_orientation_offset)
            train_filenames += trn_filenames
            valid_model_filenames += vld_filenames
            test_model_filenames += tst_filenames
         
        valid_class_filenames = []
        for model_class in valid_classes:
            filenames, _, _ = self.classTrainValidTestSplit(model_class, 
                                                     num_models=num_models, 
                                                     max_orientation_offset=max_orientation_offset)
            valid_class_filenames += filenames
         
        test_class_filenames = []
        for model_class in test_classes:
            filenames, _, _ = self.classTrainValidTestSplit(model_class, 
                                                      num_models=num_models, 
                                                      max_orientation_offset=max_orientation_offset)
            test_class_filenames += filenames

        return train_filenames, valid_model_filenames, valid_class_filenames, test_model_filenames, test_class_filenames
        