# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import glob

from generic_pose.utils.data_preprocessing import label2DenseWeights, quat2Uniform, resizeAndPad, transparentOverlay
import generic_pose.utils.transformations as tf_trans

_datasets_dir = os.path.dirname(os.path.abspath(__file__))

class PoseImageDataSet(Dataset):
    def __init__(self, data_folders,
                 img_size,
                 model_filenames = None,
                 background_filenames = None,
                 classification = True,
                 num_bins = (50, 50, 25),
                 distance_sigma = 5):

        super(PoseImageDataSet, self).__init__()
        
        self.img_size = img_size
        
        self.classification = classification        
        self.num_bins = num_bins
        self.distance_sigma = distance_sigma

        self.model_filenames = model_filenames
        self.background_filenames = background_filenames
                    
        self.class_bins = num_bins
    
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        if(type(data_folders) is list):
            files = []
            for folder in data_folders:
                files.extend(glob.glob(folder + '/**/*.npy', recursive=True))
        elif(type(data_folders) is str):
            files = glob.glob(data_folders + '/**/*.npy', recursive=True)
        else:
            raise AssertionError('Invalid data_folders type {}'.format(type(data_folders)))
            
        self.data_filenames = []
        self.data_models = []
        self.data_classes = []

        self.model_idxs = {}
        self.class_idxs = {}
        
        for j, path in enumerate(files):
            [model_class, model, filename] = '.'.join(path.split('.')[:-1]).split('/')[-3:]

            if(model_class in self.class_idxs):
                self.class_idxs[model_class].append(model)
            else:
                self.class_idxs[model_class] = [model]

            if(model in self.model_idxs):
                self.model_idxs[model].append(j)
            else:
                self.model_idxs[model] = [j]

            self.data_filenames.append('.'.join(path.split('.')[:-1]))
            self.data_models.append(model)
            self.data_classes.append(model_class)

    def __getitem__(self, index):
        model = self.data_models[index]
        assert len(self.model_idxs[model]) > 1, "Model must have > 1 view (model: {})".format(model)
        query_idx = self.model_idxs[model][np.random.randint(0, len(self.model_idxs[model]))]

        # Images
        origin_img = cv2.imread(self.data_filenames[index] + '.png', cv2.IMREAD_UNCHANGED)
        query_img = cv2.imread(self.data_filenames[query_idx] + '.png', cv2.IMREAD_UNCHANGED)
        
        origin_img = self.preprocessImages(origin_img)
        query_img = self.preprocessImages(query_img)        

        origin_img = self.normalize(self.to_tensor(origin_img))
        query_img  = self.normalize(self.to_tensor(query_img))

        # Rotations
        origin_quat = np.load(self.data_filenames[index] + '.npy')
        query_quat = np.load(self.data_filenames[query_idx] + '.npy')

        offset_quat = tf_trans.quaternion_multiply(query_quat, tf_trans.quaternion_conjugate(origin_quat))
        offset_u = quat2Uniform(offset_quat)
        
        if(type(self.model_filenames) is dict):
            model_file = self.model_filenames[model]
        elif(type(self.model_filenames) is str):
            model_file = self.model_filenames
        else:
            model_file = os.join(_datasets_dir, 'jet.obj')

        if(self.classification):
            offset_class = label2DenseWeights(offset_u, (self.num_bins[0],self.num_bins[1],self.num_bins[2]), self.distance_sigma)
        else:
            offset_class = np.zeros(1)
        
        offset_quat = torch.from_numpy(offset_quat)
        offset_class = torch.from_numpy(offset_class)                          
        
        return origin_img, query_img, offset_quat, offset_class, origin_quat, model_file
 
    def preprocessImages(self, image):
        if(self.background_filenames is not None):
            bg_idx = np.random.randint(0, len(self.background_filenames))
            background = cv2.imread(self.background_filenames[bg_idx])
        else:
            background = None

        if (len(image.shape) == 2):
            image = np.expand_dims(image, axis=2)
        
        if(image.shape[2] == 4):
            image = transparentOverlay(image, background)
            
        image = resizeAndPad(image, self.img_size)
        
        return image

    def __len__(self):
        return len(self.data_filenames)