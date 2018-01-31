# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob

from generic_pose.utils.data_preprocessing import label2DenseWeights, quat2Uniform, resizeAndPad, transparentOverlay

class PoseImageDataSet(Dataset):
    def __init__(self, img_size, data_folder,
                 background_filenames = None,
                 num_bins = (100,100,50),
                 distance_sigma = 5,
                 num_model_imgs = 250000):

        super(PoseImageDataSet, self).__init__()
        
        self.img_size = img_size
        self.num_bins = num_bins
        self.distance_sigma = distance_sigma
        self.data_folder = data_folder
        
        self.model_filenames = []
        self.camera_dist = 2
        
        self.class_bins = (100, 100, 50)
        if(background_filenames is None):
            self.background_filenames = []
        else:
            self.background_filenames = background_filenames
                
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        files = glob.glob(self.data_folder + '/**/*.npz', recursive=True)
        
        self.data_filenames = []
        for filename in files:
            self.data_filenames.append('.'.join(filename.split('.')[:-1]))
        
        if(num_model_imgs < len(self.data_filenames)):
            self.data_filenames = self.data_filenames[:num_model_imgs]
        
    def __getitem__(self, index):
        origin_img = cv2.imread(self.data_filenames[index] + '_origin.png', cv2.IMREAD_UNCHANGED)
        query_img = cv2.imread(self.data_filenames[index] + '_query.png', cv2.IMREAD_UNCHANGED)
        
        npzfile = np.load(self.data_filenames[index] + '.npz')
        offset_quat = npzfile['offset_quat']
        offset_u = quat2Uniform(offset_quat)
        origin_quat = np.load(self.data_filenames[index] + '_origin.npy')
        
        #Need to fix this
        model_class, model_name = self.data_filenames[index].split('/')[-1].split('_')[:2]
        model_file = os.path.join('/media/bokorn/ExtraDrive1/shapenetcore', model_class, model_name, 'model.obj')
        offset_class = label2DenseWeights(offset_u, (self.num_bins[0],self.num_bins[1],self.num_bins[2]), self.distance_sigma)
        #offset_w = npzfile['offset_w']
    
        origin_img = self.preprocessImages(origin_img)
        query_img = self.preprocessImages(query_img)        

        origin_img = self.normalize(self.to_tensor(origin_img))
        query_img  = self.normalize(self.to_tensor(query_img))
        
        offset_quat = torch.from_numpy(offset_quat)
        offset_class = torch.from_numpy(offset_class)
                                      
        return origin_img, query_img, offset_quat, offset_class, origin_quat, model_file
 
    def preprocessImages(self, image):
        num_background_files = len(self.background_filenames)
        if(num_background_files  > 0):
            bg_idx = np.random.randint(0, num_background_files)
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
