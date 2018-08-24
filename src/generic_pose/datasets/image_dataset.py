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

from generic_pose.utils.image_preprocessing import preprocessImages
from quat_math import quatDiff, quatAngularDiff, angularPDF, invAngularPDF

class PoseImageDataset(Dataset):
    def __init__(self, 
                 img_size,
                 crop_percent = None,
                 max_orientation_offset = None,
                 max_orientation_iters = 200,
                 rejection_thresh_angle = None,
                 background_filenames = None,
                 *args, **kwargs):

        super(PoseImageDataset, self).__init__()
        
        self.img_size = img_size
        self.crop_percent = crop_percent
        self.max_orientation_offset = max_orientation_offset   
        self.max_orientation_iters = max_orientation_iters
        if(rejection_thresh_angle is not None):
            self.rejection_thresh = angularPDF(rejection_thresh_angle)
        else:
            self.rejection_thresh = None
        self.background_filenames = background_filenames
    
        self.loop_truth = [1,1]

    def __getitem__(self, index):
        if(self.loop_truth is None):
            return self.getPair(index)
        else:
            return self.getLoop(index, loop_truth=self.loop_truth)

    def getPairIndex(self, model, origin_quat):
        assert len(self.model_idxs[model]) > 0, "Model must have > 1 view (model: {})".format(model)
        for j in range(self.max_orientation_iters):
            query_idx = self.model_idxs[model][np.random.randint(len(self.model_idxs[model]))]
            query_quat = self.getQuat(query_idx)
            if(self.max_orientation_offset is None and self.rejection_thresh is None):
                break
            else:
                angle_diff = quatAngularDiff(query_quat, origin_quat)
                if(self.max_orientation_offset is not None 
                   and angle_diff < self.max_orientation_offset):
                    break
                if(self.rejection_thresh is not None
                   and invAngularPDF(angle_diff, self.rejection_thresh) > np.random.rand()):
                    break
        return query_idx, query_quat

    def getPair(self, index):
        origin_quat = self.getQuat(index)
        origin_img = self.getImage(index)

        model = self.data_models[index]
        query_idx, query_quat = self.getPairIndex(model, origin_quat)
        
        query_img = self.getImage(query_idx)

        offset_quat = quatDiff(query_quats, origin_quats)
        return origin_img, query_img, offset_quat

    def getLoop(self, index, loop_truth=[1,0]):
        images = []
        models = []
        model_files = []
        quats = []
        trans = []
        
        for j, truth in enumerate(loop_truth):
            img = self.getImage(index)
            while(img is None):
                print('None image at index {}'.format(index))
                index = np.random.randint(len(self))
                img = self.getImage(index)

            model = self.data_models[index]
            images.append(img)
            models.append(model)
            
            if(type(self.model_filenames) is dict):
                model_files.append(self.model_filenames[model])
            elif(type(self.model_filenames) is str):
                model_files.append(self.model_filenames)
            else:
                model_files.append('')
            
            quats.append(self.getQuat(index))

            if(j < len(loop_truth)-1):
                if(truth or len(self.model_idxs.keys()) == 1):
                    index, _ = self.getPairIndex(model, quats[-1])
                else:
                    model = np.random.choice(list(set(self.model_idxs.keys()) ^ set([model])))
                    index = np.random.choice(self.model_idxs[model])

            if(j > 0):
                trans.append(quatDiff(quats[j], quats[j-1]))

        trans.append(quatDiff(quats[0], quats[-1]))

        return images, trans, quats, models, model_files
 
    def preprocessImages(self, image, normalize_tensor = False):
        return preprocessImages([image], self.img_size,
                                normalize_tensors = normalize_tensor,
                                background_filenames = self.background_filenames,
                                crop_percent = self.crop_percent)[0]

    def getQuat(self, index):
        raise NotImplementedError('getQuat must be implemented by child classes')
    def getImage(self, index):
        raise NotImplementedError('getImage must be implemented by child classes')
    def __len__(self):
        raise NotImplementedError('__len__ must be implemented by child classes')

