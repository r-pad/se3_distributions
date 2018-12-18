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
from generic_pose.utils.data_augmentation import augmentData
from quat_math import quatDiff, quatAngularDiff, angularPDF, invAngularPDF

class PoseImageDataset(Dataset):
    def __init__(self, 
                 img_size,
                 max_orientation_offset = None,
                 max_orientation_iters = 200,
                 rejection_thresh_angle = None,
                 background_filenames = None,
                 brightness_jitter = 0, contrast_jitter = 0, 
                 saturation_jitter = 0, hue_jitter = 0,
                 max_translation = None, max_scale = None,
                 rotate_image = False, 
                 max_num_occlusions = 0, max_occlusion_area = 0,
                 augmentation_prob = 0.0,
                 *args, **kwargs):

        super(PoseImageDataset, self).__init__()
        
        self.img_size = img_size
        self.max_orientation_offset = max_orientation_offset   
        self.max_orientation_iters = max_orientation_iters
        if(rejection_thresh_angle is not None):
            self.rejection_thresh = angularPDF(rejection_thresh_angle)
        else:
            self.rejection_thresh = None
        self.background_filenames = background_filenames
    
        self.loop_truth = [1,1]
        
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        self.hue_jitter = hue_jitter
        self.max_translation = max_translation
        self.max_scale = max_scale
        self.rotate_image = rotate_image 
        self.max_num_occlusions = max_num_occlusions
        self.max_occlusion_area = max_occlusion_area
        self.augmentation_prob = augmentation_prob

    def __getitem__(self, index):
        if(self.loop_truth is None):
        #    return self.getPair(index)
        #if(len(self.loop_truth) == ):
            return self.getData(index)
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

    def getData(self, index):
        img = self.getImage(index)
        while(img is None):
            print('None image at index {}'.format(index))
            index = np.random.randint(len(self))
            img = self.getImage(index)

        model = self.data_models[index]
        
        if(type(self.model_filenames) is dict):
            model_file = self.model_filenames[model]
        elif(type(self.model_filenames) is str):
            model_file = self.model_filenames
        else:
            model_file = ''
        
        quat = self.getQuat(index)

        return img, quat, model, model_file

    def getLoop(self, index, loop_truth=[1,0]):
        images = []
        models = []
        model_files = []
        quats = []
        trans = []
        
        for j, truth in enumerate(loop_truth):
            img, quat, model, model_file = self.getData(index)
            images.append(img)
            quats.append(quat)
            models.append(model)
            model_files.append(model_file)

            if(j < len(loop_truth)-1):
                if(truth or len(self.model_idxs.keys()) == 1):
                    index, _ = self.getPairIndex(model, quat)
                else:
                    model = np.random.choice(list(set(self.model_idxs.keys()) ^ set([model])))
                    index = np.random.choice(self.model_idxs[model])

            if(j > 0):
                trans.append(quatDiff(quats[j], quats[j-1]))

        trans.append(quatDiff(quats[0], quats[-1]))

        return images, trans, quats, models, model_files
 
    def preprocessImages(self, image, normalize_tensor = False, augment_img = False):
        if(augment_img and self.augmentation_prob > 0.0):
            image = augmentData(image, None, 
                                brightness_jitter = self.brightness_jitter,
                                contrast_jitter = self.contrast_jitter,
                                saturation_jitter = self.saturation_jitter,
                                hue_jitter = self.hue_jitter,
                                max_translation = self.max_translation,
                                max_scale = self.max_scale,
                                rotate_image = self.rotate_image,
                                max_num_occlusions = self.max_num_occlusions,
                                max_occlusion_area = self.max_occlusion_area,
                                transform_prob = self.augmentation_prob)[0]

        return preprocessImages([image], self.img_size,
                                normalize_tensors = normalize_tensor,
                                background_filenames = self.background_filenames)[0]

    def getQuat(self, index):
        raise NotImplementedError('getQuat must be implemented by child classes')
    def getImage(self, index):
        raise NotImplementedError('getImage must be implemented by child classes')
    def __len__(self):
        raise NotImplementedError('__len__ must be implemented by child classes')

