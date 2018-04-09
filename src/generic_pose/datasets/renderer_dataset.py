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

import glob

from generic_pose.utils.pose_renderer import renderView
import generic_pose.utils.transformations as tf_trans
from generic_pose.utils.data_preprocessing import (label2DenseWeights, 
                                                   resizeAndPad, 
                                                   uniformRandomQuaternion, 
                                                   transparentOverlay, 
                                                   quat2Uniform, 
                                                   randomQuatNear, 
                                                   quatDiff)

class PoseRendererDataSet(Dataset):
    def __init__(self, model_folders, img_size, 
                 background_filenames = None,
                 max_orientation_offset = None,
                 randomize_lighting = False,
                 camera_dist=2,
                 obj_dir_depth = 0,
                 classification = True,
                 num_bins = (50, 50, 25),
                 distance_sigma = 1):

        super(PoseRendererDataSet, self).__init__()
        
        self.img_size = img_size
        self.randomize_lighting = randomize_lighting
        self.model_filenames = []
        self.camera_dist = camera_dist
        self.obj_dir_depth = obj_dir_depth

        self.classification = classification        
        self.class_bins = num_bins
        self.distance_sigma = distance_sigma
        
        if(background_filenames is None):
            self.background_filenames = []
        else:
            self.background_filenames = background_filenames
        
        self.max_orientation_offset = max_orientation_offset
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.loop_truth = []
        
        if(type(model_folders) is list):
            files = []
            for folder in model_folders:
                files.extend(glob.glob(folder + '/**/*.obj', recursive=True))
        elif(type(model_folders) is str):
            files = glob.glob(model_folders + '/**/*.obj', recursive=True)
        else:
            raise AssertionError('Invalid model_folders type {}'.format(type(model_folders)))

        self.model_filenames = []
        self.model_classes = []
        
        self.class_idxs = {}
        self.class_list_idx = []
        
        for j, path in enumerate(files):
            [model_class, model] = path.split('/')[(-3-self.obj_dir_depth):(-1-self.obj_dir_depth)]
            
            if(model_class in self.class_idxs):
                self.class_idxs[model_class].append(j)
            else:
                self.class_idxs[model_class] = [j]

            self.model_filenames.append(path)
            self.model_classes.append(model_class)
            self.class_list_idx.append(len(self.class_idxs[model_class]) - 1)

    def __getitem__(self, index):
        index = index % len(self.model_filenames)
        if(len(self.loop_truth) < 2):
            return self.getPair(index)
        else:
            return self.getLoop(index, loop_truth=self.loop_truth)
            
    def getPair(self, index):
        quats, offset_quat, offset_class = self.nextRotation()
        images, model_filename = self.generateImages(index, quats)
        
        offset_quat = torch.from_numpy(offset_quat)
        offset_class = torch.from_numpy(offset_class)

        return images[0], images[1], offset_quat, offset_class, quats[0], model_filename

    def getLoop(self, index, loop_truth=[1,0,0]):
        images = []
        models = []
        model_files = []
        quats = []
        loop_len = len(loop_truth)

        if(loop_truth[0]):
            quat_prev = None
            loop_truth = loop_truth[1:]
        else:
            quat_prev = uniformRandomQuaternion()
        
        trans_test = []
        it1 = 0
        it2 = 0
        while(len(loop_truth) > 0):
            quats_render = []
            trans_next = []
            it1 += 1
            while(len(loop_truth) > 0):
                it2 += 1
                quats_next, offset_quat, offset_class = self.nextRotation(quat_prev)
                quats_render.extend(quats_next)
                trans_next.append(offset_quat)
                cont_loop = loop_truth[0]
                loop_truth = loop_truth[1:]
                if(not cont_loop):
                    break

            rendered_imgs, model_filename = self.generateImages(index, quats_render)
            
            quats.extend(quats_render)
            images.extend(rendered_imgs)
            trans_test.extend(trans_next)
            models.extend([index for _ in range(len(rendered_imgs))])
            model_files.extend([model_filename for _ in range(len(rendered_imgs))])

            if(len(loop_truth) > 0):
                index = index - np.random.randint(0, len(self.model_filenames)-1)
                quat_prev = quats_render[-1]

        trans = [quatDiff(quats[(j+1)%loop_len], quats[j]) for j in range(loop_len)]

        return images, trans, quats, models, model_files
        
    def generateRotations(self, origin_quat = None):
        if(origin_quat is None):
            origin_quat = uniformRandomQuaternion()
            
        if(self.max_orientation_offset is not None):
            query_quat, offset_quat = randomQuatNear(origin_quat, self.max_orientation_offset)
        else:
            query_quat = uniformRandomQuaternion()
            offset_quat = tf_trans.quaternion_multiply(query_quat, tf_trans.quaternion_conjugate(origin_quat))
        
        if(self.classification):
            offset_u = quat2Uniform(offset_quat)
            offset_class = label2DenseWeights(offset_u, (self.num_bins[0],self.num_bins[1],self.num_bins[2]), self.distance_sigma)
        else:
            offset_class = np.zeros(1)
            
        return origin_quat, query_quat, offset_quat, offset_class

    def nextRotation(self, origin_quat = None):
        quats = []
        if(origin_quat is None):
            origin_quat = uniformRandomQuaternion()
            quats.append(origin_quat)
            
        if(self.max_orientation_offset is not None):
            query_quat, offset_quat = randomQuatNear(origin_quat, self.max_orientation_offset)
        else:
            query_quat = uniformRandomQuaternion()
            offset_quat = tf_trans.quaternion_multiply(query_quat, tf_trans.quaternion_conjugate(origin_quat))
        
        quats.append(query_quat)
        
        if(self.classification):
            offset_u = quat2Uniform(offset_quat)
            offset_class = label2DenseWeights(offset_u, (self.num_bins[0],self.num_bins[1],self.num_bins[2]), self.distance_sigma)
        else:
            offset_class = np.zeros(1)
            
        return quats, offset_quat, offset_class

    def generateImages(self, index, render_quats):
        model_filename = self.model_filenames[index]

        # Will need to make distance random at some point
        rendered_imgs = renderView(model_filename, render_quats, self.camera_dist, standard_lighting=not self.randomize_lighting)
        
        images = [self.normalize(self.to_tensor(self.preprocessImages(img))) for img in rendered_imgs]
        return images, model_filename

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
        return len(self.model_filenames)*10000