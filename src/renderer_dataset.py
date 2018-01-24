# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
import cv2
import numpy as np
#import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob
import tempfile
import os
import shutil

from pose_renderer import renderView
import transformations as tf
from data_preprocessing import label2Probs, resizeAndPad, uniformRandomQuaternion, transparentOverlay, quat2Uniform, randomQuatNear

from multiprocessing import Pool
import datetime
from itertools import repeat

def pool_render(args):
    print(args[0])
    renderView(args[0], args[1], args[2], filenames=args[3])

class PoseRendererDataSet(Dataset):
    def __init__(self, model_filenames, img_size, 
                 background_filenames = None,
                 max_orientation_offset = None,
                 prerender = True,
                 num_model_imgs = 250000,
                 data_folder = None,
                 save_data = False,
                 num_render_workers=20):

        super(PoseRendererDataSet, self).__init__()
        
        self.img_size = img_size
        self.prerendered = prerender
        self.data_folder = data_folder
        self.save_data = save_data
        
        self.model_filenames = []
        self.camera_dist = 2
        
        self.class_bins = 360
        if(background_filenames is None):
            self.background_filenames = []
        else:
            self.background_filenames = background_filenames
        
        self.max_orientation_offset = max_orientation_offset
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        if(model_filenames is not None):
            for idx, filename in enumerate(model_filenames):
                [model_class, model, model_file] = filename.split('/')[-3:]
    
                self.model_filenames.append(filename)
            
            self.per_model_workers = max(num_render_workers//len(self.model_filenames), 1)
            self.global_model_workers = num_render_workers//len(self.model_filenames)

            if self.prerendered:
                if self.data_folder is None:
                    self.data_folder = tempfile.mkdtemp()
                elif not os.path.exists(self.data_folder):
                    os.mkdir(self.data_folder)
        
                self.prerenderData(num_model_imgs, self.data_folder)

        elif(self.data_folder is not None):
            self.prerendered = True
            self.save_data = True
            files = glob.glob(self.data_folder + '/**/*.npz', recursive=True)
            
            self.data_filenames = []
            for filename in files:
                self.data_filenames.append('.'.join(filename.split('.')[:-1]))
            
            if(num_model_imgs < len(self.data_filenames)):
                self.data_filenames = self.data_filenames[:num_model_imgs]
        else:
            raise AssertionError('Both model_filenames and data_folder cannot be None')

    def __del__(self):
        if(not self.save_data):
            shutil.rmtree(self.data_folder)
            
    def __getitem__(self, index):
        if(self.prerendered):
            origin_img = cv2.imread(self.data_filenames[index] + '_origin.png', cv2.IMREAD_UNCHANGED)
            query_img = cv2.imread(self.data_filenames[index] + '_query.png', cv2.IMREAD_UNCHANGED)
            
            npzfile = np.load(self.data_filenames[index] + '.npz')
            offset_quat = npzfile['offset_quat']
            offset_u0 = npzfile['offset_u0']
            offset_u1 = npzfile['offset_u1']
            offset_u2 = npzfile['offset_u2']
            u_bins = npzfile['u_bins']
            
        else:
            origin_img, query_img, offset_quat, offset_u0, offset_u1, offset_u2, u_bins = self.generateData(index)

        origin_img = self.preprocessImages(origin_img)
        query_img = self.preprocessImages(query_img)        

        origin_img = self.normalize(self.to_tensor(origin_img))
        query_img  = self.normalize(self.to_tensor(query_img))
        
        offset_quat = torch.from_numpy(offset_quat)
        offset_u0 = torch.from_numpy(offset_u0)
        offset_u1 = torch.from_numpy(offset_u1)
        offset_u2 = torch.from_numpy(offset_u2)
        u_bins = torch.from_numpy(u_bins)
                                      
        return origin_img, query_img, offset_quat, offset_u0, offset_u1, offset_u2, u_bins

    def prerenderData(self, num_model_imgs, data_folder):
        self.data_filenames = []
        num_digits = len(str(num_model_imgs))
        #Probably can be speeded by pooling
        for index, model_file in enumerate(self.model_filenames):
            [model_class, model] = model_file.split('/')[-3:-1]            

            data_filenames = []            
            for j in range(num_model_imgs):
                data_filenames.append(os.path.join(self.data_folder, '{0}_{1}_{2:0{3}d}'.format(model_class, model, j, num_digits)))
            self.generateDataBatch(index, data_filenames)            
            
            self.data_filenames.extend(data_filenames)

    def generateRotations(self):
        origin_quat = uniformRandomQuaternion()
        if(self.max_orientation_offset is not None):
            query_quat, offset_quat = randomQuatNear(origin_quat, self.max_orientation_offset)
        else:
            query_quat = uniformRandomQuaternion()
            offset_quat = tf.quaternion_multiply(query_quat, tf.quaternion_conjugate(origin_quat))
            
        u = quat2Uniform(offset_quat)
        u_bins = np.round(np.array(u)*self.class_bins)
        offset_u0 = label2Probs(u_bins[0], self.class_bins)
        offset_u1 = label2Probs(u_bins[1], self.class_bins)
        offset_u2 = label2Probs(u_bins[2], self.class_bins)
        
        return origin_quat, query_quat, offset_quat, offset_u0, offset_u1, offset_u2, u_bins
 
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
 
    def generateData(self, index):
        model_filename = self.model_filenames[index]

        origin_quat, query_quat, offset_quat, offset_u0, offset_u1, offset_u2, u_bins = self.generateRotations()

        render_quats = []        
        render_quats.append(origin_quat)
        render_quats.append(query_quat)
            
        # Will need to make distance random at some point
        rendered_imgs = renderView(model_filename, render_quats, self.camera_dist)
        
        origin_img = self.preprocessImages(rendered_imgs[0])
        query_img = self.preprocessImages(rendered_imgs[1])

        return origin_img, query_img, offset_quat, offset_u0, offset_u1, offset_u2, u_bins

    def generateDataBatch(self, index, filenames):
        model_filename = self.model_filenames[index]

        render_quats = []
        image_filenames = []
        
        for data_filename in filenames:
            image_filenames.append(data_filename + '_origin.png')
            image_filenames.append(data_filename + '_query.png')            

            data = self.generateRotations()            
            render_quats.append(data[0])
            render_quats.append(data[1])
            np.save(data_filename + '_origin.npy',  data[0])
            np.save(data_filename + '_query.npy',  data[1])

            np.savez(data_filename + '.npz', 
                     offset_quat = data[2], 
                     offset_u0 =  data[3], 
                     offset_u1 =  data[4],
                     offset_u2 =  data[5], 
                     u_bins =  data[6])
            
        # Will need to make distance random at some point
        pool = Pool(self.per_model_workers)
        #pool_model_filename = [model_filename]*self.per_model_workers
        #pool_camera_dist = [self.camera_dist]*self.per_model_workers
        pool_render_quats = np.split(np.array(render_quats), self.per_model_workers)
        pool_image_filenames = np.split(np.array(image_filenames), self.per_model_workers)

        args = zip(repeat(model_filename), pool_render_quats, repeat(self.camera_dist), pool_image_filenames)
        for idx, return_code in enumerate(pool.imap(pool_render, args)):
            print('{} Rendering Model {} group {}'.format(datetime.datetime.now().time(), model_filename, idx))
            

    def __len__(self):
        if(self.prerendered):
            return len(self.data_filenames)
        else:
            return len(self.model_filenames)