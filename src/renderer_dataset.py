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
import glob
import tempfile
import os
import shutil

#import quaternions as quat
from pose_renderer import renderView
import transformations as tf
from data_preprocessing import label2Probs, resizeAndPad, uniformRandomQuaternion, transparentOverlay, quat2Uniform

class PoseRendererDataSet(Dataset):
    def __init__(self, model_filenames, img_size, 
                 background_filenames = None,
                 max_orientation_offset = None,
                 prerender = True,
                 num_model_imgs = 250000,
                 data_folder = None,
                 save_data = False):

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
        
        if(model_filenames is not None):
            for idx, filename in enumerate(model_filenames):
                [model_class, model, model_file] = filename.split('/')[-3:]
    
                self.model_filenames.append(filename)
                
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

            for filename in files:
                self.data_filenames.append(filename.split('.')[:-1])
                
        else:
            raise AssertionError('Both model_filenames and data_folder cannot be None')

    def __del__(self):
        if(not self.save_data):
            shutil.rmtree(self.data_folder)
            
    def __getitem__(self, index):
        if(self.prerendered):
            origin_img = cv2.imread(self.data_filenames[index] + '_origin.png')
            query_img = cv2.imread(self.data_filenames[index] + '_query.png')
            
            if (len(origin_img.shape) == 2):
                origin_img = np.expand_dims(origin_img, axis=2)
                query_img = np.expand_dims(query_img, axis=2)
        
            origin_img = resizeAndPad(origin_img, self.img_size)
            query_img = resizeAndPad(query_img, self.img_size)
            
            npzfile = np.load(self.data_filenames[index] + '.npz')
#            print('*'*10)
#            print(type(npzfile['offset_quat']))
#            print(npzfile['offset_quat'])
#            print('-'*10)
#            print(type(npzfile['offset_u0']))
#            print(npzfile['offset_u0'])
#            print('*'*10)
#            offset_quat = torch.from_numpy(npzfile['offset_quat'])
#            offset_u0 = torch.from_numpy(npzfile['offset_u0'])
#            offset_u1 = torch.from_numpy(npzfile['offset_u1'])
#            offset_u2 = torch.from_numpy(npzfile['offset_u2'])
#            u_bins = torch.from_numpy(npzfile['u_bins'])
#            
            offset_quat = npzfile['offset_quat']
            offset_u0 = npzfile['offset_u0']
            offset_u1 = npzfile['offset_u1']
            offset_u2 = npzfile['offset_u2']
            u_bins = npzfile['u_bins']
            
        else:
            origin_img, query_img, offset_quat, offset_u0, offset_u1, offset_u2, u_bins = self.generateData(index)

        origin_img = origin_img.astype('float32')
        query_img = query_img.astype('float32')
        origin_img = np.rollaxis(origin_img, 2)
        query_img = np.rollaxis(query_img, 2)

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
            self.generateData(index, data_filenames)            
            
            self.data_filenames.extend(data_filenames)
            
    def generateData(self, index, filenames = [None]):
        model_filename = self.model_filenames[index]

        render_quats = []

        for fn in filenames:
            origin_quat = uniformRandomQuaternion()
            if(self.max_orientation_offset is not None):
                offset_axis = np.random.randn(3)
                offset_axis /= np.linalg.norm(offset_axis)
                offset_angle = self.max_orientation_offset * np.random.rand()
                offset_quat = tf.quaternion_about_axis(offset_angle, offset_axis)
                query_quat = tf.quaternion_multiply(origin_quat, offset_quat)
            else:
                query_quat = uniformRandomQuaternion()
                offset_quat = tf.quaternion_multiply(query_quat, tf.quaternion_conjugate(origin_quat))
                
            u = quat2Uniform(offset_quat)
            u_bins = np.round(np.array(u)*self.class_bins/(2.*np.pi))
            offset_u0 = label2Probs(u_bins[0], self.class_bins)
            offset_u1 = label2Probs(u_bins[1], self.class_bins)
            offset_u2 = label2Probs(u_bins[2], self.class_bins)
            
            render_quats.append(origin_quat)
            render_quats.append(query_quat)
            
            if(fn is not None):         
                np.savez(fn + '.npz', 
                         offset_quat = offset_quat, 
                         offset_u0 = offset_u0, 
                         offset_u1 = offset_u1,
                         offset_u2 = offset_u2, 
                         u_bins = u_bins)
            
        # Will need to make distance random at some point
        rendered_imgs = renderView(model_filename, render_quats, self.camera_dist)
        
        num_background_files = len(self.background_filenames)
        for j, fn in enumerate(filenames):
            if(num_background_files  > 0):
                o_idx = np.random.randint(0, num_background_files)
                q_idx = np.random.randint(0, num_background_files)
                origin_background = cv2.imread(self.background_filenames[o_idx])
                query_background = cv2.imread(self.background_filenames[q_idx])
            else:
                origin_background = None
                query_background = None
                
            origin_img = transparentOverlay(rendered_imgs[2*j], origin_background)
            query_img = transparentOverlay(rendered_imgs[2*j+1], query_background)
    
            origin_img = resizeAndPad(origin_img, self.img_size)
            query_img = resizeAndPad(query_img, self.img_size)
            
            if(fn is not None):
                cv2.imwrite(fn + '_origin.png', origin_img)
                cv2.imwrite(fn + '_query.png', query_img)
        
        if(filenames == [None]):
            return origin_img, query_img, offset_quat, offset_u0, offset_u1, offset_u2, u_bins

    def __len__(self):
        if(self.prerendered):
            return len(self.data_filenames)
        else:
            return len(self.model_filenames)