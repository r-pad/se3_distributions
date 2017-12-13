# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:44:12 2017

@author: bokorn
"""
import cv2
import numpy as np
#import os
import torch
from torch.utils.data import Dataset
import glob

#import quaternions as quat
from pose_renderer import camera2quat
import transformations as tf

def calc_viewloss_vec(size, sigma):
    band    = np.linspace(-1*size, size, 1 + 2*size, dtype=np.int16)
    vec     = np.linspace(-1*size, size, 1 + 2*size, dtype=np.float)
    prob    = np.exp(-1 * abs(vec) / sigma)
    prob    = prob / np.sum(prob)

    return band, prob

def label_to_probs(angle, angle_bins = 360, band_width = 7, sigma=5):
    '''
    Returns three arrays for the viewpoint labels, one for each rotation axis.
    A label is given by angle
    :return:
    '''
    # Calculate object multiplier
    angle = int(angle)
    label = np.zeros(angle_bins, dtype=np.float)
    
    # calculate probabilities
    band, prob = calc_viewloss_vec(band_width, sigma)

    for i in band:
        ind = np.mod(angle + i + 360, 360)
        label[ind] = prob[i + band_width]

    return label

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


#filepath = os.path.realpath(__file__)
#filedir = os.path.abspath(os.path.join(filepath, os.pardir))
#project_dir = os.path.abspath(os.path.join(filedir, os.pardir))



class PoseDataLoader(Dataset):
    def __init__(self, data_dir, img_size):
        super(PoseDataLoader, self).__init__()
        self.img_size = img_size
        self.data_dir = data_dir
        self.filenames = []
        self.quats = []
        #self.eulers = []
        self.models = []
        self.model_idxs = {}
        self.euler_bins = 360
        self.max_offset = np.pi/2
        
        idx = 0;
        files = glob.glob(self.data_dir + '/**/*.png', recursive=True)
        for filename in files:                
            [model_name, model, azimuth, elevation, tilt, depth] = filename.split('/')[-1].split('.')[-2].split('_')
            azimuth = azimuth[1:]
            elevation = elevation[1:]
            tilt = tilt[1:]
            if(model in self.model_idxs):
                self.model_idxs[model].append(idx)
            else:
                self.model_idxs[model] = [idx]
                
            self.filenames.append(filename)
            q_blender = camera2quat(float(azimuth), 
                                    float(elevation), 
                                    float(tilt))
            self.quats.append(np.roll(q_blender,-1))
#            self.eulers.append(np.array([float(azimuth), 
#                               float(elevation), 
#                               float(tilt)]))
                               
            self.models.append(model)
            idx += 1
        
#    def __init__(self, label_dir, img_size):
#        super(PoseDataLoader, self).__init__()
#        self.img_size = img_size
#        self.label_dir = label_dir
#        self.filenames = []
#        self.quats = []
#        self.eulers = []
#        self.models = []
#        self.model_idxs = {}
#        self.euler_bins = 360
#        
#        idx = 0;
#        for file in os.listdir(label_dir):            
#            if file.endswith("_train.txt"):
#                with open(os.path.join(label_dir, file)) as f:
#                    model_name = file.split('.')[0]
#                    data = f.read().split('\n')
#                    for d in data:
#                        if(len(d) == 0):
#                            break
#                        [filename, class_idx, azimuth, elevation, tilt] = d.split(' ')
#                        model = filename.split('/')[-2]
#                        
#                        if(model in self.model_idxs):
#                            self.model_idxs[model].append(idx)
#                        else:
#                            self.model_idxs[model] = [idx]
#                            
#                        self.filenames.append(filename)
#                        self.quats.append(quat.euler2quat(float(azimuth)*np.pi/180.0, 
#                                                          float(elevation)*np.pi/180.0, 
#                                                          float(tilt)*np.pi/180.0))
#                        self.euler_bins.append(np.array(float(azimuth), 
#                                           float(elevation), 
#                                           float(tilt)))
#                                           
#                        self.models.append(model)
#                        idx += 1

    def generateCluttered(self, filename, model, 
                          num_objects = 1, 
                          percent_occluded = 0.0):
        # Model image is PNG with alpha channel in 3
        model_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)        
        composite_img = np.zeros(self.img_size + (3,))
        
        for j in range(num_objects):
            while True:
                model_idx = np.random.randint(0, len(self.model_idxs))    
                if (model_idx != model):
                    break
            clutter_idx = self.model_idxs[model_idx][np.random.randint(0, len(self.model_idxs[model_idx]))]
            clutter_filename = self.filenames[clutter_idx]
            clutter_img = cv2.imread(clutter_filename, cv2.IMREAD_UNCHANGED)
            

        
        


    def __getitem__(self, index):
        origin_filename = self.filenames[index]
        origin_pose = self.quats[index]
#        origin_euler = self.eulers[index]

        model = self.models[index]
        
        query_idx = self.model_idxs[model][np.random.randint(0, len(self.model_idxs[model]))]
        assert len(self.model_idxs[model]) > 1, "Model must have > 1 view (model: {})".format(model)

        orientation_diff = float('inf')
        
        while orientation_diff > self.max_offset:
            diff_not_found = True
            while diff_not_found:
                #print(model, query_idx, index)
                query_idx = self.model_idxs[model][np.random.randint(0, len(self.model_idxs[model]))]    
                if(query_idx != index):
                    diff_not_found = False
                
            query_filename = self.filenames[query_idx]
            query_pose = self.quats[query_idx]
    #        query_euler = self.eulers[query_idx]
            
            origin_img = cv2.imread(origin_filename)
            query_img = cv2.imread(query_filename)
            
            if (len(origin_img.shape) == 2):
                origin_img = np.expand_dims(origin_img, axis=2)
                query_img = np.expand_dims(query_img, axis=2)
            
            origin_img = resizeAndPad(origin_img, self.img_size)
            query_img = resizeAndPad(query_img, self.img_size)
            
            origin_img = origin_img.astype('float32')
            query_img = query_img.astype('float32')
            origin_img = np.rollaxis(origin_img, 2)
            query_img = np.rollaxis(query_img, 2)
            
            d_quat = tf.quaternion_multiply(query_pose, 
                                            tf.quaternion_conjugate(origin_pose))
        
            orientation_diff = 2.0*np.arccos(d_quat[3])
        
        conj_q = torch.from_numpy(tf.quaternion_conjugate(d_quat))
        angles = tf.euler_from_quaternion(d_quat)
        d_euler = np.round(np.array(angles)*self.euler_bins/(2.*np.pi))
        
        d_azim = torch.from_numpy(label_to_probs(d_euler[0], self.euler_bins))
        d_elev = torch.from_numpy(label_to_probs(d_euler[1], self.euler_bins))
        d_tilt = torch.from_numpy(label_to_probs(d_euler[2], self.euler_bins))
           
        d_quat = torch.from_numpy(d_quat)

        return origin_img, query_img, conj_q, d_quat, d_azim, d_elev, d_tilt, d_euler

    def __len__(self):
        return len(self.filenames)
