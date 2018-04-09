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

from generic_pose.utils.pose_renderer import renderView
import generic_pose.utils.transformations as tf_trans
from generic_pose.utils.data_preprocessing import (label2DenseWeights, 
                                                   resizeAndPad, 
                                                   uniformRandomQuaternion, 
                                                   transparentOverlay, 
                                                   quat2Uniform, 
                                                   randomQuatNear, 
                                                   quatDiff)

from multiprocessing import Pool
#import multiprocessing
#from concurrent.futures import ProcessPoolExecutor

import datetime
from itertools import repeat
from functools import partial

_datasets_dir = os.path.dirname(os.path.abspath(__file__))

#
#class NoDaemonProcess(multiprocessing.Process):
#    # make 'daemon' attribute always return False
#    def _get_daemon(self):
#        return False
#    def _set_daemon(self, value):
#        pass
#    daemon = property(_get_daemon, _set_daemon)
#
#class Pool(multiprocessing.Pool):
#    Process = NoDaemonProcess

def pool_render(args):
    print(args[0])
    renderView(args[0], args[1], args[2], filenames=args[3], standard_lighting=args[4])

def generateRotations(max_orientation_offset = None):
    origin_quat = uniformRandomQuaternion()
    if(max_orientation_offset is not None):
        query_quat, offset_quat = randomQuatNear(origin_quat, max_orientation_offset)
    else:
        query_quat = uniformRandomQuaternion()
        offset_quat = tf_trans.quaternion_multiply(query_quat, tf_trans.quaternion_conjugate(origin_quat))
        
    return origin_quat, query_quat, offset_quat
 
def generateDataBatch(model_filename, data_folder, num_model_imgs,
                      obj_dir_depth = 0,
                      camera_dist = 2, 
                      randomize_lighting = False, 
                      max_orientation_offset=None):

    num_digits = len(str(num_model_imgs))
    [model_class, model] = model_filename.split('/')[(-3-obj_dir_depth):(-1-obj_dir_depth)]
    os.makedirs(os.path.join(data_folder, '{0}/{1}'.format(model_class, model)), exist_ok=True)
            
    filenames = []
    for j in range(num_model_imgs):
        filenames.append(os.path.join(data_folder, '{0}/{1}/{0}_{1}_{2:0{3}d}'.format(model_class, model, j, num_digits)))
     
    render_quats = []
    image_filenames = []
    
    for data_filename in filenames:
        image_filenames.append(data_filename + '_origin.png')
        image_filenames.append(data_filename + '_query.png')            

        data = generateRotations(max_orientation_offset)
        render_quats.append(data[0])
        render_quats.append(data[1])
        np.save(data_filename + '_origin.npy',  data[0])
        np.save(data_filename + '_query.npy',  data[1])

        np.savez(data_filename + '.npz', 
                 offset_quat = data[2])

    args = [model_filename, render_quats, camera_dist, image_filenames, not randomize_lighting]
    print('{} Rendering Model {}'.format(datetime.datetime.now().time(), model_filename))
    pool_render(args)
    return filenames

class BatchRenderer(object):
    def __init__(self, model_filenames, 
                 data_folder, 
                 img_size, 
                 background_filenames = None,
                 max_orientation_offset = None,
                 prerender = True,
                 randomize_lighting = False,
                 camera_dist=2,
                 num_model_imgs = 250000,
                 num_render_workers = 20,
                 obj_dir_depth = 0):

        super(BatchRenderer, self).__init__()
        
        self.img_size = img_size
        self.prerendered = prerender
        self.data_folder = data_folder
        self.randomize_lighting = randomize_lighting
        self.model_filenames = []
        self.camera_dist = camera_dist
        self.obj_dir_depth = obj_dir_depth
        
        if(background_filenames is None):
            self.background_filenames = []
        else:
            self.background_filenames = background_filenames
        
        self.max_orientation_offset = max_orientation_offset
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        self.model_filenames = model_filenames
        self.num_render_workers = num_render_workers
        
        #self.per_model_workers = max(num_render_workers//len(self.model_filenames), 1)
        #self.global_model_workers = num_render_workers//self.per_model_workers

        if self.data_folder is None:
            self.data_folder = tempfile.mkdtemp()
        elif not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)

        if(len(self.model_filenames) < self.num_render_workers):
            self.renderDataImage(num_model_imgs, self.data_folder)
        else:
            self.renderDataModel(num_model_imgs, self.data_folder)

    def renderDataImage(self, num_model_imgs, data_folder):
        self.data_filenames = []
        num_digits = len(str(num_model_imgs))
        #Probably can be speeded by pooling
        for index, model_file in enumerate(self.model_filenames):
            [model_class, model] = model_file.split('/')[(-3-self.obj_dir_depth):(-1-self.obj_dir_depth)]
            os.makedirs(os.path.join(self.data_folder, '{0}/{1}'.format(model_class, model)), exist_ok=True)
            
            data_filenames = []      
            for j in range(num_model_imgs):
                data_filenames.append(os.path.join(self.data_folder, '{0}/{1}/{0}_{1}_{2:0{3}d}'.format(model_class, model, j, num_digits)))
                
            self.generateDataBatch(index, data_filenames)            
            
            self.data_filenames.extend(data_filenames)

    def renderDataModel(self, num_model_imgs, data_folder):
        self.data_filenames = []
        #Probably can be speeded by pooling

        batch_render = partial(generateDataBatch, 
                               data_folder = self.data_folder,
                               num_model_imgs = num_model_imgs,
                               obj_dir_depth = self.obj_dir_depth,
                               camera_dist = self.camera_dist, 
                               randomize_lighting = self.randomize_lighting, 
                               max_orientation_offset=self.max_orientation_offset)
                       
        #pool = Pool(self.global_model_workers)
        pool = Pool(self.num_render_workers)
        for j, filenames in enumerate(pool.imap(batch_render, self.model_filenames)):
            self.data_filenames.extend(filenames)
            print('Process {}: {}'.format(j, len(filenames)))
        
    def generateRotations(self):
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

        origin_quat, query_quat, offset_quat, offset_class = self.generateRotations()

        render_quats = []        
        render_quats.append(origin_quat)
        render_quats.append(query_quat)
            
        # Will need to make distance random at some point
        rendered_imgs = renderView(model_filename, render_quats, self.camera_dist, standard_lighting=not self.randomize_lighting)
        
        origin_img = self.preprocessImages(rendered_imgs[0])
        query_img = self.preprocessImages(rendered_imgs[1])

        return origin_img, query_img, offset_quat, offset_class, origin_quat, query_quat, model_filename

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
                     offset_class =  data[3])
            
        # Will need to make distance random at some point
        pool = Pool(self.num_render_workers)
        #pool_model_filename = [model_filename]*self.per_model_workers
        #pool_camera_dist = [self.camera_dist]*self.per_model_workers
        pool_render_quats = np.array_split(np.array(render_quats), self.per_model_workers)
        pool_image_filenames = np.array_split(np.array(image_filenames), self.per_model_workers)

        args = zip(repeat(model_filename), pool_render_quats, repeat(self.camera_dist), pool_image_filenames, repeat(not self.randomize_lighting))
        for idx, return_code in enumerate(pool.imap(pool_render, args)):
            print('{} Rendering Model {} group {}'.format(datetime.datetime.now().time(), model_filename, idx))            

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_file', type=str, default=None)
    parser.add_argument('--valid_data_file', type=str, default=None)
    parser.add_argument('--obj_dir_depth', type=int, default=0)
    parser.add_argument('--background_data_file', type=str, default=None)
    parser.add_argument('--max_orientation_offset', type=float, default=None)
    parser.add_argument('--randomize_lighting', dest='randomize_lighting', action='store_true')
    parser.add_argument('--camera_dist', type=float, default=2)

    #parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_render_workers', type=int, default=20)
    #parser.add_argument('--num_render_workers', type=int, default=20)

    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--num_train_imgs', type=int, default=50000)
    parser.add_argument('--num_valid_imgs', type=int, default=5000)
    parser.add_argument('--train_data_folder', type=str, default=None)
    parser.add_argument('--valid_data_folder', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()

    if(args.train_data_file is not None):
        with open(args.train_data_file, 'r') as f:    
            train_filenames = f.read().split()
    else:
        train_filenames = None
    
    if(args.valid_data_file is not None):
        with open(args.valid_data_file, 'r') as f:    
            valid_filenames = f.read().split()    
    else:
        valid_filenames = None
        
    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
                
    BatchRenderer(model_filenames = train_filenames,
                  data_folder = args.train_data_folder,
                  img_size = (args.width,args.height),
                  background_filenames = background_filenames,
                  max_orientation_offset = args.max_orientation_offset,
                  randomize_lighting = args.randomize_lighting,
                  camera_dist = args.camera_dist,                        
                  num_model_imgs = args.num_train_imgs,
                  num_render_workers = args.num_render_workers,
                  obj_dir_depth = args.obj_dir_depth)

    BatchRenderer(model_filenames = valid_filenames,
                  data_folder = args.valid_data_folder,                  
                  img_size = (args.width,args.height),
                  background_filenames = background_filenames,
                  max_orientation_offset = args.max_orientation_offset,
                  randomize_lighting = args.randomize_lighting,
                  camera_dist = args.camera_dist,
                  num_model_imgs = args.num_valid_imgs,
                  num_render_workers = args.num_render_workers,
                  obj_dir_depth = args.obj_dir_depth)

if __name__=='__main__':
    main()
