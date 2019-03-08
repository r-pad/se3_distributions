# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torch.optim import Adam

import glob
import cv2
import os
import time
import numpy as np

from generic_pose.eval.pose_grid_estimator import PoseGridEstimator
from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
from generic_pose.utils.image_preprocessing import preprocessImages, unprocessImages

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    
    
class MultiObjectPoseEstimator(object):
    def __init__(self, weight_paths, render_paths):
        self.pose_estimators = []

        for weight_file, render_folder in zip(weight_paths, render_paths):
            model = gen_pose_net('alexnet','sigmoid', output_dim = 1, 
                                 pretrained = True, siamese_features = False)
            load_state_dict(model, weight_file)
            model.eval()
            model.cuda()
            self.pose_estimators.append(PoseGridEstimator(render_folder, model))
    
    def __call__(self, img, class_idx):
        poses, mode_idxs = self.pose_estimators[class_idx].getPose(img)
        renders = self.pose_estimators[class_idx].grid_renders[mode_idxs]
        return poses, renders

def evaluate(image_dir, results_dir, class_names, weights_dir, renders_dir, image_ext = 'jpg'):
    pose_estimator = MultiObjectPoseEstimator(weights_dir, renders_dir)
    image_filenames = glob.glob(image_dir + '**/*.' + image_ext, recursive = True) 
    
    for cat_name in class_names.keys():
        if not os.path.exists(os.path.join(results_dir, cat_name)):
            os.makedirs(os.path.join(results_dir, cat_name))
    
    for fn in image_filenames:
        base_name = os.path.splitext(os.path.basename(fn))[0]
        img_class_name = fn.split('/')[-2]
        img_class_idx = class_names[img_class_name]
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        #boarder = max(img.shape[:2])//2
        #img = np.pad(img,((boarder, boarder),(boarder, boarder),(0,0)), 
        #        mode = 'constant', constant_values=0)
        poses, renders = pose_estimator(img, img_class_idx) 

        cv2.imwrite(os.path.join(results_dir, img_class_name, base_name + '.{}'.format(image_ext)), img)
        img = preprocessImages([img], (224,224),
                               normalize_tensors = True,
                               background = None,
                               background_filenames = None, 
                               remove_mask = True, 
                               vgg_normalize = False)
        cv2.imwrite(os.path.join(results_dir, img_class_name, base_name+'_prep.{}'.format(image_ext)), 
                    unprocessImages(img)[0])
        cv2.imwrite(os.path.join(results_dir, img_class_name, base_name+'_rend.{}'.format(image_ext)), 
                    unprocessImages(renders.unsqueeze(0))[0])
    return 

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--image_dir', type=str, default = '/home/bokorn/data/surgical_tools/images/')
    parser.add_argument('--class_names', type=str, nargs='+', default = ['hemostat','scalpel','scissors'])
    parser.add_argument('--results_dir', type=str, default = '/home/bokorn/results/sugical/test/')
    parser.add_argument('--weights_dir', type=str, nargs='+', 
        default = ['/scratch/bokorn/results/sugical/hemostat_black_jitter/2019-03-07_02-03-40/weights/checkpoint_200000.pth',
                   '/scratch/bokorn/results/sugical/scalpel_gray_jitter/2019-03-07_02-03-43/weights/checkpoint_200000.pth',
                   '/scratch/bokorn/results/sugical/scissor_metal_jitter/2019-03-07_02-03-48/weights/checkpoint_200000.pth'])
    parser.add_argument('--renders_dir', type=str, nargs='+', 
        default = ['/scratch/bokorn/data/demo/surgical/hemostat_black/base_renders/2/',
                   '/scratch/bokorn/data/demo/surgical/scalpel_gray/base_renders/2/',
                   '/scratch/bokorn/data/demo/surgical/scissor_metal/base_renders/2/'])
#    parser.add_argument('--results_dir', type=str, default = '/home/bokorn/results/sugical_far/test/')
#    parser.add_argument('--weights_dir', type=str, nargs='+', 
#        default = ['/home/bokorn/pretrained/surgical/hemostat_black_jitter/2019-02-21_06-58-49/weights/checkpoint_200000.pth',
#                   '/home/bokorn/pretrained/surgical/scalpel_gray_jitter/2019-02-21_06-58-36/weights/checkpoint_200000.pth',
#                   '/home/bokorn/pretrained/surgical/scissor_metal_jitter/2019-02-21_06-58-40/weights/checkpoint_200000.pth'])
#    parser.add_argument('--renders_dir', type=str, nargs='+', 
#        default = ['/scratch/bokorn/data/demo/surgical_far/hemostat_black/base_renders/2/',
#                   '/scratch/bokorn/data/demo/surgical_far/scalpel_gray/base_renders/2/',
#                   '/scratch/bokorn/data/demo/surgical_far/scissor_metal/base_renders/2/'])


    parser.add_argument('--image_ext', type=str, default = 'png')
    args = parser.parse_args()
     
    class_names = {}
    for j, cls in enumerate(args.class_names):
        class_names[cls] = j

    evaluate(image_dir = args.image_dir, 
             results_dir = args.results_dir,
             class_names = class_names, 
             weights_dir = args.weights_dir, 
             renders_dir = args.renders_dir, 
             image_ext = args.image_ext)

