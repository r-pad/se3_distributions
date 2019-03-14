#!/usr/bin/env python
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
import scipy.io as sio

from generic_pose.eval.pose_grid_estimator import PoseGridEstimator
from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
from generic_pose.utils import to_np
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
            if torch.cuda.is_available():
                model.cuda()
            self.pose_estimators.append(PoseGridEstimator(render_folder, model))

    def __call__(self, img, class_idx):
        poses, mode_idxs = self.pose_estimators[class_idx].getPose(img)
        renders = self.pose_estimators[class_idx].grid_renders[mode_idxs]
        return poses, renders

    def getDistances(self, img, class_idx):
        dists = self.pose_estimators[class_idx].getDistances(img)
        return dists

 
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
        dists = pose_estimator.getDistances(img, img_class_idx) 
        poses, renders = pose_estimator(img, img_class_idx) 
        output_prefix = os.path.join(results_dir, img_class_name, base_name)
        sio.savemat(output_prefix + '.mat', {'outputs':to_np(dists), 'quat_gt':poses,
                'verts':pose_estimator.pose_estimators[img_class_idx].grid_vertices}) 
        cv2.imwrite(output_prefix + '.{}'.format(image_ext), img)
        img = preprocessImages([img], (224,224),
                               normalize_tensors = True,
                               background = None,
                               background_filenames = None, 
                               remove_mask = True, 
                               vgg_normalize = False)
        cv2.imwrite(output_prefix + '_prep.{}'.format(image_ext), 
                    unprocessImages(img)[0])
        cv2.imwrite(output_prefix + '_rend.{}'.format(image_ext), 
                    unprocessImages(renders.unsqueeze(0))[0])
    return 

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--image_dir', type=str, default = '/home/bokorn/data/surgical/images/coco/masked_images/')
    parser.add_argument('--class_names', type=str, nargs='+', default = ['hemostat','scalpel','scissors'])
    parser.add_argument('--results_dir', type=str, default = '/home/bokorn/data/surgical/results/')
    parser.add_argument('--weights_dir', type=str, nargs='+', 
        default = ['/home/bokorn/data/surgical/weights/hemostat_black/weights.pth',
                   '/home/bokorn/data/surgical/weights/scalpel_gray/weights.pth', 
                   '/home/bokorn/data/surgical/weights/scissor_metal/weights.pth',])
    
    parser.add_argument('--renders_dir', type=str, nargs='+', 
        default = ['/home/bokorn/data/surgical/weights/hemostat_black/',
                   '/home/bokorn/data/surgical/weights/scalpel_gray/', 
                   '/home/bokorn/data/surgical/weights/scissor_metal/',])

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

