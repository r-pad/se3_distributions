# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
from model_renderer.pose_renderer import BpyRenderer

import cv2
import os
import time
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils.ycb_utils import getOcclusionPercentage, setYCBCameraMatrix 
from generic_pose.utils.pose_processing import quatAngularDiffDot

def main():
    augmentation_prob = 0.0
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    dataset = YCBDataset(data_dir=benchmark_folder,
                         image_set='train_split',
                         img_size=(224,224),
                         obj=1,
                         use_syn_data = True)

    dataset.loop_truth = None
 
    train_quats = np.load(os.path.join(benchmark_folder, 'quats', 
        dataset.getObjectName()+'_train_split_syn_quats.npy'))
 
    renderer = BpyRenderer()
    setYCBCameraMatrix(renderer)
    renderer.loadModel(dataset.getModelFilename())
    results_dir = '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/metrics/'
 
    occlusion = []
    min_distance = []
    mean_h = []
    mean_s = []
    mean_v = []
    for idx in range(len(dataset)):
        img = dataset.getImage(idx, preprocess = False)
        if(img is None):
            print('Image at {} is None'.format(idx))
            continue
    return

    if(False):
        occlusion.append(getOcclusionPercentage(dataset, renderer, idx))
        min_distance.append(quatAngularDiffDot(np.expand_dims(dataset.getQuat(idx), 0), 
            train_quats).min()*180/np.pi)
        mask = img[:,:,3]
        hsv = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        mean_h.append(h[mask==255].mean())
        mean_s.append(s[mask==255].mean())
        mean_v.append(v[mask==255].mean())
    
    np.savez(results_dir + 'train_w_syn_metrics.npz', 
             occlusion = occlusion,
             min_distance = min_distance,
             mean_h = mean_h,
             mean_s = mean_s,
             mean_v = mean_v)

    print("DONE")
    return 

if __name__=='__main__':
    main()

