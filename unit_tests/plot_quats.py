# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
import numpy as np
import os
from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.eval.display_quaternions import plotQuatBall, plotQuatKDE

def main():
    print('Plot Quats')
    results_dir = '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/metrics/'
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    
    valid_dataset = YCBDataset(data_dir=benchmark_folder,
                               image_set='valid_split',
                               img_size=(224,224),
                               obj=1)
    valid_dataset.loop_truth = None
    
    train_dataset = YCBDataset(data_dir=benchmark_folder,
                               image_set='train_split',
                               img_size=(224,224),
                               use_syn_data = True,
                               obj=1)
    train_dataset.loop_truth = None
    
 
    valid_quats = valid_dataset.quats
    train_quats = train_dataset.quats
    print(len(train_quats))
    valid_size = valid_quats.shape[0]
    train_size = train_quats.shape[0]
    #plotQuatKDE(valid_quats, img_prefix=results_dir+'kde_valid') 
    #plotQuatKDE(train_quats, img_prefix=results_dir+'kde_train') 
    plotQuatBall(valid_quats, img_prefix=results_dir+'valid_') 
    plotQuatBall(train_quats, img_prefix=results_dir+'train_syn_') 
    plotQuatBall(np.concatenate([train_quats, valid_quats]), 
        dists=np.concatenate([np.ones(train_size), np.zeros(valid_size)]),
        img_prefix=results_dir+'all_syn_') 
    print('Done')
    
if __name__=='__main__':
    main()

