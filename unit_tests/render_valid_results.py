# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import os
import cv2
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils.image_preprocessing import unprocessImages, transparentOverlay

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

def main():
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    dataset = YCBDataset(data_dir=benchmark_folder,
                                            image_set='valid_split',
                                            img_size=(224,224),
                                            use_posecnn_masks = True,
                                            obj=1)
    dataset.loop_truth = None
    dataset_metrics = np.load("/home/bokorn/results/ycb_finetune/01_002_master_chef_can/metrics/valid_metrics.npz")
    performance_metrics = np.load("/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_1.0/valid_metrics.npz")
 
    loader = DataLoader(dataset, num_workers=4, batch_size=16, shuffle=False)
    base_render_folder = os.path.join(benchmark_folder,
                                      'base_renders',
                                      dataset.getObjectName(),
                                      '{}'.format(2))
    base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
    base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
  
  
    #model = gen_pose_net('alexnet','sigmoid', output_dim = 1, pretrained = True, siamese_features = True)
    #weight_file = '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_1.0/checkpoint_532000.pth'
    #load_state_dict(model, weight_file)
    #model.eval()
    #model.cuda()
   
    results_dir = '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_1.0/v_images/'
    for j in range(len(dataset)):
        rank_gt = performance_metrics['rank_gt'][j]
        dist_top = performance_metrics['dist_top'][j]
        output_gt = performance_metrics['output_gt'][j]
        occ = dataset_metrics['occlusion'][j]
        
        img = transparentOverlay(dataset.getImage(j, preprocess = False)).astype(np.uint8)
        top_img = unprocessImages(base_renders[performance_metrics['top_idx'][j]].unsqueeze(0))[0].astype(np.uint8)
        true_img = unprocessImages(base_renders[performance_metrics['true_idx'][j]].unsqueeze(0))[0].astype(np.uint8)

        ax = plt.subplot2grid((3,4), (0,1), colspan=3, rowspan=3)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.xlabel(dataset.data_filenames[j])
        
        ax = plt.subplot2grid((3,4), (0,0))
        ax.barh([1,2,3,4], 
                [rank_gt/len(base_vertices), output_gt, dist_top/180, occ], 
                tick_label=['rank_gt', 'output', 'dist', 'occ'])
        ax.axis([0,1,0,5])
        
        ax.text(0,1,'{}'.format(rank_gt), va='center')
        ax.text(0,2,'{:.3f}'.format(output_gt), va='center')
        ax.text(0,3,'{}'.format(int(dist_top)), va='center')
        ax.text(0,4,'{:.3f}'.format(occ), va='center')
       
        ax = plt.subplot2grid((3,4), (1,0))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(cv2.cvtColor(top_img, cv2.COLOR_BGRA2RGB))
        plt.xlabel('Top Render')
        
        ax = plt.subplot2grid((3,4), (2,0))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(cv2.cvtColor(true_img, cv2.COLOR_BGRA2RGB))
        plt.xlabel('GT Render')
 
        plt.savefig(results_dir + 'valid_{:04d}.png'.format(j))
        plt.gcf().clear()

    import IPython; IPython.embed() 
    return 

if __name__=='__main__':
    main()

