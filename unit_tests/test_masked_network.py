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

import cv2
import os
import time
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.models.pose_networks import gen_pose_net, load_state_dict, add_mask_input
from generic_pose.losses.distance_utils import evaluateDataset 
from generic_pose.utils.image_preprocessing import unprocessImages

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

        
def main():
    augmentation_prob = 0.0
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    evaluate_valid = False
    
    if(evaluate_valid):
        dataset = YCBDataset(data_dir=benchmark_folder,
                                            image_set='valid_split',
                                            img_size=(224,224),
                                            obj=1, remove_mask = False)
    else:
        dataset = YCBDataset(data_dir=benchmark_folder,
                                            image_set='train_split',
                                            img_size=(224,224),
                                            obj=1,
                                            use_syn_data = True, remove_mask = False)

    dataset.loop_truth = None
                
    loader = DataLoader(dataset, num_workers=4, batch_size=16, shuffle=False)
    base_render_folder = os.path.join(benchmark_folder,
                                      'base_renders',
                                      dataset.getObjectName(),
                                      '{}'.format(2))
    base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
    base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
  
  
    model = gen_pose_net('alexnet','sigmoid', output_dim = 1, pretrained = True, siamese_features = False)
    weight_file = '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_1.0/checkpoint_532000.pth'
    load_state_dict(model, weight_file)
    add_mask_input(model) 
    model.eval()
    model.cuda()
    optimizer = None #Adam(model.parameters(), lr=1e-5)
   
    print('Dataset Size:', len(loader))
    import IPython; IPython.embed()
    metrics = evaluateDataset(model, loader, base_vertices, base_renders)

if __name__=='__main__':
    main()

