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

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
from generic_pose.losses.distance_utils import evaluateDataset 
from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.eval.plot_accuracy import plotAccuracy
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

def getPrefix(dataset, index):
    return os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index])
        
def debug(obj, results_prefix, image_set, benchmark_dir):
    dataset = YCBDataset(data_dir=benchmark_dir,
                         image_set=image_set,
                         img_size=(224,224),
                         use_sym = True,
                         use_posecnn_masks = False,
                         obj=obj)

    dataset.loop_truth = None
    dataset.resample_on_none = False 
    loader = DataLoader(dataset, num_workers=4, batch_size=16, shuffle=False)
    #base_render_folder = os.path.join(benchmark_dir,
    #                                  'base_renders',
    #                                  dataset.getObjectName(),
    #                                  '{}'.format(2))
    #base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
    #base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))

    import IPython; IPython.embed() 
    #model = gen_pose_net('alexnet','sigmoid', output_dim = 1, pretrained = True, siamese_features = False)
    return 

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('obj', type=int) 
    parser.add_argument('--results_prefix', type=str, default=None)
    parser.add_argument('--image_set', type=str, default='train_split')
    parser.add_argument('--benchmark_dir', type=str, 
        default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/')
   
    args = parser.parse_args()
    
    if(args.results_prefix is None):
        args.results_prefix = '/home/bokorn/results/test/model_{}_{}_'.format(args.obj, args.image_set)
    
    debug(args.obj, args.results_prefix, args.image_set, args.benchmark_dir)

