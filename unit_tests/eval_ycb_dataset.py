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

        
def evaluate(obj, weights_dir, results_prefix, image_set, benchmark_dir):
    dataset = YCBDataset(data_dir=benchmark_dir,
                         image_set=image_set,
                         img_size=(224,224),
                         use_posecnn_masks = True,
                         obj=obj)

    dataset.loop_truth = None
    dataset.resample_on_none = False 
    loader = DataLoader(dataset, num_workers=4, batch_size=16, shuffle=False, pin_memory=True)
    base_render_folder = os.path.join(benchmark_dir,
                                      'base_renders',
                                      dataset.getObjectName(),
                                      '{}'.format(2))
    base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
    base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
  
  
    model = gen_pose_net('alexnet','sigmoid', output_dim = 1, pretrained = True, siamese_features = False)
    files = glob.glob(weights_dir + '**/checkpoint_*.pth', recursive=True)

    max_step = 0
    weight_file = ''
    for fn in files:
        step = int(fn.split('_')[-1][:-4])
        if(step >= max_step):
            max_step = step
            weight_file = fn

    assert weight_file != ''

    load_state_dict(model, weight_file)
    model.eval()
    model.cuda()
    optimizer = None #Adam(model.parameters(), lr=1e-5)
  
    metrics = evaluateDataset(model, loader, base_vertices, base_renders) 
    plotAccuracy(metrics, results_prefix)
    
    np.savez(results_prefix + 'metrics.npz', **metrics)
    sorted_indice = np.argsort(metrics['rank_gt'])
    best_file = open(results_prefix + 'best.txt', 'w')
    worst_file = open(results_prefix + 'worst.txt', 'w')
    for j in range(1):
        b_j = sorted_indice[j]
        w_j = sorted_indice[-(j+1)]
        best_file.write(dataset.data_filenames[b_j] + '\n')
        worst_file.write(dataset.data_filenames[w_j] + '\n')
        cv2.imwrite(results_prefix + 'worst_{:03d}_{}.png'.format(j, metrics['rank_gt'][w_j]), 
                    dataset.getImage(w_j, preprocess = False)[0])
        cv2.imwrite(results_prefix + 'worst_{:03d}_{}_p.png'.format(j, metrics['output_gt'][w_j]), 
                    unprocessImages(dataset.__getitem__(w_j)[0].unsqueeze(0))[0])
        cv2.imwrite(results_prefix + 'worst_{:03d}_{}_r.png'.format(j, metrics['top_idx'][w_j]), 
                    unprocessImages(base_renders[metrics['top_idx'][w_j]].unsqueeze(0))[0])

        cv2.imwrite(results_prefix + 'best_{:03d}_{}.png'.format(j, metrics['rank_gt'][b_j]), 
                    dataset.getImage(b_j, preprocess = False)[0])
        cv2.imwrite(results_prefix + 'best_{:03d}_{}_p.png'.format(j, metrics['output_gt'][b_j]), 
                    unprocessImages(dataset.__getitem__(b_j)[0].unsqueeze(0))[0])

    best_file.close()
    worst_file.close()
    return 

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('obj', type=int) 
    parser.add_argument('--results_prefix', type=str, default=None)
    parser.add_argument('--weights_dir', type=str, default=None)
    parser.add_argument('--image_set', type=str, default='valid_split')
    parser.add_argument('--benchmark_dir', type=str, 
        default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/')
   
    args = parser.parse_args()
    
    if(args.weights_dir is None):
        args.weights_dir = '/scratch/bokorn/results/ycb_finetune/model_{}/'.format(args.obj)
    if(args.results_prefix is None):
        args.results_prefix = '/home/bokorn/results/ycb_finetune/model_{}/{}_'.format(args.obj, args.image_set)
    
    evaluate(args.obj, args.weights_dir, args.results_prefix, args.image_set, args.benchmark_dir)

