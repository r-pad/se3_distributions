# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

from model_renderer.pose_renderer import BpyRenderer
print('\n'*5)

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
             color_scheme='Linux', call_pdb=1)

import numpy as np
import cv2
import os
import time
import torch
 
from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils import to_np, to_var
from generic_pose.utils.image_preprocessing import preprocessImages, unprocessImages
from generic_pose.eval.exemplar_pose_estimator import ExemplarDistPoseEstimator
#from generic_pose.eval.hyper_distance import ExemplarDistPoseEstimator 
from generic_pose.utils.pose_processing import quatAngularDiffBatch
from generic_pose.eval.posecnn_eval import evaluateQuat, getYCBThresholds

from quat_math import (quatAngularDiff,
                       quat2AxisAngle, 
                       quaternion_about_axis, 
                       quaternion_multiply, 
                       quaternion_inverse,
                       quaternion_from_matrix)

from generic_pose.eval.display_quaternions import plotQuatBall


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

def evaluateVideos(estimator, data_loader, num_samples = float('inf')):
    gt_ranks = []
    gt_outputs = []
    top_ranks = []
    top_dists = []
    video_quats = []
    agg_dists = []
    error_add = []

    dataset = data_loader.dataset 
    videos = dataset.videos.keys()

    points = dataset.getObjectPoints()
    cls = dataset.getObjectName()
    use_sym = cls == '024_bowl' or cls == '036_wood_block' or cls == '061_foam_brick'

    thresholds = getYCBThresholds()
    thresh = thresholds[dataset.obj]
    
    for vid in videos:
        dataset.data_filenames = dataset.videos[vid]

        gt_ranks.append([])
        gt_outputs.append([])
        top_ranks.append([])
        top_dists.append([])
        video_quats.append([])
        agg_dists.append(np.zeros(estimator.base_vertices.shape[0]))
        error_add.append([])
        #t = time.time()
        print('Video {} Size: {}'.format(vid, len(data_loader)))
        #import IPython; IPython.embed()
        for j, (imgs, _, quats, _, _) in enumerate(data_loader):
            #print('Load Time: ', time.time() - t)
            #print(vid, j)
            #t = time.time()
            dists = estimator.estimate(imgs[0], preprocess=False).detach()
            #dists = estimator.baseDistance(imgs[0], preprocess=False)
            #print('Estimation Time 0: ', time.time() - t)
            #t = time.time()
            num_verts = dists.shape[0]
            q_true = to_np(quats[0][0])
            true_dists = quatAngularDiffBatch(np.tile(q_true, (estimator.base_size,1)), estimator.base_vertices)
            top_idx = dists.max(0)[1]
            q_top = estimator.base_vertices[top_idx]
            true_idx = np.argmin(true_dists)
            err = evaluateQuat(q_true, q_top, points, use_sym=use_sym) 
            #print('Estimation Time 1: ', time.time() - t)
            #t = time.time()
            rank_gt = (torch.sort(dists, 0)[1] == int(true_idx)).nonzero()[0]
            #print('Estimation Time 2: ', time.time() - t)
            #t = time.time()
            rank_top = np.nonzero(np.argsort(true_dists) == top_idx)[0]
            #print('Estimation Time 3: ', time.time() - t)
            #t = time.time()
            output_gt = dists[true_idx]
            dist_top = true_dists[top_idx]*180/np.pi
            
            gt_ranks[-1].append(int(rank_gt[0]))
            gt_outputs[-1].append(float(output_gt))
            top_ranks[-1].append(int(rank_top[0]))
            top_dists[-1].append(dist_top)
            video_quats[-1].append(to_np(quats[0].flatten()))
            agg_dists[-1] += true_dists
            error_add[-1].append(err)
            #print('Evaluation Time: ', time.time() - t)
            #t = time.time()
            if num_samples is not None and j >= num_samples-1:
                break
        #gt_ranks[-1] /= j+1
        #gt_outputs[-1] /= j+1
        #top_ranks[-1] /= j+1
        #top_dists[-1] /= j+1
   
    return {'gt_ranks' : gt_ranks, 
            'top_ranks' : top_ranks,
            'gt_outputs' : gt_outputs, 
            'top_dists' : top_dists,
            'video_quats' : video_quats, 
            'agg_dists' : np.array(agg_dists),
            'error_add' : np.array(error_add),
            'threshold' : thresh}

    

def getArgs():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    #parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/ycb_finetune/035_power_drill/base/')
    #parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/shapenet/distance/shapenet_exp_fo20_th25/2018-08-03_02-29-12/weights/checkpoint_86000.pth')
    #parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/ycb_finetune/video_results/002_master_chef_can_')
    #parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/ycb_finetune/01_002_master_chef_can/shapenet_exp_fo20_th25/weight/checkpoint_5000.pth')
    parser.add_argument('--results_prefix', type=str) 
    parser.add_argument('--weight_file', type=str) 
    parser.add_argument('--data_folder', type=str, default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset')
    parser.add_argument('--target_object', type=int, default='1')

    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='exp')
    parser.add_argument('--num_samples', type=float, default=float('inf'))
    parser.add_argument('--base_level', type=int, default=2)

    args = parser.parse_args()
    return args

def main(weight_file,
         data_folder,
         target_object,
         results_prefix,
         model_type = 'alexnet',
         compare_type = 'sigmoid',
         base_level = 2,
         num_samples = 1,
         ):
    from torch.utils.data import DataLoader
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    import pickle

    print(results_prefix)
    dist_net = gen_pose_net(model_type.lower(), 
                            compare_type.lower(), 
                            output_dim = 1,
                            pretrained = False)

    load_state_dict(dist_net, weight_file)

    train_dataset = YCBDataset(data_dir=data_folder, 
                         image_set='train_split',
                         img_size=(224, 224),
                         obj=args.target_object)

    train_dataset.loop_truth = [1]
    train_dataset.splitImages()
    train_data_loader = DataLoader(train_dataset, num_workers=3, batch_size=1, shuffle=False)
  
    base_render_folder = os.path.join(data_folder,
                                     'base_renders',
                                      train_dataset.getObjectName(),
                                     '{}'.format(base_level))

    estimator = ExemplarDistPoseEstimator(dist_net, 
                                          renderer_transform_func = ycbRenderTransform,
                                          model_filename = train_dataset.getModelFilename(),
                                          base_render_folder = base_render_folder,
                                          base_level=base_level)

    results_dir = os.path.dirname(results_prefix)
    os.makedirs(results_dir, exist_ok=True)
    train_res = evaluateVideos(estimator, train_data_loader, num_samples = num_samples)
    del train_datase, train_data_loader
    
    with open(results_prefix + 'train_results.pkl', 'wb') as f:
        pickle.dump(train_res, f, -1)

    
    valid_dataset = YCBDataset(data_dir=data_folder, 
                         image_set='valid_split',
                         img_size=(224, 224),
                         obj=args.target_object)

    valid_dataset.loop_truth = [1]
    valid_dataset.splitImages()
    valid_data_loader = DataLoader(valid_dataset, num_workers=3, batch_size=1, shuffle=False)
    valid_res = evaluateVideos(estimator, valid_data_loader, num_samples = num_samples)
    del valid_datase, valid_data_loader
   
    with open(results_prefix + 'valid_results.pkl', 'wb') as f:
        pickle.dump(valid_res, f, -1)
    
    import IPython; IPython.embed()

    plotQuatBall(np.stack(train_res['video_quats']).reshape(-1,4), 
                 np.stack(train_res['gt_ranks']).flatten(), 
                 img_prefix = results_prefix + 'train_gt_rank')
    plotQuatBall(np.stack(valid_res['video_quats']).reshape(-1,4), 
                 np.stack(valid_res['gt_ranks']).flatten(), 
                 img_prefix = results_prefix + 'valid_gt_rank')
    import IPython; IPython.embed()

if __name__=='__main__':
    args = getArgs()
    main(weight_file = args.weight_file,
         data_folder = args.data_folder,
         target_object = args.target_object,
         results_prefix = args.results_prefix,
         model_type = args.model_type,
         compare_type = args.compare_type,
         base_level = args.base_level,
         num_samples = args.num_samples)
