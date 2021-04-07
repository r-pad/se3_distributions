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
 
from se3_distributions.bbTrans.discretized4dSphere import S3Grid
from se3_distributions.datasets.numpy_dataset import NumpyImageDataset
from se3_distributions.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from se3_distributions.utils import to_np, to_var
from se3_distributions.training.finetune_distance_utils import evaluateRenderedDistance
from se3_distributions.utils.image_preprocessing import preprocessImages, unprocessImages
from se3_distributions.losses.quaternion_loss import quaternionAngles
from se3_distributions.eval.exemplar_pose_estimator import ExemplarDistPoseEstimator 
from se3_distributions.utils.pose_processing import quatAngularDiffBatch
from se3_distributions.utils.tetra_utils import topTetrahedron
from quat_math import (quatAngularDiff,
                       quat2AxisAngle, 
                       quaternion_about_axis, 
                       quaternion_multiply, 
                       quaternion_inverse,
                       quaternion_from_matrix)

from se3_distributions.eval.display_quaternions import plotQuatBall

def evaluateRefinement(estimator, data_loader, num_samples = float('inf'), 
                       refinement_steps = 3,
                       results_prefix = '/home/bokorn/results/test/'):
    eval_times = []
    gt_ranks = []
    gt_outputs = []
    top_ranks = []
    top_dists = []
    #tetras = estimator.grid.GetTetras(2)
    
    for j, (imgs, _, quats, _, _) in enumerate(data_loader):
        cv2.imwrite(results_prefix + '{}_real.png'.format(j), unprocessImages(imgs[0])[0])
        t = time.time()
        dists = estimator.baseDistance(imgs[0], preprocess=False) 
        eval_times.append(time.time()-t)
        #print('Estimation time: {}s'.format(round(eval_times[-1], 4)))
        dists = dists
        num_verts = dists.shape[0]
        q_true = to_np(quats[0][0])
        true_dists = quatAngularDiffBatch(np.tile(q_true, (estimator.base_size,1)), estimator.base_vertices)
        top_idx = np.argmax(dists)
        print(np.argmax(dists))
        true_idx = np.argmin(true_dists)

        gt_ranks.append([])
        gt_outputs.append([])
        top_ranks.append([])
        top_dists.append([])

        
        rank_gt = np.nonzero(np.argsort(dists) == true_idx)[0]
        rank_top = np.nonzero(np.argsort(true_dists) == top_idx)[0]
        output_gt = dists[true_idx]
        dist_top = true_dists[top_idx]*180/np.pi
       
        gt_ranks[-1].append(rank_gt)
        gt_outputs[-1].append(output_gt)
        top_ranks[-1].append(rank_top)
        top_dists[-1].append(dist_top)

        #top_tetra_idx = topTetrahedron(-dists, tetras, np.max)    
        #true_tetra_idx = topTetrahedron(-true_dists, tetras, np.max)
        num_verts = estimator.grid.vertices.shape[0]
        for k in range(refinement_steps):

            plotQuatBall(estimator.grid.vertices, estimator.vert_dists, gt_quat = q_true,
                     img_prefix = results_prefix + '{}_{}_'.format(j, k))
            cv2.imwrite(results_prefix + '{}_{}_top.png'.format(j, k), unprocessImages(estimator.renders[top_idx:top_idx+1])[0])
            estimator.refine()
            print('{} new vertices created at level {}'.format(estimator.grid.vertices.shape[0]-num_verts, k+1))
            num_verts = estimator.grid.vertices.shape[0]
            dists = estimator.vert_dists
            true_dists = quatAngularDiffBatch(np.tile(q_true, (estimator.grid.vertices.shape[0],1)), estimator.grid.vertices)
            top_idx = np.argmax(dists)
            true_idx = np.argmin(true_dists)
            print('Top distance: {}'.format(true_dists[top_idx]))
            rank_gt = np.nonzero(np.argsort(dists) == true_idx)[0]
            rank_top = np.nonzero(np.argsort(true_dists) == top_idx)[0]
            output_gt = dists[true_idx]
            dist_top = true_dists[top_idx]*180/np.pi
           
            gt_ranks[-1].append(rank_gt)
            gt_outputs[-1].append(output_gt)
            top_ranks[-1].append(rank_top)
            top_dists[-1].append(dist_top)
        plotQuatBall(estimator.grid.vertices, estimator.vert_dists, gt_quat = q_true, 
                     img_prefix = results_prefix + '{}_{}_'.format(j, refinement_steps))
        #import IPython; IPython.embed()
        estimator.reset()
        if num_samples is not None and j > num_samples-1:
            break
    
    print('Mean Estimation Time: {} sec'.format(np.mean(eval_times)))
    return np.array(gt_ranks), np.array(top_ranks),\
           np.array(gt_outputs), np.array(top_dists)

def getArgs():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/ycb_finetune/035_power_drill/present/')
    parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/ycb_finetune/035_power_drill/weights/checkpoint_10000.pth')
    #parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/shapenet/distance/shapenet_exp_fo20_th25/2018-08-03_02-29-12/weights/checkpoint_86000.pth')
    #parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/ycb_finetune/002_master_chef_can/refine/')
    #parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/ycb_finetune/002_master_chef_can/shapenet_exp_fo20_th25/weight/checkpoint_5000.pth')
    parser.add_argument('--data_folder', type=str, default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset')
    parser.add_argument('--dataset_type', type=str, default='ycb')

    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='exp')

    parser.add_argument('--num_samples', type=int, default=25)

    parser.add_argument('--base_level', type=int, default=2)

    args = parser.parse_args()
    return args

def main(weight_file,
         data_folder,
         dataset_type,
         results_prefix,
         model_type = 'alexnet',
         compare_type = 'sigmoid',
         base_level = 2,
         num_samples = 1,
         ):
    from torch.utils.data import DataLoader
    from se3_distributions.models.pose_networks import gen_pose_net, load_state_dict
    
    print(results_prefix)
    t = time.time()
    dist_net = gen_pose_net(model_type.lower(), 
                            compare_type.lower(), 
                            output_dim = 1,
                            pretrained = False)

    load_state_dict(dist_net, weight_file)
    #dist_net.load_state_dict(torch.load(weight_file))
    #print('Weights load time: {}s'.format(round(time.time()-t, 2)))
    t = time.time()
    if(dataset_type.lower() == 'numpy'):
        from se3_distributions.datasets.numpy_dataset import NumpyImageDataset
        dataset = NumpyImageDataset(data_folders=data_folder,
                                    img_size = (224, 224))
    else:
        target_object = 15
        dataset = YCBDataset(data_dir=data_folder, 
                             image_set='valid_split',
                             img_size=(224, 224),
                             obj=target_object)
    dataset.loop_truth = [1]
    data_loader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
    #print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))
    
    base_render_folder = os.path.join(data_folder,
                                     'base_renders',
                                      dataset.getObjectName(),
                                     '{}'.format(base_level))

    t = time.time()
    estimator = ExemplarDistPoseEstimator(dist_net, 
                                          renderer_transform_func = ycbRenderTransform,
                                          model_filename = dataset.getModelFilename(),
                                          base_render_folder = base_render_folder,
                                          base_level=base_level)
    #print('Estimator initialization time: {}s'.format(round(time.time()-t, 2)))

    res = evaluateRefinement(estimator, data_loader, num_samples = num_samples,
                            refinement_steps = 5, results_prefix = results_prefix)
    
    results_dir = os.path.dirname(results_prefix)
    os.makedirs(results_dir, exist_ok=True)
    np.savez(results_prefix + 'distance.npz',
            gt_ranks = res[0],
            top_ranks = res[1],
            gt_outputs = res[2],
            top_dists = res[3])

    import IPython; IPython.embed()

if __name__=='__main__':
    args = getArgs()
    main(weight_file = args.weight_file,
         data_folder = args.data_folder,
         dataset_type = args.dataset_type,
         results_prefix = args.results_prefix,
         model_type = args.model_type,
         compare_type = args.compare_type,
         base_level = args.base_level,
         num_samples = args.num_samples)
