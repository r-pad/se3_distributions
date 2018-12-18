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
 
from generic_pose.bbTrans.discretized4dSphere import S3Grid
from generic_pose.datasets.numpy_dataset import NumpyImageDataset
from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils import to_np, to_var
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.utils.image_preprocessing import preprocessImages, unprocessImages
from generic_pose.losses.quaternion_loss import quaternionAngles
from generic_pose.eval.exemplar_pose_estimator import ExemplarDistPoseEstimator 
from generic_pose.utils.pose_processing import quatAngularDiffBatch

from quat_math import (quatAngularDiff,
                       quat2AxisAngle, 
                       quaternion_about_axis, 
                       quaternion_multiply, 
                       quaternion_inverse,
                       quaternion_from_matrix)

def evaluateRefinement(estimator, data_loader, num_samples = float('inf')):
    gt_ranks = []
    gt_outputs = []
    top_ranks = []
    top_dists = []
    eval_times = []
    angle_vec = []
    output_vec = []
    for j, (imgs, _, quats, _, _) in enumerate(data_loader):
        t = time.time()
        dists = estimator.baseDistance(imgs[0], preprocess=False) 
        eval_times.append(time.time()-t)
        #print('Estimation time: {}s'.format(round(eval_times[-1], 4)))
        dists = -to_np(dists.detach())
        num_verts = dists.shape[0]
        q_true = to_np(quats[0][0])
        true_dists = quatAngularDiffBatch(np.tile(q_true, (estimator.base_size,1)), estimator.base_vertices)

        top_idx = np.argmin(dists)
        true_idx = np.argmin(true_dists)

        angle_vec.extend(true_dists.tolist())
        output_vec.extend(dists.tolist())

        rank_gt = np.nonzero(np.argsort(dists) == true_idx)[0]
        rank_top = np.nonzero(np.argsort(true_dists) == top_idx)[0]
        output_gt = dists[true_idx]
        dist_top = true_dists[top_idx]*180/np.pi
        gt_ranks.append(rank_gt)
        gt_outputs.append(output_gt)
        top_ranks.append(rank_top)
        top_dists.append(dist_top)
        if num_samples is not None and j > num_samples-1:
            break
    
    data = np.array([angle_vec, output_vec])
    print('Mean Estimation Time: {} sec'.format(np.mean(eval_times)))
    return np.array(gt_ranks), np.array(top_ranks),\
           np.array(gt_outputs), np.array(top_dists),\
           data

def getArgs():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/ycb_finetune/035_power_drill/base/')
    parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/shapenet/distance/shapenet_exp_fo20_th25/2018-08-03_02-29-12/weights/checkpoint_86000.pth')
    #parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/ycb_finetune/035_power_drill/eval/')
    #parser.add_argument('--weight_file', type=str, default='/home/bokorn/results/ycb_finetune/035_power_drill/weights/checkpoint_10000.pth')
    parser.add_argument('--data_folder', type=str, default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset')
    parser.add_argument('--dataset_type', type=str, default='ycb')

    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='exp')

    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--inverse_distance', dest='inverse_distance', action='store_true')

    parser.add_argument('--base_level', type=int, default=2)

    args = parser.parse_args()
    return args

def main(weight_file,
         data_folder,
         dataset_type,
         results_prefix,
         model_type = 'alexnet',
         compare_type = 'sigmoid',
         inverse_distance = True,
         base_level = 2,
         num_samples = 1,
         ):
    from torch.utils.data import DataLoader
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    
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
        from generic_pose.datasets.numpy_dataset import NumpyImageDataset
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

    res = evaluateRefinement(estimator, data_loader, num_samples = num_samples)
    
    results_dir = os.path.dirname(results_prefix)
    os.makedirs(results_dir, exist_ok=True)
    np.savez(results_prefix + 'distance.npz',
            gt_ranks = res[0],
            top_ranks = res[1],
            gt_outputs = res[2],
            top_dists = res[3],
            data = res[4])

    import IPython; IPython.embed()

if __name__=='__main__':
    args = getArgs()
    main(weight_file = args.weight_file,
         data_folder = args.data_folder,
         dataset_type = args.dataset_type,
         results_prefix = args.results_prefix,
         model_type = args.model_type,
         compare_type = args.compare_type,
         inverse_distance = args.inverse_distance,
         base_level = args.base_level,
         num_samples = args.num_samples)
