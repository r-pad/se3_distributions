# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""
import bpy
import numpy as np
import cv2

from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.eval.hyper_distance import ExemplarDistPoseEstimator 
from quat_math import (quatAngularDiff,
                       quat2AxisAngle, 
                       quaternion_about_axis, 
                       quaternion_multiply, 
                       quaternion_inverse,
                       quaternion_from_matrix)

from generic_pose.utils import to_np

delta_quat = np.array([.5,.5,.5,.5])
#flip_x = np.array([-1,1,1,1])
#def convertQuat(q):
#    q_flip = quaternion_inverse(q.copy())
#    q_flip[2] *= -1
#    return quaternion_multiply(delta_quat, q_flip)

from generic_pose.eval.evaluate_render_distance import evaluateDistanceNetwork

def getArgs():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--weight_file', type=str)
    parser.add_argument('--data_folder', type=str)

    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='exp')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--inverse_distance', dest='inverse_distance', action='store_true')

    parser.add_argument('--base_level', type=int, default=0)

    args = parser.parse_args()

    return args

def main(weight_file,
         data_folder,
         results_dir,
         model_type = 'alexnet',
         compare_type = 'sigmoid',
         inverse_distance = True,
         base_level = 2,
         ):
    import os
    import time
    import cv2
    import torch
    from torch.utils.data import DataLoader
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    
    t = time.time()
    dist_net = gen_pose_net(model_type.lower(), 
                            compare_type.lower(), 
                            output_dim = 1,
                            pretrained = False)

    load_state_dict(dist_net, weight_file)
    #print('Weights load time: {}s'.format(round(time.time()-t, 2)))

    from generic_pose.datasets.sixdc_dataset import SixDCDataset
    
    t = time.time()
    data_loader = DataLoader(SixDCDataset(data_dir=data_folder,
                                          img_size = (224, 224)),
                             num_workers=0, 
                             batch_size=1, 
                             shuffle=False)

    data_loader.dataset.loop_truth = [1]
    #print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))
    os.makedirs(results_dir, exist_ok=True)
    for obj in [1,2,5,6,8,9,11,12]:
        print('Object {:02d}'.format(obj))
        data_loader.dataset.setSequence('02', obj)
        model_filename = data_loader.dataset.model_filenames[obj]
        model_scale = data_loader.dataset.model_scales[obj]
        estimator = ExemplarDistPoseEstimator(model_filename, dist_net, 
                                              use_bpy_renderer = True, 
                                              base_level=base_level, 
                                              model_scale = model_scale)
        res = evaluateDistanceNetwork(estimator, data_loader, 
                                      inverse_distance = inverse_distance,
                                      use_converter = True)
        del estimator
        torch.cuda.empty_cache()

        model_results_dir = os.path.join(results_dir, '02_{:02}'.format(obj))
        os.makedirs(model_results_dir, exist_ok=True)
        np.savez(os.path.join(model_results_dir, 'distance.npz'),
                gt_ranks = res[0],
                top_ranks = res[1],
                gt_outputs = res[2],
                top_dists = res[3],
                data = res[4])
       
        image_path = os.path.join(model_results_dir, 'images') 
        os.makedirs(image_path, exist_ok=True)
        images = res[5]

        if(len(images['gt_pos']) > 0):
            data_idx = images['gt_pos'][3]
            cv2.imwrite(image_path + '/gt_pos_top_img_{}.png'.format(data_idx), 
                        images['gt_pos'][0])
            cv2.imwrite(image_path + '/gt_pos_tgt_img_{}.png'.format(data_idx), 
                        images['gt_pos'][1])
            cv2.imwrite(image_path + '/gt_pos_gt_img_{}.png'.format(data_idx), 
                        images['gt_pos'][2])
            with open(image_path + '/gt_pos.txt', 'w') as f:
                f.write('gt_rank:   {}\n'.format(res[0][data_idx]))
                f.write('top_rank:  {}\n'.format(res[1][data_idx]))
                f.write('gt_output: {}\n'.format(res[2][data_idx]))
                f.write('top_dist:  {}\n'.format(res[3][data_idx]))

        if(len(images['gt_neg']) > 0):
            data_idx = images['gt_neg'][3]
            cv2.imwrite(image_path + '/gt_neg_top_img_{}.png'.format(data_idx), 
                        images['gt_neg'][0])
            cv2.imwrite(image_path + '/gt_neg_tgt_img_{}.png'.format(data_idx), 
                        images['gt_neg'][1])
            cv2.imwrite(image_path + '/gt_neg_gt_img_{}.png'.format(data_idx), 
                        images['gt_neg'][2])
            with open(image_path + '/gt_neg.txt', 'w') as f:
                f.write('gt_rank:   {}\n'.format(res[0][data_idx]))
                f.write('top_rank:  {}\n'.format(res[1][data_idx]))
                f.write('gt_output: {}\n'.format(res[2][data_idx]))
                f.write('top_dist:  {}\n'.format(res[3][data_idx]))

        if(len(images['top_pos']) > 0):
            data_idx = images['top_pos'][3]
            cv2.imwrite(image_path + '/top_pos_top_img_{}.png'.format(data_idx), 
                        images['top_pos'][0])
            cv2.imwrite(image_path + '/top_pos_tgt_img_{}.png'.format(data_idx), 
                        images['top_pos'][1])
            cv2.imwrite(image_path + '/top_pos_gt_img_{}.png'.format(data_idx), 
                        images['top_pos'][2])
            with open(image_path + '/top_pos.txt', 'w') as f:
                f.write('gt_rank:   {}\n'.format(res[0][data_idx]))
                f.write('top_rank:  {}\n'.format(res[1][data_idx]))
                f.write('gt_output: {}\n'.format(res[2][data_idx]))
                f.write('top_dist:  {}\n'.format(res[3][data_idx]))

        if(len(images['top_neg']) > 0):
            data_idx = images['top_neg'][3]
            cv2.imwrite(image_path + '/top_neg_top_img_{}.png'.format(data_idx), 
                        images['top_neg'][0])
            cv2.imwrite(image_path + '/top_neg_tgt_img_{}.png'.format(data_idx), 
                        images['top_neg'][1])
            cv2.imwrite(image_path + '/top_neg_gt_img_{}.png'.format(data_idx), 
                        images['top_neg'][2])
            with open(image_path + '/top_neg.txt', 'w') as f:
                f.write('gt_rank:   {}\n'.format(res[0][data_idx]))
                f.write('top_rank:  {}\n'.format(res[1][data_idx]))
                f.write('gt_output: {}\n'.format(res[2][data_idx]))
                f.write('top_dist:  {}\n'.format(res[3][data_idx]))

        print('Mean GT Rank: {}'.format(res[0].mean()))
        print('Mean Top Rank: {}'.format(res[1].mean()))
        print('Mean Top Dist: {}'.format(res[3].mean()))

if __name__=='__main__':
    #args = getArgs()
    main(weight_file = '/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/2018-05-17_17-42-17/weights/checkpoint_262000.pth',
         data_folder = '/scratch/bokorn/data/benchmarks/linemod6DC/',
         results_dir = '/home/bokorn/results/linemod6dc/bce')
