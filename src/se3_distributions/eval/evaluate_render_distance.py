# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
import cv2

from se3_distributions.utils.image_preprocessing import unprocessImages
from se3_distributions.eval.hyper_distance import ExemplarDistPoseEstimator 
from quat_math import (quatAngularDiff,
                       quat2AxisAngle, 
                       quaternion_about_axis, 
                       quaternion_multiply, 
                       quaternion_inverse,
                       quaternion_from_matrix)

from se3_distributions.utils import to_np

delta_quat = np.array([.5,.5,.5,.5])
#flip_x = np.array([-1,1,1,1])
#def convertQuat(q):
#    q_flip = quaternion_inverse(q.copy())
#    q_flip[2] *= -1
#    return quaternion_multiply(delta_quat, q_flip)

def evaluateDistanceNetwork(estimator, data_loader, trans_mat=np.eye(4), 
                            inverse_distance = True, 
                            use_converter = False):
    trans_quat = quaternion_inverse(quaternion_from_matrix(trans_mat))
    if(use_converter): 
        def convertQuat(q):
            q_flip = quaternion_inverse(q.copy())
            q_flip = quaternion_multiply(trans_quat, q_flip)
            q_flip[2] *= -1
            return quaternion_multiply(delta_quat, q_flip)
    else:
        def convertQuat(q):
            return q

    def trueDiff(q):
        q_adj = convertQuat(q)
        true_diff = []
        for v in estimator.base_vertices:
            true_diff.append(quatAngularDiff(q_adj, v))
        return np.array(true_diff)

  
    if(inverse_distance):
        sign = -1.0
    else:
        sign = 1
    gt_ranks = []
    gt_outputs = []
    top_ranks = []
    top_dists = []
    angle_vec = []
    output_vec = []
    gt_pos_images = []
    gt_neg_images = []
    top_pos_images = []
    top_neg_images = []

    for j, (imgs, quats, _, _) in enumerate(data_loader):
        diff = estimator.estimate(imgs, preprocess=False) 
        diff = to_np(diff.detach())
        num_verts = diff.shape[0]
        true_dist = trueDiff(to_np(quats[0]))
        top_idx = np.argmin(sign*diff)
        true_idx = np.argmin(true_dist)

        angle_vec.extend(true_dist.tolist())
        output_vec.extend(diff.tolist())

        rank_gt = np.nonzero(np.argsort(sign*diff) == true_idx)[0][0]
        rank_top = np.nonzero(np.argsort(true_dist) == top_idx)[0][0]
        output_gt = diff[true_idx]
        dist_top = true_dist[top_idx]*180/np.pi
        gt_ranks.append(rank_gt)
        gt_outputs.append(output_gt)
        top_ranks.append(rank_top)
        top_dists.append(dist_top)
        #print('True ranking: {}'.format(np.nonzero(np.argsort(-diff) == true_idx)[0][0]))
        #print('Top scored ranking: {}'.format(np.nonzero(np.argsort(true_dist) == top_idx)[0][0]))   
        #print('Top distance: {}'.format(true_dist[top_idx]*180/np.pi))
        if(False):
            if(rank_gt < num_verts*0.05):
                top_img = unprocessImages(estimator.base_renders[top_idx:top_idx+1])[0]
                tgt_img = unprocessImages(imgs[0])[0]
                gt_img  = unprocessImages(estimator.base_renders[true_idx:true_idx+1])[0]
                gt_pos_images = [top_img, tgt_img, gt_img, j]
            elif(rank_gt > num_verts*0.95):
                top_img = unprocessImages(estimator.base_renders[top_idx:top_idx+1])[0]
                tgt_img = unprocessImages(imgs[0])[0]
                gt_img  = unprocessImages(estimator.base_renders[true_idx:true_idx+1])[0]
                gt_neg_images = [top_img, tgt_img, gt_img, j]
            elif(rank_top < num_verts*0.05):
                top_img = unprocessImages(estimator.base_renders[top_idx:top_idx+1])[0]
                tgt_img = unprocessImages(imgs[0])[0]
                gt_img  = unprocessImages(estimator.base_renders[true_idx:true_idx+1])[0]
                top_pos_images = [top_img, tgt_img, gt_img, j]
            elif(rank_top > num_verts*0.95):
                top_img = unprocessImages(estimator.base_renders[top_idx:top_idx+1])[0]
                tgt_img = unprocessImages(imgs[0])[0]
                gt_img  = unprocessImages(estimator.base_renders[true_idx:true_idx+1])[0]
                top_neg_images = [top_img, tgt_img, gt_img, j]

        #input("Press Enter to continue...")
        #import IPython; IPython.embed()
    images = {'gt_pos':gt_pos_images,
              'gt_neg':gt_neg_images,
              'top_pos':top_pos_images,
              'top_neg':top_neg_images}

    data = np.array([angle_vec, output_vec])
    return np.array(gt_ranks), np.array(top_ranks),\
           np.array(gt_outputs), np.array(top_dists),\
           data, images

def getArgs():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--results_prefix', type=str)
    parser.add_argument('--weight_file', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset_type', type=str, default='numpy')
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--model_filename', type=str, default=None)

    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='exp')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--inverse_distance', dest='inverse_distance', action='store_true')

    parser.add_argument('--base_level', type=int, default=0)

    args = parser.parse_args()
    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None

    if(args.model_filename is None):
        if(args.data_folder[-1] == '/'):
            args.data_folder = args.data_folder[:-1]
        model_class, model_name = args.data_folder.split('/')[-2:]
        if(model_class == 'linemod'):
            args.model_filename = '/scratch/bokorn/data/benchmarks/linemod/' + \
                                  '{}/mesh.ply'.format(model_name)
        else:
            args.model_filename = '/scratch/bokorn/data/models/shapenetcore/' + \
                                  '{}/{}/model.obj'.format(model_class, model_name)

    return args

def main(weight_file,
         model_filename,
         data_folder,
         dataset_type,
         results_prefix,
         model_type = 'alexnet',
         compare_type = 'sigmoid',
         inverse_distance = True,
         background_filenames = None,
         base_level = 0,
         ):
    import os
    import time
    import cv2
    import torch
    from torch.utils.data import DataLoader
    from se3_distributions.models.pose_networks import gen_pose_net, load_state_dict
    
    print(results_prefix)
    t = time.time()
    dist_net = gen_pose_net(model_type.lower(), 
                            compare_type.lower(), 
                            output_dim = 1,
                            pretrained = False)

    load_state_dict(dist_net, weight_file)
    #print('Weights load time: {}s'.format(round(time.time()-t, 2)))

    if(dataset_type.lower() == 'numpy'):
        from se3_distributions.datasets.numpy_dataset import NumpyImageDataset as Dataset
    elif(dataset_type.lower() == 'linemod'):
        from se3_distributions.datasets.benchmark_dataset import LinemodDataset as Dataset
    elif(dataset_type.lower() == 'linemod_masked'):
        from functools import partial
        from se3_distributions.datasets.benchmark_dataset import LinemodDataset
        Dataset = partial(LinemodDataset, use_mask = True)
    else:
        raise ValueError('Dataset type {} not implemented'.format(data_type))
    
    t = time.time()
    data_loader = DataLoader(Dataset(data_folders=data_folder,
                                     img_size = (224, 224),
                                     max_orientation_offset = None,
                                     max_orientation_iters = None,
                                     model_filenames=None,
                                     background_filenames = background_filenames),
                             num_workers=0, 
                             batch_size=1, 
                             shuffle=True)

    data_loader.dataset.loop_truth =None # [1]
    #print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    if('cam/mesh.ply' in model_filename or 
       'eggbox/mesh.ply' in model_filename):
        trans_filename = model_filename.replace('mesh.ply', 'transform.dat')
        with open(trans_filename, 'r') as f:
            trans_data = f.read()
        trans = np.array([float(line.split()[1]) for line in trans_data.splitlines()[1:]])
        trans_mat = np.vstack([trans.reshape((3,4)), np.array([0,0,0,1])])
    else:
        trans_mat = np.eye(4)

    estimator = ExemplarDistPoseEstimator(model_filename, dist_net, use_bpy_renderer = True, base_level=base_level)
    res = evaluateDistanceNetwork(estimator, data_loader, trans_mat, 
                                  inverse_distance = inverse_distance,
                                  use_converter = 'linemod' in dataset_type.lower())
    results_dir = os.path.dirname(results_prefix)
    os.makedirs(results_dir, exist_ok=True)
    np.savez(results_prefix + 'distance.npz',
            gt_ranks = res[0],
            top_ranks = res[1],
            gt_outputs = res[2],
            top_dists = res[3],
            data = res[4])
   
    image_path = os.path.join(results_dir, 'images') 
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
            f.write('top_dist:  {}\n'.format(res[3][data_idx]*180/np.pi))

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
            f.write('top_dist:  {}\n'.format(res[3][data_idx]*180/np.pi))

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
            f.write('top_dist:  {}\n'.format(res[3][data_idx]*180/np.pi))

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
            f.write('top_dist:  {}\n'.format(res[3][data_idx]*180/np.pi))

    print('Mean GT Rank: {}'.format(res[0].mean()))
    print('Mean Top Rank: {}'.format(res[1].mean()))
    print('Mean Top Dist: {}'.format(res[2].mean()))

if __name__=='__main__':
    args = getArgs()
    main(weight_file = args.weight_file,
         model_filename = args.model_filename,
         data_folder = args.data_folder,
         dataset_type = args.dataset_type,
         results_prefix = args.results_prefix,
         model_type = args.model_type,
         compare_type = args.compare_type,
         background_filenames = args.background_filenames,
         base_level = args.base_level)
