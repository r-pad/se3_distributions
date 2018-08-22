# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
import cv2

from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.eval.hyper_distance import ExemplarDistPoseEstimator, vert600
from quat_math import (quatAngularDiff,
                       quat2AxisAngle, 
                       quaternion_about_axis, 
                       quaternion_multiply, 
                       quaternion_inverse,
                       quaternion_from_matrix)

from generic_pose.utils import to_np

delta_quat = np.array([.5,.5,.5,.5])

def render_check(estimator, data_loader, trans_mat = np.eye(4), use_converter = False):

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

    dist_0 = float('inf')
    img_0  = None
    quat_0 = None
    dist_x = float('inf')
    img_x  = None
    quat_x = None
    dist_y = float('inf')
    img_y  = None
    quat_y = None
    dist_z = float('inf')
    img_z  = None
    quat_z = None

    for imgs, _, quats, _, _ in data_loader:
        d0 = quatAngularDiff(quats[0][0], quaternion_about_axis(0,[1,0,0]))
        dx = quatAngularDiff(quats[0][0], quaternion_about_axis(np.pi/2,[1,0,0]))
        dy = quatAngularDiff(quats[0][0], quaternion_about_axis(np.pi/2,[0,1,0]))
        dz = quatAngularDiff(quats[0][0], quaternion_about_axis(np.pi/2,[0,0,1]))
        
        if(d0 < dist_0):
            dist_0 = d0
            img_0  = imgs[0]
            quat_0 = to_np(quats[0][0]).copy()
        if(dx < dist_x):
            dist_x = dx
            img_x  = imgs[0]
            quat_x = to_np(quats[0][0]).copy()
        if(dy < dist_y):
            dist_y = dy
            img_y  = imgs[0]
            quat_y = to_np(quats[0][0]).copy()
        if(dz < dist_z):
            dist_z = dz
            img_z  = imgs[0]
            quat_z = to_np(quats[0][0]).copy()

    cv2.imwrite('/home/bokorn/results/test/0.png', unprocessImages(img_0)[0])
    cv2.imwrite('/home/bokorn/results/test/x.png', unprocessImages(img_x)[0])
    cv2.imwrite('/home/bokorn/results/test/y.png', unprocessImages(img_y)[0])
    cv2.imwrite('/home/bokorn/results/test/z.png', unprocessImages(img_z)[0])

    cv2.imwrite('/home/bokorn/results/test/r0.png', estimator.renderPoses([convertQuat(quat_0)])[0])
    cv2.imwrite('/home/bokorn/results/test/rx.png', estimator.renderPoses([convertQuat(quat_x)])[0])
    cv2.imwrite('/home/bokorn/results/test/ry.png', estimator.renderPoses([convertQuat(quat_y)])[0])
    cv2.imwrite('/home/bokorn/results/test/rz.png', estimator.renderPoses([convertQuat(quat_z)])[0])

#    cv2.imwrite('/home/bokorn/results/test/b0.png', estimator.renderPoses([quat_0])[0])
#    cv2.imwrite('/home/bokorn/results/test/bx.png', estimator.renderPoses([quat_x])[0])
#    cv2.imwrite('/home/bokorn/results/test/by.png', estimator.renderPoses([quat_y])[0])
#    cv2.imwrite('/home/bokorn/results/test/bz.png', estimator.renderPoses([quat_z])[0])
    import IPython; IPython.embed()
    
def main():
    import time
    import cv2
    import torch
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser
    from generic_pose.models.pose_networks import gen_pose_net
    
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
 
    args = parser.parse_args()

    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None

    t = time.time()
    dist_net = gen_pose_net(args.model_type.lower(), 
                            args.compare_type.lower(), 
                            output_dim = 1,
                            pretrained = False)

    dist_net.load_state_dict(torch.load(args.weight_file))
    print('Weights load time: {}s'.format(round(time.time()-t, 2)))

    if(args.dataset_type.lower() == 'numpy'):
        from generic_pose.datasets.numpy_dataset import NumpyImageDataset as Dataset
    elif(args.dataset_type.lower() == 'linemod'):
        from generic_pose.datasets.benchmark_dataset import LinemodDataset as Dataset
    elif(args.dataset_type.lower() == 'linemod_masked'):
        from functools import partial
        from generic_pose.datasets.benchmark_dataset import LinemodDataset
        Dataset = partial(LinemodDataset, use_mask = True)
    else:
        raise ValueError('Dataset type {} not implemented'.format(args.data_type))
    
    t = time.time()
    data_loader = DataLoader(Dataset(data_folders=args.data_folder,
                                     img_size = (224, 224),
                                     max_orientation_offset = None,
                                     max_orientation_iters = None,
                                     model_filenames=None,
                                     background_filenames = background_filenames),
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size, 
                             shuffle=True)

    data_loader.dataset.loop_truth = [1]
    print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    if(args.model_filename is None and 'linemod' in args.dataset_type.lower()):
        _, _, _, _, model_filenames = next(iter(data_loader))
        args.model_filename = model_filenames[0][0] 
    else:
        model_class, model_name = args.data_folder.split('/')[-3:-1]
        if(model_class == 'linemod'):
            args.model_filename = '/scratch/bokorn/data/benchmarks/linemod/' + \
                                  '{}/mesh.ply'.format(model_name)
        else:
            args.model_filename = '/scratch/bokorn/data/models/shapenetcore/' + \
                                  '{}/{}/model.obj'.format(model_class, model_name)

    if('cam/mesh.ply' in args.model_filename or 
       'eggbox/mesh.ply' in args.model_filename):
        trans_filename = args.model_filename.replace('mesh.ply', 'transform.dat')
        with open(trans_filename, 'r') as f:
            trans_data = f.read()
        trans = np.array([float(line.split()[1]) for line in trans_data.splitlines()[1:]])
        trans_mat = np.vstack([trans.reshape((3,4)), np.array([0,0,0,1])])
    else:
        trans_mat = np.eye(4)

    estimator = ExemplarDistPoseEstimator(args.model_filename, dist_net)
    res = render_check(estimator, data_loader, trans_mat, 'linemod' in args.dataset_type.lower())
    np.savez(args.results_prefix + 'distance.npz',
            gt_ranks = res[0],
            top_ranks = res[1],
            top_dists = res[2])
    print('Mean GT Rank: {}'.format(res[0].mean()))
    print('Mean Top Rank: {}'.format(res[1].mean()))
    print('Mean Top Dist: {}'.format(res[2].mean()))

if __name__=='__main__':
    main() 
