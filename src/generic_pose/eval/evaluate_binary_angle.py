# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:15:38 2018

@author: bokorn
"""

import numpy as np

import torch
from torch.utils.data import DataLoader

from generic_pose.training.binary_angle_utils import evaluateBinaryEstimate
from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.training.utils import to_np
from quat_math import quat2AxisAngle

from tqdm import tqdm, trange
import time

def evaluate(model, data_loader, 
             target_angle = np.pi/4.0,
             prediction_thresh = 0.5,
             num_samples = None, 
             init_offset = np.pi/36.0,
             histogram_bins = 100,
             loss_type = 'BCE'):
    model.cuda()
    model.eval()
    if num_samples is None:
        num_samples = len(data_loader)

    angle_inc = (np.pi - init_offset) / num_samples

    count_hist = np.zeros(histogram_bins)
    output_hist = np.zeros(histogram_bins)
    output2_hist = np.zeros(histogram_bins)
    accuracy_hist = np.zeros(histogram_bins)
    pbar = tqdm(trange(num_samples))
#    pbar.set_description('Load time for {}: {:.2}s'.format(0.0, 0.0))
    data = [[],[],[]]

    true_pos_imgs = []
    true_neg_imgs = []
    false_pos_imgs = []
    false_neg_imgs = []

    for batch_idx in pbar:
        max_angle = init_offset + batch_idx * angle_inc
        data_loader.dataset.max_orientation_offset = max_angle
        t = time.time() 
        images, trans, _, _, _ = next(iter(data_loader))
        load_time = time.time() - t
        results = evaluateBinaryEstimate(model, images[0], images[1], trans[0], 
                                         target_angle = target_angle,
                                         prediction_thresh = prediction_thresh,
                                         loss_type=loss_type,
                                         optimizer = None, disp_metrics = True)
        angle_vec = results['angle_vec'] 
        counts, edges = np.histogram(angle_vec, bins = histogram_bins, range=(0.0,180.0))
        count_hist += counts
        output_vec = results['output_vec'].flatten()
        outputs, _ = np.histogram(angle_vec, weights=output_vec, 
                                  bins = histogram_bins, range=(0.0,180.0))
        output_hist += outputs
        correct_vec = to_np(results['correct_vec']).flatten()

        accuracys, _ = np.histogram(angle_vec, weights=correct_vec, 
                                    bins = histogram_bins, range=(0.0,180.0))
        accuracy_hist += accuracys
        data[0].extend(angle_vec.tolist())
        data[1].extend(output_vec.tolist())
        data[2].extend(correct_vec.tolist())
        pbar.set_description('Angle Diff {}'.format(round(quat2AxisAngle(trans[0][0])[1]*180/np.pi, 1)))
#        pbar.set_description('Load time for {}: {:.2}s'.format(round(max_angle*180/np.pi, 1),
#                                                               load_time))

        succ_idx = np.flatnonzero(correct_vec)
        fail_idx = np.flatnonzero(correct_vec==0)
        #tps = np.nonzero(np.bitwise_and(correct_vec, angle_vec < target_angle))
        #tns = np.nonzero(np.bitwise_and(correct_vec, angle_vec > target_angle))
        #fps = np.nonzero(np.bitwise_and(np.bitwise_not(correct_vec), angle_vec > target_angle))
        #fns = np.nonzero(np.bitwise_and(np.bitwise_not(correct_vec), angle_vec < target_angle))
        if(len(succ_idx) > 0):
            idx = succ_idx[np.argmin(np.abs(angle_vec[succ_idx]-max_angle*180/np.pi))]
            if(angle_vec[idx] < target_angle*180.0/np.pi):
                true_pos_imgs.append([angle_vec[idx], 
                                      unprocessImages(images[0][idx:idx+1])[0],
                                      unprocessImages(images[1][idx:idx+1])[0],])
            else:
                true_neg_imgs.append([angle_vec[idx], 
                                      unprocessImages(images[0][idx:idx+1])[0],
                                      unprocessImages(images[1][idx:idx+1])[0],])
        if(len(fail_idx) > 0):
            idx = fail_idx[np.argmin(np.abs(angle_vec[fail_idx]-max_angle*180/np.pi))]
            if(angle_vec[idx] < target_angle*180.0/np.pi):
                false_neg_imgs.append([angle_vec[idx], 
                                       unprocessImages(images[0][idx:idx+1])[0],
                                       unprocessImages(images[1][idx:idx+1])[0],])
            else:
                false_pos_imgs.append([angle_vec[idx], 
                                       unprocessImages(images[0][idx:idx+1])[0],
                                       unprocessImages(images[1][idx:idx+1])[0],])

    output_hist /= count_hist
    accuracy_hist /= count_hist
    result_imgs = {'true_pos':true_pos_imgs,
                   'true_neg':true_neg_imgs,
                   'false_pos':false_pos_imgs,
                   'false_neg':false_neg_imgs}

    return count_hist, output_hist, accuracy_hist, edges, data, result_imgs

def main():
    import os
    import cv2
    from argparse import ArgumentParser
    from generic_pose.models.pose_networks import gen_pose_net
    
    parser = ArgumentParser()
    parser.add_argument('--results_prefix', type=str)
    parser.add_argument('--weight_file', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset_type', type=str, default='numpy')
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--target_angle', type=float, default=np.pi/4)
    parser.add_argument('--max_orientation_iters', type=int, default=200)    
    
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
    model = gen_pose_net('alexnet', 'sigmoid', output_dim = 1, pretrained = True)
    model.load_state_dict(torch.load(args.weight_file))
    print('Model load time: {}s'.format(round(time.time()-t, 2)))

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
                                     max_orientation_iters = args.max_orientation_iters,
                                     model_filenames=None,
                                     background_filenames = background_filenames),
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size, 
                             shuffle=True)
    data_loader.dataset.loop_truth = [1,0]
    print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    t = time.time()
    histograms = evaluate(model, data_loader, args.target_angle, num_samples=args.num_samples)
    np.savez(args.results_prefix+'histograms.npz',
             count_hist = histograms[0], 
             output_hist = histograms[1], 
             accuracy_hist = histograms[2],
             hist_edges = histograms[3],
             data = histograms[4])
    img_dict = histograms[5]
    os.makedirs(args.results_prefix + 'imgs', exist_ok=True)
    os.makedirs(args.results_prefix + 'imgs/true_pos', exist_ok=True)
    for j, (angle, img0, img1) in enumerate(img_dict['true_pos']):
        cv2.imwrite(args.results_prefix + 'imgs/true_pos/img_{}_0_{:.1f}.png'.format(j, angle), img0)
        cv2.imwrite(args.results_prefix + 'imgs/true_pos/img_{}_1_{:.1f}.png'.format(j, angle), img1)
    os.makedirs(args.results_prefix + 'imgs/true_neg', exist_ok=True)
    for j, (angle, img0, img1) in enumerate(img_dict['true_neg']):
        cv2.imwrite(args.results_prefix + 'imgs/true_neg/img_{}_0_{:.1f}.png'.format(j, angle), img0)
        cv2.imwrite(args.results_prefix + 'imgs/true_neg/img_{}_1_{:.1f}.png'.format(j, angle), img1)

    os.makedirs(args.results_prefix + 'imgs/false_pos', exist_ok=True)
    for j, (angle, img0, img1) in enumerate(img_dict['false_pos']):
        cv2.imwrite(args.results_prefix + 'imgs/false_pos/img_{}_0_{:.1f}.png'.format(j, angle), img0)
        cv2.imwrite(args.results_prefix + 'imgs/false_pos/img_{}_1_{:.1f}.png'.format(j, angle), img1)
    os.makedirs(args.results_prefix + 'imgs/false_neg', exist_ok=True)
    for j, (angle, img0, img1) in enumerate(img_dict['false_neg']):
        cv2.imwrite(args.results_prefix + 'imgs/false_neg/img_{}_0_{:.1f}.png'.format(j, angle), img0)
        cv2.imwrite(args.results_prefix + 'imgs/false_neg/img_{}_1_{:.1f}.png'.format(j, angle), img1)

    #import IPython; IPython.embed()
    
if __name__=='__main__':
    main()
