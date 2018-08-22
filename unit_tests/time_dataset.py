# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

import time

def timeDataLoader(data_loader,
                   num_batches = 10,
                   plot_prefix = '/home/bokorn/results/test/'):
    load_times = []
    t = time.time()
    for j in range(num_batches):
    #for j, _ in enumerate(data_loader):
        t = time.time()
        _ = next(iter(data_loader))
        load_times.append(time.time()-t)
        #t = time.time()
        print(load_times[-1])
        #if (j >= num_batches - 1):
        #    break
    load_times = np.array(load_times)
    print('Mean Load Time: {}'.format(np.mean(load_times)))
    print('Min Load Time:  {}'.format(np.min(load_times)))
    print('Max Load Time:  {}'.format(np.max(load_times)))

    plt.plot(load_times)

    plt.xlabel('Batch Index')
    plt.ylabel('Load Time (s)')
    plt.savefig(plot_prefix + 'load_times.png')
    plt.gcf().clear()

    return load_times

def main():
    import os
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--results_prefix', type=str, default='/home/bokorn/results/test/')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset_type', type=str, default='numpy')
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--max_orientation_angle', type=float, default=None)
    parser.add_argument('--max_orientation_iters', type=int, default=200)  
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_batches', type=int, default=100)
 
    args = parser.parse_args()

    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None

    if(args.dataset_type.lower() == 'numpy'):
        from generic_pose.datasets.numpy_dataset import NumpyImageDataset as Dataset
    elif(args.dataset_type.lower() == 'linemod'):
        from generic_pose.datasets.benchmark_dataset import LinemodDataset as Dataset
    elif(args.dataset_type.lower() == 'linemod_masked'):
        from functools import partial
        from generic_pose.datasets.benchmark_dataset import LinemodDataset
        Dataset = partial(LinemodDataset, use_mask = True)
    #elif(args.dataset_type.lower() = 'png'):
    #    import torch.
    else:
        raise ValueError('Dataset type {} not implemented'.format(args.dataset_type))

    t = time.time()
    data_loader = DataLoader(Dataset(data_folders=args.data_folder,
                                     img_size = (224, 224),
                                     max_orientation_offset = args.max_orientation_angle,
                                     max_orientation_iters = args.max_orientation_iters,
                                     model_filenames=None,
                                     background_filenames = background_filenames),
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True)

    data_loader.dataset.loop_truth = [1,0]
    print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    t = time.time()
    timeDataLoader(data_loader, num_batches = args.num_batches, 
                   plot_prefix = args.results_prefix)
    import IPython; IPython.embed()

if __name__=='__main__':
    main()
