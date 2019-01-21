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

import os
import time
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
from generic_pose.utils import to_var, to_np
from generic_pose.losses.distance_utils import getDistanceLoss, getIndices
from generic_pose.utils.pose_processing import quatAngularDiffDot, quatAngularDiffBatch

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

        
def main():
    augmentation_prob = 0.0
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    train_dataset = YCBDataset(data_dir=benchmark_folder,
                                        image_set='train_split',
                                        img_size=(224,224),
                                        obj=1,
                                        use_syn_data=True,
                                        augmentation_prob = augmentation_prob)

    train_dataset.loop_truth = None
    batch_size = 16 
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=batch_size, shuffle=True)
    base_render_folder = os.path.join(benchmark_folder,
                                      'base_renders',
                                      train_dataset.getObjectName(),
                                      '{}'.format(2))
    base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
    base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
  
  
    model = gen_pose_net('alexnet','sigmoid', output_dim = 1, pretrained = True, siamese_features = True)
    model.train()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-5)
   
    t = time.time()
    load_times = []
    to_np_times = []
    dist_times = []
    to_var_times = []
    forward_times = []
    backward_times = []
    distanceLoss, distanceError, dist_sign = getDistanceLoss('exp', np.pi/20)
    for batch_idx, (query_imgs, query_quats, _1, _2) in enumerate(train_loader):
        log_data = not((batch_idx+1) % 10)
        load_times.append(time.time()-t)
        t = time.time()
        
        query_quats = to_np(query_quats)
        to_np_times.append(time.time()-t)
        t = time.time()
        
        dist = quatAngularDiffBatch(query_quats, base_vertices[:batch_size])
        dist_times.append(time.time()-t)
        t = time.time()
                
        grid_img_samples = to_var(base_renders[:batch_size])
        query_img_samples = to_var(query_imgs)
        dist_true = to_var(torch.tensor(dist)).detach()
        to_var_times.append(time.time()-t)
        t = time.time()
        
        grid_features = model.originFeatures(grid_img_samples)
        query_features = model.queryFeatures(query_img_samples)
        dist_est = model.compare_network(grid_features, query_features)
        loss = distanceLoss(dist_est.flatten(), dist_true, reduction='mean')

        forward_times.append(time.time()-t)
        t = time.time()

        model.train()
        loss.backward()
        optimizer.step()

        backward_times.append(time.time()-t)
        t = time.time()

        if(log_data):
            print("Load Times:", load_times)
            print("To NP Times:", to_np_times)
            print("Distance Times:", dist_times)
            print("To Var Times:", to_var_times)
            print("Forward Times:", forward_times)
            print("Backward Times:", backward_times)
            print("Mean Load Time:", np.mean(load_times[1:-1]))
            print("Mean To NP Times:", np.mean(to_np_times[1:-1]))
            print("Mean Distance Times:", np.mean(dist_times[1:-1]))
            print("Mean To Var Times:", np.mean(to_var_times[1:-1]))
            print("Mean Forward Time:", np.mean(forward_times[1:-1]))
            print("Mean Backward Time:", np.mean(backward_times[1:-1]))
            break
    print(np.mean(load_times)/100)


if __name__=='__main__':
    main()

