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
    train_dataset.append_rendered = True
        
    train_loader = DataLoader(train_dataset, num_workers=6, batch_size=16, shuffle=True)
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
   
    per_instance = False 

    t = time.time()
    load_times = []
    train_times = []
    for batch_idx, (query_imgs, query_quats, _1, _2) in enumerate(train_loader):
        #print(load_times[-1])
        del _1, _2
        log_data = not((batch_idx+1) % 100)
        #pre_info, pre_objs  = getTensors()     
        torch.cuda.empty_cache()
        #import IPython; IPython.embed()
        load_times.append(time.time()-t)
        t = time.time()
        train_results = evaluateRenderedDistance(model, None, None,
                                                 query_imgs, query_quats,
                                                 base_renders, base_vertices,
                                                 loss_type = 'exp',
                                                 falloff_angle = 20*np.pi/180,
                                                 optimizer = optimizer, 
                                                 disp_metrics = log_data,
                                                 num_indices = 8,
                                                 per_instance = per_instance,
                                                 sample_by_loss = False,
                                                 top_n = 0,
                                                 uniform_prop = 1,
                                                 loss_temperature = None,
                                                 sampling_distribution = None)

        torch.cuda.empty_cache()
        train_times.append(time.time()-t)
        t = time.time()

        if(log_data):
            print("Load Times:", load_times)
            print("Train Times:", train_times)
            print("Mean Load Time:", np.mean(load_times[1:-1]))
            print("Mean Train Time:", np.mean(train_times[1:-1]))
            break
    print(np.mean(load_times)/100)


if __name__=='__main__':
    main()

