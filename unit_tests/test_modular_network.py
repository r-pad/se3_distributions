# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:50:47 2018

@author: bokorn
"""
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from generic_pose.datasets.image_dataset import PoseImageDataSet

from generic_pose.models.pose_networks import gen_pose_net
from generic_pose.models.feature_networks import feature_networks
from generic_pose.models.compare_networks import compare_networks

#data_file = '/home/bokorn/src/generic_pose/generic_pose/training_sets/robot_plane_valid.txt'
#with open(data_file, 'r') as f:    
#    data_folders = f.read().split()
data_folders = ['/scratch/bokorn/data/renders/drill_1_renders/valid/']
model_data_file = '/scratch/bokorn/data/models/sawyer_gripper/electric_gripper_w_fingers.obj'

loader = DataLoader(PoseImageDataSet(data_folders=data_folders,
                                     img_size = (224, 224),
                                     model_filenames=model_data_file,
                                     background_filenames = None,
                                     classification = False,
                                     num_bins= (1,1,1),
                                     distance_sigma=1),
                    num_workers=4, 
                    batch_size=1, 
                    shuffle=True)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

for feature_type in sorted(['alexnet_fc1', 'alexnet_fc2']): #feature_networks.keys()):
#    feature_model, feature_size = feature_networks[feature_type]()
#    print('{}: {}'.format(feature_type, get_n_params(feature_model)))
#    continue    
    print('=================='.format(feature_type))
    print('== {} =='.format(feature_type))
    print('=================='.format(feature_type))
    for compare_type in sorted(compare_networks.keys()):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        k, (origin, query, quat_true, class_true, origin_quat, model_file) = next(enumerate(loader))
        origin = Variable(origin)
        query = Variable(query)

        torch.manual_seed(int(time.time()*1000))
        torch.cuda.manual_seed_all(int(time.time()*1000))
        
        model = gen_pose_net(feature_type, compare_type, output_dim = 4, pretrained = True)
        model.eval()
        v_pret1 = model.features(origin)
        model = gen_pose_net(feature_type, compare_type, output_dim = 4, pretrained = False)
        model.eval()
        v_rand1 = model.features(origin)
        model = gen_pose_net(feature_type, compare_type, output_dim = 4, pretrained = True)
        model.eval()
        v_pret2 = model.features(origin)
        model = gen_pose_net(feature_type, compare_type, output_dim = 4, pretrained = False)
        model.eval()
        v_rand2 = model.features(origin)
        print('{} {}:'.format(feature_type, compare_type))
        print('Pretrained: {}, {}'.format((v_pret1.data.numpy()).sum(), (v_pret2.data.numpy()).sum()))
        print('Random:     {}, {}'.format((v_rand1.data.numpy()).sum(), (v_rand2.data.numpy()).sum()))
        if(np.abs(v_pret1.data.numpy()).sum() < 1e-9):
            print('ERROR: Pretrain is close to zero')
        if(np.any((v_pret1 != v_pret2).data.numpy())):
            print('ERROR: Pretrain does not match')
        if(np.abs((v_rand1 - v_rand2).data.numpy()).sum() < 1e-9):
            print('ERROR: Random does match')
        if(np.abs((v_pret1 - v_rand1).data.numpy()).sum() < 1e-9):
            print('ERROR: Pretrain matchs Random')
        