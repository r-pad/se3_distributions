# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from logger import Logger

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np

from torch.optim import Adam, Adadelta, SGD
from torch.utils.data import DataLoader

import os
import time
import gc
from itertools import cycle

from generic_pose.datasets.numpy_dataset import NumpyImageDataset
from generic_pose.utils import to_np
from generic_pose.training.pose_distance_utils import evaluateDistance

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

import psutil

class PoseDistanceTrainer(object):
    def __init__(self, 
                 train_data_folder,
                 valid_class_folder = None,
                 valid_model_folder = None,
                 valid_pose_folder = None,
                 img_size = (227,227),
                 falloff_angle = np.pi/4,
                 rejection_thresh_angle = 25*np.pi/180,
                 max_orientation_angle = None,
                 max_orientation_iters = 200,
                 batch_size = 32,
                 num_workers = 4,
                 model_filenames = None,
                 background_filenames = None,
                 cross_model_eval = True,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.falloff_angle = falloff_angle
        self.cross_model_eval = cross_model_eval

        self.train_loader = DataLoader(NumpyImageDataset(data_folders=train_data_folder,
                                               img_size = img_size,
                                               rejection_thresh_angle = rejection_thresh_angle,
                                               max_orientation_offset = max_orientation_angle,
                                               max_orientation_iters = max_orientation_iters,
                                               model_filenames=model_filenames,
                                               background_filenames = background_filenames,
                                               classification=False,
                                               num_bins=(1,1,1),
                                               distance_sigma=1),
                                       num_workers=num_workers, 
                                       batch_size=int(batch_size/2), 
                                       shuffle=True)
        self.train_loader.dataset.loop_truth = [1,0]
        
        self.valid_types = []
        self.valid_loaders = []
        
        if(valid_class_folder is not None):
            self.valid_class_loader = DataLoader(NumpyImageDataset(data_folders=valid_class_folder,
                                                   img_size = img_size,
                                                   rejection_thresh_angle = rejection_thresh_angle,
                                                   max_orientation_offset = max_orientation_angle,
                                                   max_orientation_iters = max_orientation_iters,
                                                   model_filenames=model_filenames,
                                                   background_filenames = background_filenames,
                                                   classification=False,
                                                   num_bins=(1,1,1),
                                                   distance_sigma=1),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/4), 
                                           shuffle=True)
            self.valid_class_loader.dataset.loop_truth = [1,0]
            self.valid_types.append('valid_class')
            self.valid_loaders.append(cycle(iter(self.valid_class_loader)))

        if(valid_model_folder is not None):
            self.valid_model_loader = DataLoader(NumpyImageDataset(data_folders=valid_model_folder,
                                                   img_size = img_size,
                                                   rejection_thresh_angle = rejection_thresh_angle,
                                                   max_orientation_offset = max_orientation_angle,
                                                   max_orientation_iters = max_orientation_iters,
                                                   model_filenames=model_filenames,
                                                   background_filenames = background_filenames,
                                                   classification=False,
                                                   num_bins=(1,1,1),
                                                   distance_sigma=1),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/4), 
                                           shuffle=True)
            self.valid_model_loader.dataset.loop_truth = [1,0]
            self.valid_types.append('valid_model')
            self.valid_loaders.append(cycle(iter(self.valid_model_loader)))
        
        if(valid_pose_folder is not None):
            self.valid_pose_loader = DataLoader(NumpyImageDataset(data_folders=valid_pose_folder,
                                                   img_size = img_size,
                                                   rejection_thresh_angle = rejection_thresh_angle,
                                                   max_orientation_offset = max_orientation_angle,
                                                   max_orientation_iters = max_orientation_iters,
                                                   model_filenames=model_filenames,
                                                   background_filenames = background_filenames,
                                                   classification=False,
                                                   num_bins=(1,1,1),
                                                   distance_sigma=1),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_pose_loader.dataset.loop_truth = [1,0]
            self.valid_types.append('valid_pose')
            self.valid_loaders.append(cycle(iter(self.valid_pose_loader)))

    def train(self, model, results_dir,
              loss_type = 'exp',
              num_epochs = 100000,
              log_every_nth = 100,
              checkpoint_every_nth = 10000,
              lr = 1e-5,
              optimizer = 'SGD',
              num_display_imgs=1):
        
        model.train()
        model.cuda()
        if(optimizer.lower() == 'sgd'):
            self.optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        elif(optimizer.lower() == 'adam'):
            self.optimizer = Adam(model.parameters(), lr=lr)
        elif(optimizer.lower() == 'adadelta'):
            self.optimizer = Adadelta(model.parameters(), lr=lr)
        else:
            raise AssertionError('Unsupported Optimizer {}, only SGD, Adam, and Adadelta supported'.format(optimizer))
            
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        log_dir = os.path.join(results_dir,'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_log_dir = os.path.join(log_dir,'train')
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        
        train_logger = Logger(train_log_dir)        
        
        valid_logger_list = []
        for valid_name in self.valid_types:
            valid_log_dir = os.path.join(log_dir,valid_name)
            if not os.path.exists(valid_log_dir):
                os.makedirs(valid_log_dir)  
            valid_logger_list.append(Logger(valid_log_dir))
                    
        weights_dir = os.path.join(results_dir,'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            
                
        cumulative_batch_idx = 0
        min_loss = float('inf')
        print('Starting Training')
        log_time = time.time()
        dataset_size = len(self.train_loader)
        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (images, trans, _, _, _) in enumerate(self.train_loader):
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                train_results = evaluateDistance(model, images[0], images[1], trans[0], 
                                                 loss_type = loss_type,
                                                 falloff_angle = self.falloff_angle,
                                                 optimizer = self.optimizer, 
                                                 disp_metrics = log_data)

                if log_data:
                    print("epoch {} ({}):: cumulative_batch_idx {}".format(epoch_idx, time.time() - log_time, cumulative_batch_idx + 1))

                    print('Timestamps: {}'.format(time.time()))

                    log_time = time.time()
                    train_info = {}
                    for k,v in train_results.items():
                        if('vec' not in k):
                            train_info[k] = v
                        else:
                            train_logger.histo_summary(k,v, cumulative_batch_idx+1)

                    
                    for tag, value in train_info.items():
                        train_logger.scalar_summary(tag, value, cumulative_batch_idx+1)

                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        train_logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                        train_logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)
                                
                    self.optimizer.zero_grad()
                    #gc.collect()
                    #########################################
                    ############ VALIDATION SETS ############
                    #########################################
                    valid_results = {}
                    for valid_logger, valid_loader in zip(valid_logger_list, self.valid_loaders):
                        v_images, v_trans, _, _, _ = next(valid_loader)
                        
                        valid_results = evaluateDistance(model, v_images[0], v_images[1], v_trans[0],
                                                         loss_type = loss_type,
                                                         falloff_angle = self.falloff_angle,
                                                         optimizer = None, disp_metrics = True)

                        valid_info = {}
                        for k,v in valid_results.items():
                            if('vec' not in k):
                                valid_info[k] = v
                            else:
                                valid_logger.histo_summary(k,v, cumulative_batch_idx+1)

                       
                        for tag, value in valid_info.items():
                            valid_logger.scalar_summary(tag, value, cumulative_batch_idx+1)
                        
                        #gc.collect()

                    if('loss' in valid_results and valid_results['loss'] < min_loss):
                        min_loss = valid_results['loss']
                        weights_filename = os.path.join(weights_dir, 'best_quat.pth')
                        print("saving model ", weights_filename)
                        torch.save(model.state_dict(), weights_filename)                        

                checkpoint_model = not((cumulative_batch_idx+1) % checkpoint_every_nth)
                if(checkpoint_model):
                    checkpoint_weights_filename = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(cumulative_batch_idx+1))
                    print("checkpointing model ", checkpoint_weights_filename)
                    torch.save(model.state_dict(), checkpoint_weights_filename)
                cumulative_batch_idx += 1

def main():
    import datetime
    from argparse import ArgumentParser
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_folder', type=str, default=None)
    parser.add_argument('--valid_class_folder', type=str, default=None)
    parser.add_argument('--valid_model_folder', type=str, default=None)
    parser.add_argument('--valid_pose_folder', type=str, default=None)
    
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--model_data_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--falloff_angle', type=float, default=45.0)
    parser.add_argument('--rejection_thresh_angle', type=float, default=25.0)
    parser.add_argument('--max_orientation_iters', type=int, default=200)

    parser.add_argument('--cross_model_eval', dest='cross_model_eval', action='store_true')    

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='exp')
    parser.add_argument('--nonsiamese_features', dest='siamese_features', action='store_false')
    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=100)
    parser.add_argument('--checkpoint_every_nth', type=int, default=10000)
    parser.add_argument('--num_display_imgs', type=int, default=0)

    args = parser.parse_args()

    if(args.model_data_file is not None):
        model_filenames = {}
        if(args.model_data_file[-4:] == '.txt'):
            with open(args.model_data_file, 'r') as f:    
                filenames = f.read().split()
        else:
            filenames = [args.model_data_file]
        
        for path in filenames:
            model = path.split('/')[-2]
            model_filenames[model] = path
    else:
        model_filenames = None

    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None
        
    trainer = PoseDistanceTrainer(train_data_folder = args.train_data_folder,
                                  valid_class_folder = args.valid_class_folder,
                                  valid_model_folder = args.valid_model_folder,
                                  valid_pose_folder = args.valid_pose_folder,
                                  img_size = (args.width,args.height),
                                  falloff_angle = args.falloff_angle*np.pi/180.0,
                                  rejection_thresh_angle = args.rejection_thresh_angle*np.pi/180.0,
                                  max_orientation_iters = args.max_orientation_iters,
                                  batch_size = args.batch_size,
                                  num_workers = args.num_workers,
                                  model_filenames = model_filenames,
                                  background_filenames = background_filenames,
                                  seed = args.random_seed,
                                  )


    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    model = gen_pose_net(args.model_type.lower(), 
                         args.compare_type.lower(), 
                         output_dim = 1,
                         pretrained = args.pretrained,
                         siamese_features = args.siamese_features)

    if args.weight_file is not None:
        load_state_dict(model, args.weight_file)

    trainer.train(model, results_dir,
                  loss_type = args.loss_type,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  num_display_imgs = args.num_display_imgs,
                  checkpoint_every_nth = args.checkpoint_every_nth)

if __name__=='__main__':
    main()
