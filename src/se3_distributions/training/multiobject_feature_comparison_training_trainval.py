# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import Adam, Adadelta, SGD
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange

from object_pose_utils.bbTrans.discretized4dSphere import S3Grid
from object_pose_utils.utils import to_np, to_var

from object_pose_utils.datasets.feature_dataset import UniformFeatureDataset, FeatureDataset
from se3_distributions.losses.feature_distance_loss import multiObjectLoss
from logger import Logger

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

class FeatureComparisonTrainer(object):
    def __init__(self, 
                 dataset_root,
                 feature_root,
                 feature_key,
                 num_augs,
                 falloff_angle,
                 batch_size,
                 num_workers,
                 seed,
                 fill_with_exact):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.falloff_angle = falloff_angle
        self.model_datasets = [UniformFeatureDataset(dataset_root = dataset_root,
                                                     feature_root = feature_root, 
                                                     mode = 'train_sym_val',
                                                     resample_on_error = True,
                                                     feature_key = feature_key,
                                                     num_augs = num_augs,
                                                     fill_with_exact = fill_with_exact,
                                                     object_label = obj) for obj in range(1,22)]

        self.train_dataset = ConcatDataset(self.model_datasets)
        self.train_loader = DataLoader(self.train_dataset,
                                       num_workers=num_workers-1,
                                       batch_size=batch_size, 
                                       shuffle=True)


        self.valid_dataset = FeatureDataset(dataset_root = dataset_root,
                                            feature_root = feature_root,
                                            mode = 'valid',
                                            resample_on_error = True,
                                            feature_key = feature_key,
                                            num_augs = 0,
                                            object_list = list(range(1,22)))
        
        self.valid_loader = DataLoader(self.valid_dataset,
                                       num_workers=1, 
                                       batch_size=batch_size, 
                                       shuffle=True)

        classes = self.valid_dataset.classes
        self.grid_vertices = {}
        self.grid_features = {}
        for obj in range(1,22):
            self.grid_vertices[obj] = to_var(torch.load(os.path.join(feature_root, 'grid', 
                '{}_vertices.pt'.format(classes[obj]))))
            self.grid_features[obj] = to_var(torch.load(os.path.join(feature_root, 'grid', 
                '{}_{}_features.pt'.format(feature_key, classes[obj]))))

    def train(self, model, 
              log_dir,
              checkpoint_dir,
              num_epochs,
              log_every_nth,
              checkpoint_every_nth,
              lr,
              optimizer,
              weight_top,
              ):
        
        model.train()
        model.cuda()
        last_checkpoint_filename = None
        if(optimizer.lower() == 'sgd'):
            self.optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        elif(optimizer.lower() == 'adam'):
            self.optimizer = Adam(model.parameters(), lr=lr)
        elif(optimizer.lower() == 'adadelta'):
            self.optimizer = Adadelta(model.parameters(), lr=lr)
        else:
            raise AssertionError('Unsupported Optimizer {}, only SGD, Adam, and Adadelta supported'.format(optimizer))
            
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir, 'all')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, 'all')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        log_dir = os.path.join(log_dir,'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_log_dir = os.path.join(log_dir,'train')
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        train_logger = Logger(train_log_dir)        
        
        valid_log_dir = os.path.join(log_dir,'valid')
        if not os.path.exists(valid_log_dir):
            os.makedirs(valid_log_dir)  
        valid_logger = Logger(valid_log_dir)        
                    
        weights_dir = os.path.join(checkpoint_dir,'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            
        cumulative_batch_idx = 0
        min_loss = float('inf')
        print('Starting Training')
        dataset_size = len(self.train_loader)
        for epoch_idx in trange(1, num_epochs+1):
            for batch_idx, data in tqdm(enumerate(self.train_loader), total = len(self.train_loader)):
                obj, feat, quat = data
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                torch.cuda.empty_cache()
                grid_features = []
                grid_vertices = []
                for idx in to_np(obj).flat:
                    grid_features.append(self.grid_features[idx])
                    grid_vertices.append(self.grid_vertices[idx])
                grid_features = torch.cat(grid_features)
                grid_vertices = torch.cat(grid_vertices)
            
                train_results = multiObjectLoss(model, obj.cuda()-1, 
                                                to_var(feat), to_var(quat),
                                                grid_features, grid_vertices,
                                                falloff_angle = self.falloff_angle,
                                                weight_top = weight_top,
                                                optimizer = self.optimizer, 
                                                calc_metrics = log_data,
                                                )

                torch.cuda.empty_cache()

                if log_data:
                    #print("epoch {} ({}):: cumulative_batch_idx {}".format(epoch_idx, time.time() - log_time, cumulative_batch_idx + 1))

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
                    #########################################
                    ############ VALIDATION SETS ############
                    #########################################
                    with torch.no_grad(): 
                        obj, feat, quat = next(iter(self.valid_loader))
                        torch.cuda.empty_cache()
                        grid_features = []
                        grid_vertices = []
                        for idx in to_np(obj).flat:
                            grid_features.append(self.grid_features[idx])
                            grid_vertices.append(self.grid_vertices[idx])
                        grid_features = torch.cat(grid_features)
                        grid_vertices = torch.cat(grid_vertices)
                        valid_results = multiObjectLoss(model, obj.cuda()-1, 
                                                        to_var(feat), to_var(quat),
                                                        grid_features, grid_vertices,
                                                        falloff_angle = self.falloff_angle,
                                                        weight_top = weight_top,
                                                        optimizer = None, 
                                                        calc_metrics = True,
                                                        )

                    torch.cuda.empty_cache()
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
                    
                    if(last_checkpoint_filename is not None):
                        os.remove(last_checkpoint_filename)
                    last_checkpoint_filename = checkpoint_weights_filename

                cumulative_batch_idx += 1

def main():
    import datetime
    from argparse import ArgumentParser
    from se3_distributions.models.compare_networks import SigmoidCompareNet
    from se3_distributions.models.deterministic_compare_networks import SigmoidCompareNet as DetSigmoidCompareNet

    parser = ArgumentParser()

    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--feature_folder', type=str)
    parser.add_argument('--feature_key', type=str, default = 'feat')
    parser.add_argument('--num_augs', type=int, default = 0)
    parser.add_argument('--fill_with_exact', action='store_true')
    parser.add_argument('--weight_file', type=str, default=None)

    parser.add_argument('--feature_size', type=int, default=1024)
    parser.add_argument('--falloff_angle', type=float, default=20.0)
    parser.add_argument('--weight_top', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dropout', action='store_true')
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--log_dir', type=str, default='results/') 
    parser.add_argument('--checkpoint_dir', type=str, default=None) 
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_every_nth', type=int, default=100)
    parser.add_argument('--checkpoint_every_nth', type=int, default=1000)

    args = parser.parse_args()

    trainer = FeatureComparisonTrainer(dataset_root = args.dataset_folder,
                                       feature_root = args.feature_folder,
                                       feature_key = args.feature_key,
                                       num_augs = args.num_augs,
                                       fill_with_exact = args.fill_with_exact,
                                       falloff_angle = args.falloff_angle*np.pi/180.0,
                                       batch_size = args.batch_size,
                                       num_workers = args.num_workers,
                                       seed = args.random_seed,
                                       )

    if(args.checkpoint_dir is None):
        args.checkpoint_dir = args.log_dir
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir,current_timestamp)    
    checkpoint_dir = os.path.join(args.checkpoint_dir,current_timestamp)    
     
    if(args.dropout):
        model = SigmoidCompareNet(args.feature_size, 21)
    else:
        model = DetSigmoidCompareNet(args.feature_size, 21)

    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    trainer.train(model,
                  log_dir = log_dir,
                  checkpoint_dir = checkpoint_dir,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  checkpoint_every_nth = args.checkpoint_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  weight_top = args.weight_top,
                  )

if __name__=='__main__':
    import socket
    import seuss_cluster_alerts as sca
    hostname = socket.gethostname()
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if(gpu_id is not None):
        hostname += ' GPU {}'.format(gpu_id)

    try:
        main()
        sca.sendAlert('bokorn@andrew.cmu.edu', 
                       message_subject='Job Completed on {}'.format(hostname))

    except:
        e = sys.exc_info()
        sca.sendAlert('bokorn@andrew.cmu.edu', 
                message_subject='Job Failed on {}'.format(hostname),
                message_text=str(e))
        raise
