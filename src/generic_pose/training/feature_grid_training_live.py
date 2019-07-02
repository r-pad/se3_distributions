# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

import torch
from torch.autograd import Variable
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

from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.uniform_ycb_dataset import UniformYCBDataset
from object_pose_utils.datasets.image_processing import ColorJitter, ImageNormalizer
from object_pose_utils.datasets.ycb_occlusion_augmentation import YCBOcclusionAugmentor
from object_pose_utils.datasets.point_processing import PointShifter
from object_pose_utils.datasets.inplane_rotation_augmentation import InplaneRotator

from dense_fusion.network import PoseNetGlobal
from generic_pose.losses.feature_grid_loss import evaluateLoss
from logger import Logger

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

class FeatureGridTrainer(object):
    def __init__(self, 
                 obj,
                 dataset_root,
                 feature_root,
                 falloff_angle,
                 num_augs,
                 batch_size,
                 num_workers,
                 seed):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.obj = obj
        self.falloff_angle = falloff_angle
        output_format = [otypes.OBJECT_LABEL,
                         otypes.QUATERNION, otypes.IMAGE_CROPPED, 
                         otypes.DEPTH_POINTS_MASKED_AND_INDEXES]

        preprocessors = [YCBOcclusionAugmentor(dataset_root), 
                         ColorJitter(),
                         InplaneRotator(),
                        ]
        postprocessors = [ImageNormalizer(), PointShifter()]
        
        self.batch_size = batch_size

        self.train_dataset = YCBDataset(dataset_root = dataset_root, 
                             mode= 'train_sym_offset',
                             object_list = [obj], 
                             output_data = output_format,
                             resample_on_error = True,
                             fill_with_exact = True,
                             add_syn_background = True,
                             add_syn_noise = True,
                             preprocessors = preprocessors, 
                             postprocessors = postprocessors,
                             image_size = [640, 480], num_points=1000)
        
        self.train_loader = DataLoader(self.train_dataset,
                                       num_workers=num_workers-1,
                                       batch_size=1, 
                                       shuffle=True)

        self.valid_dataset = YCBDataset(dataset_root = dataset_root,
                                        mode='valid',
                                        object_list = [obj], 
                                        output_data = output_format,
                                        resample_on_error = True,
                                        preprocessors = [],
                                        postprocessors = [ImageNormalizer()],
                                        image_size = [640, 480], num_points=1000)
        
        self.valid_loader = DataLoader(self.valid_dataset,
                                       num_workers=1, 
                                       batch_size=1, 
                                       shuffle=True)

        classes = self.valid_dataset.classes
        self.grid_vertices = {}
        #for obj in range(1,22):
        self.grid_vertices[obj] = torch.load(os.path.join(feature_root, 'grid', 
            '{}_vertices.pt'.format(classes[obj])))

    def train(self, model, 
              log_dir,
              checkpoint_dir,
              num_epochs,
              log_every_nth,
              checkpoint_every_nth,
              lr,
              optimizer,
              #weight_top,
              ):
        self.estimator = PoseNetGlobal(num_points = 1000, num_obj = 21)
        self.estimator.cuda()
        estimator_weights = '/home/bokorn/src/DenseFusion/trained_checkpoints/ycb/pose_model_global_13_0.02780649198161978.pth'
        self.estimator.load_state_dict(torch.load(estimator_weights))

        self.estimator.eval()
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
        log_dir = os.path.join(log_dir, self.valid_dataset.classes[self.obj])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, self.valid_dataset.classes[self.obj])
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
        feat = []
        quat = []
        obj = []
        dataset_size = len(self.train_loader)
        for epoch_idx in trange(1, num_epochs+1):
            for iter_idx, data in tqdm(enumerate(self.train_loader), total = len(self.train_loader)):
                idx, q, img, points, choose = data
                obj.append(int(idx))
                idx = idx - 1
                img = Variable(img).cuda()
                points = Variable(points).cuda()
                choose = Variable(choose).cuda()
                idx = Variable(idx).cuda()
                feat.append(self.estimator.globalFeature(img, points, choose, idx)[0].detach())
                quat.append(q)
                if((iter_idx+1) % self.batch_size == 0 and iter_idx > 0):
                    feat = torch.stack(feat)
                    quat = torch.cat(quat)

                    log_data = not((cumulative_batch_idx+1) % log_every_nth)
                    torch.cuda.empty_cache()
                    grid_vertices = []
                    for idx in obj:
                        grid_vertices.append(self.grid_vertices[idx])
                    grid_vertices = torch.cat(grid_vertices)
                    train_results = evaluateLoss(model, 
                                                 to_var(feat), to_var(quat),
                                                 to_var(grid_vertices),
                                                 falloff_angle = self.falloff_angle,
                                                 #weight_top = weight_top,
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
                        feat = []
                        quat = []
                        obj = []
                        for _ in range(self.batch_size):
                            data = next(iter(self.valid_loader))
                            idx, q, img, points, choose = data
                            obj.append(int(idx))
                            idx = idx - 1
                            img = Variable(img).cuda()
                            points = Variable(points).cuda()
                            choose = Variable(choose).cuda()
                            idx = Variable(idx).cuda()
                            feat.append(self.estimator.globalFeature(img, points, choose, idx)[0].detach())
                            quat.append(q)
                        feat = torch.stack(feat)
                        quat = torch.cat(quat)

                        torch.cuda.empty_cache()
                        grid_vertices = []
                        for idx in obj:
                            grid_vertices.append(self.grid_vertices[idx])
                        grid_vertices = torch.cat(grid_vertices)
                        valid_results = evaluateLoss(model, 
                                                     to_var(feat), to_var(quat),
                                                     to_var(grid_vertices),
                                                     falloff_angle = self.falloff_angle,
                                                     #weight_top = weight_top,
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
                    feat = []
                    quat = []
                    obj = []
 

def main():
    import datetime
    from argparse import ArgumentParser
    from generic_pose.models.compare_networks import SingleNet

    parser = ArgumentParser()

    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--feature_folder', type=str)

    parser.add_argument('--object_index', type=int)
    parser.add_argument('--weight_file', type=str, default=None)

    parser.add_argument('--num_augs', type=int, default = 0)
    parser.add_argument('--feature_size', type=int, default=1024)
    parser.add_argument('--falloff_angle', type=float, default=20.0)
    #parser.add_argument('--weight_top', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--log_dir', type=str, default='results/') 
    parser.add_argument('--checkpoint_dir', type=str, default=None) 
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_every_nth', type=int, default=100)
    parser.add_argument('--checkpoint_every_nth', type=int, default=1000)

    args = parser.parse_args()

    trainer = FeatureGridTrainer(obj = args.object_index,
                                       dataset_root = args.dataset_folder,
                                       feature_root = args.feature_folder,
                                       falloff_angle = args.falloff_angle*np.pi/180.0,
                                       num_augs = args.num_augs,
                                       batch_size = args.batch_size,
                                       num_workers = args.num_workers,
                                       seed = args.random_seed,
                                       )


    if(args.checkpoint_dir is None):
        args.checkpoint_dir = args.log_dir
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir,current_timestamp)    
    checkpoint_dir = os.path.join(args.checkpoint_dir,current_timestamp)    
    
    model = SingleNet(args.feature_size, 3885)

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
                  #weight_top = args.weight_top,
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
