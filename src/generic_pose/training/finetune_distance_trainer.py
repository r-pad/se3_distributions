# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from model_renderer.pose_renderer import BpyRenderer

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import Adam, Adadelta, SGD
from torch.utils.data import DataLoader

import os
import time
import gc
from itertools import cycle
import numpy as np
from logger import Logger

from generic_pose.bbTrans.discretized4dSphere import S3Grid
from generic_pose.datasets.numpy_dataset import NumpyImageDataset
from generic_pose.datasets.sixdc_dataset import SixDCDataset
from generic_pose.utils import to_np, to_var
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.utils.image_preprocessing import preprocessImages

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

class FinetuneDistanceTrainer(object):
    def __init__(self, 
                 benchmark_folder,
                 target_object,
                 base_level = 2,
                 renders_folder = None,
                 img_size = (224,224),
                 falloff_angle = np.pi/4,
                 rejection_thresh_angle = 25*np.pi/180,
                 max_orientation_angle = None,
                 max_orientation_iters = 200,
                 batch_size = 32,
                 num_workers = 4,
                 background_filenames = None,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.falloff_angle = falloff_angle
        
        self.sixdc_dataset = SixDCDataset(data_dir=benchmark_folder, img_size=img_size)
        self.sixdc_dataset.loop_truth = [1]
        self.sixdc_dataset.setSequence('{:02d}'.format(target_object))
        
        self.benchmark_loader = DataLoader(self.sixdc_dataset,
                                           num_workers=num_workers, 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
 
        if(renders_folder is not None):
            self.render_dataset = NumpyImageDataset(data_folders=renders_folder,
                                                    img_size = img_size,
                                                    rejection_thresh_angle = rejection_thresh_angle,
                                                    max_orientation_offset = max_orientation_angle,
                                                    max_orientation_iters = max_orientation_iters,
                                                    background_filenames = background_filenames,
                                                    classification=False,
                                                    )
       
        self.grid = S3Grid(base_level)
        self.renderer = BpyRenderer()
        self.renderer.loadModel(self.sixdc_dataset.getModelFilename(),
                                model_scale = self.sixdc_dataset.getModelScale(), 
                                emit = 0.5)
        self.renderPoses = self.renderer.renderPose
        self.base_vertices = np.unique(self.grid.vertices, axis = 0)
        self.base_size = self.base_vertices.shape[0]
        self.base_renders = preprocessImages(self.renderPoses(self.base_vertices),
                                             img_size = self.img_size,
                                             normalize_tensors = True).float()

        self.valid_types = []
        self.valid_loaders = []

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
            
        
        #from generic_pose.utils.torch_utils import getTensors
        #init_info, init_objs  = getTensors()
        #import IPython; IPython.embed()
        cumulative_batch_idx = 0
        min_loss = float('inf')
        print('Starting Training')
        log_time = time.time()
        dataset_size = len(self.benchmark_loader)
        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (query_imgs, _1, query_quats, _2, _3) in enumerate(self.benchmark_loader):
                del _1[0], _1,  _2[0], _2, _3[0], _3
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                #pre_info, pre_objs  = getTensors()     
                torch.cuda.empty_cache()
                #import IPython; IPython.embed()
                train_results = evaluateRenderedDistance(model, self.grid, self.renderer,
                                                         query_imgs[0], query_quats[0],
                                                         self.base_renders, self.base_vertices,
                                                         loss_type = loss_type,
                                                         falloff_angle = self.falloff_angle,
                                                         optimizer = self.optimizer, 
                                                         disp_metrics = log_data)

                torch.cuda.empty_cache()
                #import IPython; IPython.embed()

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

    parser.add_argument('--benchmark_folder', type=str, default=None)
    parser.add_argument('--target_object', type=int, default=1)
    parser.add_argument('--renders_folder', type=str, default=None)
    parser.add_argument('--base_level', type=int, default=2)
    
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--falloff_angle', type=float, default=45.0)
    parser.add_argument('--rejection_thresh_angle', type=float, default=25.0)
    parser.add_argument('--max_orientation_iters', type=int, default=200)

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

    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None
        
    trainer = FinetuneDistanceTrainer(benchmark_folder = args.benchmark_folder,
                                      target_object = args.target_object,
                                      renders_folder = args.renders_folder,
                                      img_size = (args.width,args.height),
                                      falloff_angle = args.falloff_angle*np.pi/180.0,
                                      rejection_thresh_angle = args.rejection_thresh_angle*np.pi/180.0,
                                      base_level = args.base_level,
                                      max_orientation_iters = args.max_orientation_iters,
                                      batch_size = args.batch_size,
                                      num_workers = args.num_workers,
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
