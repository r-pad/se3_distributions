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
from generic_pose.datasets.tensor_dataset import TensorDataset
from generic_pose.datasets.concat_dataset import ConcatDataset
from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils import to_np, to_var
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.utils.image_preprocessing import preprocessImages

from quat_math import random_quaternion

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

class FinetuneYCBTrainer(object):
    def __init__(self, 
                 benchmark_folder,
                 renders_folder,
                 target_object,
                 render_offset = None,
                 use_exact_render = False,
                 base_level = 2,
                 img_size = (224,224),
                 falloff_angle = np.pi/4,
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
        self.use_exact_render = use_exact_render

        self.ycb_dataset = YCBDataset(data_dir=benchmark_folder, 
                                        image_set='train_split',
                                        img_size=img_size,
                                        obj=target_object,
                                        use_syn_data=True)
        self.ycb_dataset.loop_truth = None
        self.ycb_dataset.append_rendered = use_exact_render

        self.rendered_dataset = TensorDataset(renders_folder,
                                              self.ycb_dataset.getModelFilename(),
                                              img_size=img_size,
                                              offset_quat = render_offset,
                                              base_level = base_level)

        self.rendered_dataset.loop_truth = None
        self.rendered_dataset.append_rendered = use_exact_render
        print("Renders Loaded")
        self.train_dataset = ConcatDataset(self.ycb_dataset, self.rendered_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                     num_workers=num_workers, 
                                     batch_size=int(batch_size/2), 
                                     shuffle=True)

        
        self.valid_dataset = YCBDataset(data_dir=benchmark_folder, 
                                        image_set='valid_split',
                                        img_size=img_size,
                                        obj=target_object)
        self.valid_dataset.loop_truth = None
        #self.valid_dataset.append_rendered = use_exact_render
        
        self.valid_loader = DataLoader(self.valid_dataset,
                                       num_workers=num_workers, 
                                       batch_size=int(batch_size/2), 
                                       shuffle=True)
 
         
        self.grid = S3Grid(base_level)
        self.renderer = BpyRenderer(transform_func = ycbRenderTransform)
        self.renderer.loadModel(self.ycb_dataset.getModelFilename(),
                                emit = 0.5)
        self.renderPoses = self.renderer.renderPose
        base_render_folder = os.path.join(benchmark_folder,
                                          'base_renders',
                                          self.ycb_dataset.getObjectName(),
                                          '{}'.format(base_level))
        if(os.path.exists(os.path.join(base_render_folder, 'renders.pt'))):
            self.base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
            self.base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
        else:
            self.base_vertices = np.unique(self.grid.vertices, axis = 0)
            self.base_renders = preprocessImages(self.renderPoses(self.base_vertices, camera_dist = 0.33),
                                                 img_size = self.img_size,
                                                 normalize_tensors = True).float()
            import pathlib
            pathlib.Path(base_render_folder).mkdir(parents=True, exist_ok=True)
            torch.save(self.base_renders, os.path.join(base_render_folder, 'renders.pt'))
            torch.save(self.base_vertices, os.path.join(base_render_folder, 'vertices.pt'))
        
        print("Grid Loaded")
        self.base_size = self.base_vertices.shape[0]

    def train(self, model, results_dir,
              loss_type = 'exp',
              num_indices = 256,
              uniform_prop = 0.5,
              loss_temperature = None,
              num_epochs = 100000,
              log_every_nth = 100,
              checkpoint_every_nth = 10000,
              lr = 1e-5,
              optimizer = 'SGD'):
        
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
            
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        log_dir = os.path.join(results_dir,'logs')
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
        dataset_size = len(self.train_loader)
        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (query_imgs, query_quats, _1, _2) in enumerate(self.train_loader):
                del _1, _2
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                #pre_info, pre_objs  = getTensors()     
                torch.cuda.empty_cache()
                #import IPython; IPython.embed()
                train_results = evaluateRenderedDistance(model, self.grid, self.renderer,
                                                         query_imgs, query_quats,
                                                         self.base_renders, self.base_vertices,
                                                         loss_type = loss_type,
                                                         falloff_angle = self.falloff_angle,
                                                         optimizer = self.optimizer, 
                                                         disp_metrics = log_data,
                                                         num_indices = num_indices,
                                                         uniform_prop = uniform_prop,
                                                         loss_temperature = loss_temperature)

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
                    query_imgs, query_quats, _1, _2 = next(iter(self.valid_loader))
                    del _1, _2
                    torch.cuda.empty_cache()
                    valid_results = evaluateRenderedDistance(model, self.grid, self.renderer,
                                                             query_imgs, query_quats,
                                                             self.base_renders, self.base_vertices,
                                                             loss_type = loss_type,
                                                             falloff_angle = self.falloff_angle,
                                                             optimizer = None, 
                                                             disp_metrics = True,
                                                             num_indices = num_indices,
                                                             uniform_prop = uniform_prop,
                                                             loss_temperature = loss_temperature)

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
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    
    parser = ArgumentParser()

    parser.add_argument('--benchmark_folder', type=str, default=None)
    parser.add_argument('--target_object', type=int, default=1)
    parser.add_argument('--renders_folder', type=str, default=None)
    parser.add_argument('--random_render_offset', dest='random_render_offset', action='store_true')
    parser.add_argument('--base_level', type=int, default=2)
    parser.add_argument('--use_exact_render', dest='use_exact_render', action='store_true')    
    
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--num_indices', type=int, default=256)
    parser.add_argument('--uniform_prop', type=float, default=0.5)
    parser.add_argument('--loss_temperature', type=float, default=None)
    parser.add_argument('--falloff_angle', type=float, default=45.0)
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
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_every_nth', type=int, default=100)
    parser.add_argument('--checkpoint_every_nth', type=int, default=500)

    args = parser.parse_args()

    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None
    
    if(args.random_render_offset):
        render_offset = random_quaternion()
    else:
        render_offset = None

    trainer = FinetuneYCBTrainer(benchmark_folder = args.benchmark_folder,
                                      target_object = args.target_object,
                                      renders_folder = args.renders_folder,
                                      render_offset = render_offset,
                                      use_exact_render = args.use_exact_render,
                                      img_size = (args.width,args.height),
                                      falloff_angle = args.falloff_angle*np.pi/180.0,
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
                  num_indices = args.num_indices,
                  uniform_prop = args.uniform_prop,
                  loss_temperature = args.loss_temperature,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  checkpoint_every_nth = args.checkpoint_every_nth)

if __name__=='__main__':
    main()
