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
from torch.utils.data import ConcatDataset

import os
import sys
import time
import gc
from itertools import cycle
import numpy as np
from logger import Logger

from generic_pose.bbTrans.discretized4dSphere import S3Grid
#from generic_pose.datasets.concat_dataset import ConcatDataset
from generic_pose.datasets.tensor_dataset import TensorDataset
from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils import to_np, to_var
from generic_pose.training.class_utils import evaluateClassDistance
from generic_pose.utils.image_preprocessing import preprocessImages
from generic_pose.utils.pose_processing import getGaussianKernal
from generic_pose.eval.posecnn_eval import getYCBThresholds
from quat_math import random_quaternion

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

class FinetuneYCBTrainer(object):
    def __init__(self, 
                 benchmark_folder,
                 target_object,
                 render_offset = None,
                 render_proportion = 1.0,
                 brightness_jitter = 0,
                 contrast_jitter = 0,
                 saturation_jitter = 0,
                 hue_jitter = 0,
                 max_translation = (0, 0),
                 max_scale = (0, 0),
                 rotate_image = False,
                 max_num_occlusions = 0,
                 max_occlusion_area = (0, 0),
                 augmentation_prob = 0,
                 base_level = 2,
                 renders_folder = None,
                 img_size = (224,224),
                 falloff_angle = np.pi/4,
                 kernal_sigma = None,
                 batch_size = 16,
                 num_workers = 4,
                 background_filenames = None,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.falloff_angle = falloff_angle
   
        self.ycb_dataset = YCBDataset(data_dir=benchmark_folder, 
                                        image_set='train_split',
                                        img_size=img_size,
                                        obj=target_object,
                                        use_syn_data=True,
                                        brightness_jitter = brightness_jitter,
                                        contrast_jitter = contrast_jitter,
                                        saturation_jitter = saturation_jitter,
                                        hue_jitter = hue_jitter,
                                        max_translation = max_translation,
                                        max_scale = max_scale,
                                        rotate_image = rotate_image,
                                        max_num_occlusions = max_num_occlusions,
                                        max_occlusion_area = max_occlusion_area,
                                        augmentation_prob = augmentation_prob)

        self.ycb_dataset.loop_truth = None
        self.ycb_dataset.append_rendered = False

        self.rendered_dataset = TensorDataset(renders_folder,
                                              self.ycb_dataset.getModelFilename(),
                                              img_size=img_size,
                                              offset_quat = render_offset,
                                              base_level = base_level,
                                        brightness_jitter = brightness_jitter,
                                        contrast_jitter = contrast_jitter,
                                        saturation_jitter = saturation_jitter,
                                        hue_jitter = hue_jitter,
                                        max_translation = max_translation,
                                        max_scale = max_scale,
                                        rotate_image = rotate_image,
                                        max_num_occlusions = max_num_occlusions,
                                        max_occlusion_area = max_occlusion_area,
                                        augmentation_prob = augmentation_prob)

        self.rendered_dataset.loop_truth = None
        self.rendered_dataset.append_rendered = False
        print("Renders Loaded")
        if(render_proportion is None):
            render_multi = 1
            ycb_multi = 1
        else:
            ycb_multi = 1
            render_multi = int(max(1,
                np.ceil(render_proportion * len(self.ycb_dataset)/len(self.rendered_dataset))))
            print('YCB Size: {}'.format(len(self.ycb_dataset)))
            print('YCB Mulit: {}'.format(ycb_multi))
            print('Render Size: {}'.format(len(self.rendered_dataset)))
            print('Render Mulit: {}'.format(render_multi))
        self.train_dataset = ConcatDataset((self.ycb_dataset,)*ycb_multi + (self.rendered_dataset,)*render_multi)
        self.train_loader = DataLoader(self.train_dataset,
                                     num_workers=num_workers-1, 
                                     batch_size=batch_size, 
                                     shuffle=True)

        
        self.valid_dataset = YCBDataset(data_dir=benchmark_folder, 
                                        image_set='valid_split',
                                        img_size=img_size,
                                        obj=target_object)
        self.valid_dataset.loop_truth = None
        #self.valid_dataset.append_rendered = use_exact_render
        self.points = self.valid_dataset.getObjectPoints()
        cls = self.valid_dataset.getObjectName()
        self.use_sym = cls == '024_bowl' or cls == '036_wood_block' or cls == '061_foam_brick'
        self.add_threshold = getYCBThresholds()[target_object]
        self.valid_loader = DataLoader(self.valid_dataset,
                                       num_workers=1, 
                                       batch_size=batch_size, 
                                       shuffle=True)
 
         
        self.grid = S3Grid(base_level)
        self.base_vertices = to_var(torch.tensor(np.unique(self.grid.vertices, axis = 0)))
        
        print("Grid Loaded")
        self.base_size = self.base_vertices.shape[0]
        if(kernal_sigma is not None):
            self.kernal = getGaussianKernal(self.base_vertices, kernal_sigma)
        else:
            self.kernal = None

    def train(self, model, 
              log_dir,
              checkpoint_dir,
              loss_type = 'CrossEntropy',
              num_epochs = 100000,
              log_every_nth = 100,
              checkpoint_every_nth = 10000,
              lr = 1e-5,
              optimizer = 'SGD',
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
            
        add_data = {'points':self.points,
                    'use_sym':self.use_sym,
                    'threshold':self.add_threshold}
        
        #from generic_pose.utils.torch_utils import getTensors
        #init_info, init_objs  = getTensors()
        #import IPython; IPython.embed()
        cumulative_batch_idx = 0
        min_loss = float('inf')
        print('Starting Training')
        log_time = time.time()
        dataset_size = len(self.train_loader)
        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (imgs, label_qs, _1, _2) in enumerate(self.train_loader):
                del _1, _2
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                #pre_info, pre_objs  = getTensors()     
                torch.cuda.empty_cache()
                #import IPython; IPython.embed()
                train_results = evaluateClassDistance(model,
                                                      imgs, label_qs,
                                                      self.base_vertices,
                                                      loss_type = loss_type,
                                                      falloff_angle = self.falloff_angle,
                                                      optimizer = self.optimizer, 
                                                      calc_metrics = log_data,
                                                      kernal = self.kernal,
                                                      )

                #print(len(gc.get_objects()))
                #print(sum(sys.getsizeof(i) for i in gc.get_objects()))
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
                    imgs, label_qs, _1, indices = next(iter(self.valid_loader))
                    del _1, 
                    torch.cuda.empty_cache()
                    add_data['trans_true'] = [] 
                    add_data['trans_pcnn'] = []
                    for idx in indices:
                        mat_true = self.valid_dataset.getTrans(idx, use_gt = True)
                        mat_pcnn = self.valid_dataset.getTrans(idx, use_gt = False)
                        add_data['trans_true'].append(mat_true[:3,3])
                        if(mat_pcnn is not None):
                            add_data['trans_pcnn'].append(mat_pcnn[:3,3])
                        else:
                            add_data['trans_pcnn'].append(None)

                    valid_results = evaluateClassDistance(model,
                                                          imgs, label_qs,
                                                          self.base_vertices,
                                                          loss_type = loss_type,
                                                          falloff_angle = self.falloff_angle,
                                                          optimizer = None, 
                                                          calc_metrics = True,
                                                          kernal = self.kernal,
                                                          add_data = add_data,
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
    from generic_pose.models.class_networks import gen_class_net
    
    parser = ArgumentParser()

    parser.add_argument('--benchmark_folder', type=str, default=None)
    parser.add_argument('--target_object', type=int, default=1)
    parser.add_argument('--renders_folder', type=str, default=None)
    parser.add_argument('--render_proportion', type=float, default=1.0)
    parser.add_argument('--base_level', type=int, default=2)
    parser.add_argument('--random_render_offset', dest='random_render_offset', action='store_true')

    parser.add_argument('--augmentation_probability', type=float, default=0)    
    parser.add_argument('--brightness_jitter', type=float, default=0)
    parser.add_argument('--contrast_jitter', type=float, default=0)
    parser.add_argument('--saturation_jitter', type=float, default=0)
    parser.add_argument('--hue_jitter', type=float, default=0)
    parser.add_argument('--max_translation', type=float, default=0) 
    parser.add_argument('--min_scale', type=float, default=0)
    parser.add_argument('--max_scale', type=float, default=0)
    parser.add_argument('--rotate_image', dest='rotate_image', action='store_true')    
    parser.add_argument('--max_num_occlusions', type=int, default=0)
    parser.add_argument('--min_occlusion_area', type=float, default=0) 
    parser.add_argument('--max_occlusion_area', type=float, default=0) 
 
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--falloff_angle', type=float, default=20.0)
    parser.add_argument('--kernal_sigma', type=float, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='alexnet')
    parser.add_argument('--output_type', type=str, default='sigmoid')
    parser.add_argument('--loss_type', type=str, default='CrossEntropy')
    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--log_dir', type=str, default='results/') 
    parser.add_argument('--checkpoint_dir', type=str, default=None) 
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_every_nth', type=int, default=100)
    parser.add_argument('--checkpoint_every_nth', type=int, default=1000)

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
  
    if(args.max_translation is not None):
        max_translation = (args.max_translation, args.max_translation)
    else:
        max_translation = None
    if(args.min_scale is None and args.max_scale is None):
        max_scale = None
    else:
        if(args.max_scale is None):
            args.max_scale = 1
        if(args.min_scale is None):
            args.min_scale = 1
        max_scale = (args.min_scale, args.max_scale)                                      
                                                                                  
    trainer = FinetuneYCBTrainer(benchmark_folder = args.benchmark_folder,
                                      target_object = args.target_object,
                                      renders_folder = args.renders_folder,
                                      render_offset = render_offset,
                                      render_proportion = args.render_proportion,
                                      brightness_jitter = args.brightness_jitter,
                                      contrast_jitter = args.contrast_jitter,
                                      saturation_jitter = args.saturation_jitter,
                                      hue_jitter = args.hue_jitter,
                                      max_translation = max_translation,
                                      max_scale = max_scale,
                                      rotate_image = args.rotate_image,
                                      max_num_occlusions = args.max_num_occlusions,
                                      max_occlusion_area = (args.min_occlusion_area, args.max_occlusion_area),
                                      augmentation_prob = args.augmentation_probability,
                                      img_size = (args.width,args.height),
                                      falloff_angle = args.falloff_angle*np.pi/180.0,
                                      kernal_sigma = args.kernal_sigma,
                                      base_level = args.base_level,
                                      batch_size = args.batch_size,
                                      num_workers = args.num_workers,
                                      background_filenames = background_filenames,
                                      seed = args.random_seed,
                                      )


    if(args.checkpoint_dir is None):
        args.checkpoint_dir = args.log_dir
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir,current_timestamp)    
    checkpoint_dir = os.path.join(args.checkpoint_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    model = gen_class_net(args.model_type.lower(), 
                          args.output_type.lower(), 
                          output_dim = trainer.base_size,
                          pretrained = args.pretrained,
                          )
    if args.weight_file is not None:
        model.load_state_dict(args.weight_file)

    trainer.train(model, log_dir, checkpoint_dir,
                  loss_type = args.loss_type,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  checkpoint_every_nth = args.checkpoint_every_nth,
                  )

if __name__=='__main__':
    main()
