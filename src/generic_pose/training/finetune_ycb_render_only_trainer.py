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
from generic_pose.training.finetune_distance_utils import evaluateRenderedDistance
from generic_pose.utils.image_preprocessing import preprocessImages
from quat_math import random_quaternion
from generic_pose.utils.tqdm_utils import std_out_err_redirect_tqdm
from tqdm import tqdm

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

class FinetuneYCBTrainer(object):
    def __init__(self, 
                 benchmark_folder,
                 target_object,
                 use_exact_render = False,
                 render_offset = None,
                 render_proportion = None,
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
                 batch_size = 16,
                 num_workers = 4,
                 background_filenames = None,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.falloff_angle = falloff_angle
        self.use_exact_render = use_exact_render

        self.valid_dataset = YCBDataset(data_dir=benchmark_folder, 
                                        image_set='valid_split',
                                        img_size=img_size,
                                        obj=target_object) 
        self.valid_dataset.loop_truth = None
        
        self.train_dataset = TensorDataset(renders_folder,
                                              self.valid_dataset.getModelFilename(),
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

        self.train_dataset.loop_truth = None
        self.train_dataset.append_rendered = use_exact_render
        print("Renders Loaded")
        self.train_loader = DataLoader(self.train_dataset,
                                     num_workers=num_workers-1, 
                                     batch_size=batch_size, 
                                     shuffle=True)

        

        #self.valid_dataset.append_rendered = use_exact_render
        
        self.valid_loader = DataLoader(self.valid_dataset,
                                       num_workers=1, 
                                       batch_size=batch_size, 
                                       shuffle=True)
 
         
        base_render_folder = os.path.join(benchmark_folder,
                                          'base_renders',
                                          self.valid_dataset.getObjectName(),
                                          '{}'.format(base_level))
        if(os.path.exists(os.path.join(base_render_folder, 'renders.pt'))):
            self.base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
            self.base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
        else:
            self.grid = S3Grid(base_level)
            self.renderer = BpyRenderer(transform_func = ycbRenderTransform)
            self.renderer.loadModel(self.valid_dataset.getModelFilename(),
                                    emit = 0.5)
            self.renderPoses = self.renderer.renderPose
            self.base_vertices = np.unique(self.grid.vertices, axis = 0)
            self.base_renders = preprocessImages(self.renderPoses(self.base_vertices, camera_dist = 0.33),
                                                 img_size = self.img_size,
                                                 normalize_tensors = True).float()
            import pathlib
            pathlib.Path(base_render_folder).mkdir(parents=True, exist_ok=True)
            torch.save(self.base_renders, os.path.join(base_render_folder, 'renders.pt'))
            torch.save(self.base_vertices, os.path.join(base_render_folder, 'vertices.pt'))
        
        print("Grid Loaded")
        self.base_renders.pin_memory()
        self.base_size = self.base_vertices.shape[0]

    def train(self, model, 
              log_dir,
              checkpoint_dir,
              loss_type = 'exp',
              num_indices = 256,
              image_chunk_size = 500,
              per_instance = False,
              top_n = 0,
              sample_by_loss = False,
              num_epochs = 100000,
              log_every_nth = 100,
              checkpoint_every_nth = 10000,
              lr = 1e-5,
              optimizer = 'SGD',
              sampling_distribution = None, 
              start_idx = 0,
              num_steps = 200000):
        
        with std_out_err_redirect_tqdm() as orig_stdout:
            pbar = tqdm(total=num_steps, initial = start_idx, 
                        file=orig_stdout, dynamic_ncols=True) 
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
                
            
            #from generic_pose.utils.torch_utils import getTensors
            #init_info, init_objs  = getTensors()
            #import IPython; IPython.embed()
            cumulative_batch_idx = start_idx
            min_loss = float('inf')
            print('Starting Training')
            log_time = time.time()
            dataset_size = len(self.train_loader)
            for epoch_idx in range(1, num_epochs+1):
                for batch_idx, (query_imgs, query_quats, _1, _2) in enumerate(self.train_loader):
                    del _1, _2
                    pbar.update()
                    log_data = not((cumulative_batch_idx+1) % log_every_nth)
                    #pre_info, pre_objs  = getTensors()     
                    torch.cuda.empty_cache()
                    #import IPython; IPython.embed()
                    train_results = evaluateRenderedDistance(model, #self.grid, self.renderer,
                                                             query_imgs, query_quats,
                                                             self.base_renders, self.base_vertices,
                                                             loss_type = loss_type,
                                                             falloff_angle = self.falloff_angle,
                                                             optimizer = self.optimizer, 
                                                             calc_metrics = log_data,
                                                             num_indices = num_indices,
                                                             image_chunk_size = image_chunk_size,
                                                             per_instance = per_instance,
                                                             sample_by_loss = sample_by_loss,
                                                             top_n = top_n,
                                                             sampling_distribution = sampling_distribution)

                    #print(len(gc.get_objects()))
                    #print(sum(sys.getsizeof(i) for i in gc.get_objects()))
                    torch.cuda.empty_cache()
                    #import IPython; IPython.embed()

                    if log_data:
                        print("epoch {} ({}):: cumulative_batch_idx {}".format(epoch_idx, time.time() - log_time, cumulative_batch_idx + 1))

                        print('Timestamps: {}'.format(time.time()))
                        #pbar.update(cumulative_batch_idx - pbar.n)

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
                        del _1,  _2
                        torch.cuda.empty_cache()
                        valid_results = evaluateRenderedDistance(model, #self.grid, self.renderer,
                                                                 query_imgs, query_quats,
                                                                 self.base_renders, self.base_vertices,
                                                                 loss_type = loss_type,
                                                                 falloff_angle = self.falloff_angle,
                                                                 optimizer = None, 
                                                                 calc_metrics = True,
                                                                 num_indices = num_indices,
                                                                 image_chunk_size = image_chunk_size)

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
                    if(cumulative_batch_idx > num_steps):
                        print('Reached Max Iterations {}'.format(cumulative_batch_idx))
                        return

def main():
    import datetime
    import glob
    from argparse import ArgumentParser
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    
    parser = ArgumentParser()

    parser.add_argument('--benchmark_folder', type=str, default=None)
    parser.add_argument('--target_object', type=int, default=1)
    parser.add_argument('--renders_folder', type=str, default=None)
    parser.add_argument('--render_proportion', type=float, default=1.0)
    parser.add_argument('--base_level', type=int, default=2)
    parser.add_argument('--use_exact_render', dest='use_exact_render', action='store_true')    
    parser.add_argument('--per_instance_sampling', dest='per_instance', action='store_true')    
    parser.add_argument('--sample_by_loss', dest='sample_by_loss', action='store_true')    
    parser.add_argument('--sampling_distribution', type=str, default='None')    
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

    parser.add_argument('--num_indices', type=int, default=256)
    parser.add_argument('--image_chunk_size', type=int, default=500)
    parser.add_argument('--top_n', type=int, default=0)
    parser.add_argument('--falloff_angle', type=float, default=45.0)

    parser.add_argument('--batch_size', type=int, default=16)
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

    parser.add_argument('--log_dir', type=str, default='results/') 
    parser.add_argument('--checkpoint_dir', type=str, default=None) 
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=200000)
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
                                      use_exact_render = args.use_exact_render,
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

    model = gen_pose_net(args.model_type.lower(), 
                         args.compare_type.lower(), 
                         output_dim = 1,
                         pretrained = args.pretrained,
                         siamese_features = args.siamese_features)

    files = glob.glob(args.checkpoint_dir + '/**/checkpoint_*.pth', recursive=True)
    max_step = 0
    weight_file = args.weight_file
    for fn in files:
        step = int(fn.split('_')[-1][:-4])
        if(step >= max_step):
            max_step = step
            weight_file = fn

    if(max_step > 0):
        print('Starting at Step {} using checkpoint {}'.format(max_step, weight_file))

    if weight_file is not None:
        load_state_dict(model, weight_file)

    sampling_distribution = eval(args.sampling_distribution)
    print('Sampling Distribution: ', sampling_distribution)
    trainer.train(model, log_dir, checkpoint_dir,
                  loss_type = args.loss_type,
                  num_indices = args.num_indices,
                  image_chunk_size = args.image_chunk_size,
                  per_instance = args.per_instance,
                  sample_by_loss = args.sample_by_loss,
                  top_n = args.top_n,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  checkpoint_every_nth = args.checkpoint_every_nth,
                  sampling_distribution = sampling_distribution,
                  start_idx = max_step,
                  num_steps = args.num_steps)

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
        e = sys.exc_info()[0]
        sca.sendAlert('bokorn@andrew.cmu.edu', 
                message_subject='Job Failed on {}'.format(hostname),
                message_text=e)
        raise
