# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from logger import Logger

import torch
import numpy as np

from torch.optim import Adam, Adadelta, SGD
from torch.utils.data import DataLoader

import os

from generic_pose.datasets.image_dataset import PoseImageDataSet

from generic_pose.losses.viewpoint_loss import ViewpointLoss
from generic_pose.utils.display_pose import makeDisplayImages
from generic_pose.training.utils import to_np
from generic_pose.training.step_utils import getAxes, evaluateStepClass
    
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    
    
class StepTrainer(object):
    def __init__(self, 
                 train_data_folder,
                 valid_class_folder = None,
                 valid_model_folder = None,
                 valid_pose_folder = None,
                 img_size = (227,227),
                 max_orientation_offset = None,
                 max_orientation_iters = 200,
                 batch_size = 32,
                 num_workers = 4,
                 model_filenames = None,
                 background_filenames = None,
                 render_distance = 2,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.class_loss = ViewpointLoss()
        self.render_distance = render_distance
        
        self.train_loader = DataLoader(PoseImageDataSet(data_folders=train_data_folder,
                                               img_size = img_size,
                                               max_orientation_offset = max_orientation_offset,
                                               max_orientation_iters = max_orientation_iters,
                                               model_filenames=model_filenames,
                                               background_filenames = background_filenames,
                                               classification=False,
                                               num_bins=(1,1,1),
                                               distance_sigma=1),
                                       num_workers=num_workers, 
                                       batch_size=batch_size, 
                                       shuffle=True)
        self.train_loader.dataset.loop_truth = [1,1]
        
        self.valid_types = []
        self.valid_loaders = []
        
        if(valid_class_folder is not None):
            self.valid_class_loader = DataLoader(PoseImageDataSet(data_folders=valid_class_folder,
                                                   img_size = img_size,
                                                   max_orientation_offset = max_orientation_offset,
                                                   max_orientation_iters = max_orientation_iters,
                                                   model_filenames=model_filenames,
                                                   background_filenames = background_filenames,
                                                   classification=False,
                                                   num_bins=(1,1,1),
                                                   distance_sigma=1),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_class_loader.dataset.loop_truth = [1,1]
            self.valid_types.append('valid_class')
            self.valid_loaders.append(self.valid_class_loader)

        if(valid_model_folder is not None):
            self.valid_model_loader = DataLoader(PoseImageDataSet(data_folders=valid_model_folder,
                                                   img_size = img_size,
                                                   max_orientation_offset = max_orientation_offset,
                                                   max_orientation_iters = max_orientation_iters,
                                                   model_filenames=model_filenames,
                                                   background_filenames = background_filenames,
                                                   classification=False,
                                                   num_bins=(1,1,1),
                                                   distance_sigma=1),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_model_loader.dataset.loop_truth = [1,1]
            self.valid_types.append('valid_model')
            self.valid_loaders.append(self.valid_model_loader)
        
        if(valid_pose_folder is not None):
            self.valid_pose_loader = DataLoader(PoseImageDataSet(data_folders=valid_pose_folder,
                                                   img_size = img_size,
                                                   max_orientation_offset = max_orientation_offset,
                                                   max_orientation_iters = max_orientation_iters,
                                                   model_filenames=model_filenames,
                                                   background_filenames = background_filenames,
                                                   classification=False,
                                                   num_bins=(1,1,1),
                                                   distance_sigma=1),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_pose_loader.dataset.loop_truth = [1,1]
            self.valid_types.append('valid_pose')
            self.valid_loaders.append(self.valid_pose_loader)

    def train(self, model, results_dir, num_bins,
              step_angle = np.pi/4.0,
              use_softmax_labels=True, 
              loss_type = 'dot',
              num_epochs = 100000,
              log_every_nth = 10,
              checkpoint_every_nth = 1000,
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
        min_loss_quat = float('inf')
        
        print('Starting Training')
        
        bin_axes = getAxes(num_bins-1)

        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (images, trans, quats, models, model_files) in enumerate(self.train_loader):
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                train_results = evaluateStepClass(model, 
                                                  images[0], images[1], 
                                                  quats[0], quats[1],
                                                  bin_axes = bin_axes,
                                                  step_angle = step_angle,
                                                  use_softmax_labels=use_softmax_labels,
                                                  loss_type=loss_type,
                                                  optimizer = self.optimizer, 
                                                  disp_metrics = log_data)

                if log_data:
                    print("epoch {} :: cumulative_batch_idx {}".format(epoch_idx, cumulative_batch_idx + 1))
                    print('train deltas')
                    print(np.round(to_np(train_results['delta_vec'][0])*180/np.pi))
                    print('train labels')
                    print(np.round(to_np(train_results['label_vec'][0]),2))
                    print('train results')
                    print(np.round(to_np(train_results['class_vec'][0]),2))
                    train_info = {}
                    for k,v in train_results.items():
                        if('vec' not in k):
                            train_info[k] = v
                    
                    for tag, value in train_info.items():
                        train_logger.scalar_summary(tag, value, cumulative_batch_idx+1)

                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        train_logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                        train_logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)

                    #########################################
                    ############ VALIDATION SETS ############
                    #########################################
                    for valid_logger, valid_loader in zip(valid_logger_list, self.valid_loaders):
                        v_images, v_trans, v_quats, v_models, v_model_files = next(iter(valid_loader))
                        
                        valid_results = evaluateStepClass(model, 
                                                          v_images[0], v_images[1], 
                                                          v_quats[0], v_quats[1],
                                                          bin_axes = bin_axes,
                                                          step_angle = step_angle,
                                                          use_softmax_labels=use_softmax_labels,
                                                          loss_type=loss_type,
                                                          optimizer = self.optimizer, 
                                                          disp_metrics = log_data)

#                        print('valid deltas')
#                        print(np.round(to_np(valid_results['delta_vec'][0])*180/np.pi))
#                        print('valid labels')
#                        print(np.round(to_np(valid_results['label_vec'][0]),2))
#                        print('valid results')
#                        print(np.round(to_np(valid_results['class_vec'][0]),2))

                        valid_info = {}
                        for k,v in valid_results.items():
                            if('vec' not in k):
                                valid_info[k] = v
                        
                        for tag, value in valid_info.items():
                            valid_logger.scalar_summary(tag, value, cumulative_batch_idx+1)
    
                    if('loss_quat' in valid_results and valid_results['loss_quat'] < min_loss_quat):
                        min_loss_quat = min(min_loss_quat, valid_results.setdefault('loss_quat', -float('inf')))
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
    from generic_pose.models.pose_networks import gen_pose_net
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_folder', type=str, default=None)
    parser.add_argument('--valid_class_folder', type=str, default=None)
    parser.add_argument('--valid_model_folder', type=str, default=None)
    parser.add_argument('--valid_pose_folder', type=str, default=None)
    
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--model_data_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--single_model', dest='single_model', action='store_true')
    parser.add_argument('--render_distance', type=float, default=2.0)

    parser.add_argument('--num_bins', type=int, default=15)
    parser.add_argument('--step_angle', type=float, default=np.pi/4.0)
    parser.add_argument('--loss_type', type=str, default='dot')
    parser.add_argument('--softmax_labels', dest='use_softmax_labels', action='store_true')

    parser.add_argument('--max_orientation_offset', type=float, default=None)
    parser.add_argument('--max_orientation_iters', type=int, default=200)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loop_length', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--distance_sigma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='vgg16')
    parser.add_argument('--compare_type', type=str, default='basic')

    parser.add_argument('--random_init', dest='pretrained', action='store_false')    

    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=50)
    parser.add_argument('--checkpoint_every_nth', type=int, default=1000)
    parser.add_argument('--num_display_imgs', type=int, default=0)

    args = parser.parse_args()
    
    render_distance = args.render_distance
    if(args.model_data_file is not None):
        if(args.single_model):
            model_filenames = args.model_data_file
            
        else:
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
        
    trainer = StepTrainer(train_data_folder = args.train_data_folder,
                          valid_class_folder = args.valid_class_folder,
                          valid_model_folder = args.valid_model_folder,
                          valid_pose_folder = args.valid_pose_folder,
                          img_size = (args.width,args.height),
                          max_orientation_offset = args.max_orientation_offset,
                          max_orientation_iters = args.max_orientation_iters,
                          batch_size = args.batch_size,
                          num_workers = args.num_workers,
                          model_filenames = model_filenames,
                          background_filenames = background_filenames,
                          seed = args.random_seed,
                          render_distance = render_distance,
                          )

    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    model = gen_pose_net(args.model_type.lower(), 
                         args.compare_type.lower(), 
                         output_dim = args.num_bins,
                         pretrained = args.pretrained)

    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    trainer.train(model, results_dir,
                  num_bins = args.num_bins,
                  step_angle = args.step_angle,
                  use_softmax_labels = args.use_softmax_labels,
                  loss_type=args.loss_type,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  num_display_imgs = args.num_display_imgs,
                  checkpoint_every_nth = args.checkpoint_every_nth)

if __name__=='__main__':
    main()
