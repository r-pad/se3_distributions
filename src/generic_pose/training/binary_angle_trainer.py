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

from generic_pose.datasets.numpy_dataset import NumpyImageDataset

from generic_pose.training.utils import to_np
from generic_pose.training.binary_angle_utils import evaluateBinaryEstimate

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    
    
class BinaryAngleTrainer(object):
    def __init__(self, 
                 train_data_folder,
                 valid_class_folder = None,
                 valid_model_folder = None,
                 valid_pose_folder = None,
                 img_size = (227,227),
                 target_angle = None,
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
        self.target_angle = target_angle
        self.cross_model_eval = cross_model_eval

        self.train_loader = DataLoader(NumpyImageDataset(data_folders=train_data_folder,
                                               img_size = img_size,
                                               max_orientation_offset = None,
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
                                                   max_orientation_offset = None,
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
            self.valid_loaders.append(self.valid_class_loader)

        if(valid_model_folder is not None):
            self.valid_model_loader = DataLoader(NumpyImageDataset(data_folders=valid_model_folder,
                                                   img_size = img_size,
                                                   max_orientation_offset = None,
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
            self.valid_loaders.append(self.valid_model_loader)
        
        if(valid_pose_folder is not None):
            self.valid_pose_loader = DataLoader(NumpyImageDataset(data_folders=valid_pose_folder,
                                                   img_size = img_size,
                                                   max_orientation_offset = None,
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
            self.valid_loaders.append(self.valid_pose_loader)

    def train(self, model, results_dir,
              loss_type,
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

        dataset_size = len(self.train_loader)
        for epoch_idx in range(1, num_epochs+1):
            for batch_idx in range(dataset_size):
                self.train_loader.dataset.max_orientation_offset = None
                tf_images, tf_trans, _, _, _ = next(iter(self.train_loader))
                self.train_loader.dataset.max_orientation_offset = self.target_angle * 0.9
                tc_images, tc_trans, _, _, _ = next(iter(self.train_loader))
                
                images = [torch.cat((f_val, c_val), 0) for f_val, c_val in zip(tf_images, tc_images)]
                trans = [torch.cat((f_val, c_val), 0) for f_val, c_val in zip(tf_trans, tc_trans)]
                    
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                train_results = evaluateBinaryEstimate(model, images[0], images[1], trans[0], 
                                                       target_angle = self.target_angle,
                                                       loss_type=loss_type,
                                                       optimizer = self.optimizer, 
                                                       disp_metrics = log_data)

                if log_data:
                    print("epoch {} :: cumulative_batch_idx {}".format(epoch_idx, cumulative_batch_idx + 1))
                    
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
                                
                    if('angle_vec' in train_results.keys()):
                        train_logger.histo_summary('angle_vec',train_results['angle_vec'], cumulative_batch_idx+1)

                    self.optimizer.zero_grad()

                    #########################################
                    ############ VALIDATION SETS ############
                    #########################################
                    for valid_logger, valid_loader in zip(valid_logger_list, self.valid_loaders):
                        valid_loader.dataset.max_orientation_offset = None
                        vf_images, vf_trans, _, _, _ = next(iter(valid_loader))
                        valid_loader.dataset.max_orientation_offset = self.target_angle * 0.9
                        vc_images, vc_trans, _, _, _ = next(iter(valid_loader))
                        v_images = [torch.cat((f_val, c_val), 0) for f_val, c_val in zip(vf_images, vc_images)]
                        v_trans = [torch.cat((f_val, c_val), 0) for f_val, c_val in zip(vf_trans, vc_trans)]
                        
                        valid_results = evaluateBinaryEstimate(model, v_images[0], v_images[1], v_trans[0],
                                                               target_angle = self.target_angle,
                                                               loss_type=loss_type,
                                                               optimizer = None, disp_metrics = True)

                        valid_info = {}
                        for k,v in valid_results.items():
                            if('vec' not in k):
                                valid_info[k] = v
                        
                        for tag, value in valid_info.items():
                            valid_logger.scalar_summary(tag, value, cumulative_batch_idx+1)
    
                        if('angle_vec' in valid_results.keys()):
                            valid_logger.histo_summary('angle_vec',valid_results['angle_vec'], cumulative_batch_idx+1)
                        
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

    parser.add_argument('--loss_type', type=str, default='BCE')

    parser.add_argument('--target_angle', type=float, default=45.0)
    parser.add_argument('--max_orientation_iters', type=int, default=200)

    parser.add_argument('--cross_model_eval', dest='cross_model_eval', action='store_true')    

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='vgg16')
    parser.add_argument('--compare_type', type=str, default='sigmoid')
 
    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=50)
    parser.add_argument('--checkpoint_every_nth', type=int, default=1000)
    parser.add_argument('--num_display_imgs', type=int, default=0)

    args = parser.parse_args()

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
        
    trainer = BinaryAngleTrainer(train_data_folder = args.train_data_folder,
                                 valid_class_folder = args.valid_class_folder,
                                 valid_model_folder = args.valid_model_folder,
                                 valid_pose_folder = args.valid_pose_folder,
                                 img_size = (args.width,args.height),
                                 target_angle = args.target_angle*np.pi/180.0,
                                 max_orientation_iters = args.max_orientation_iters,
                                 batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 model_filenames = model_filenames,
                                 background_filenames = background_filenames,
                                 seed = args.random_seed,
                                 cross_model_eval = args.cross_model_eval,
                                 )


    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    model = gen_pose_net(args.model_type.lower(), 
                         args.compare_type.lower(), 
                         output_dim = 1,
                         pretrained = args.pretrained)

    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    trainer.train(model, results_dir,
                  loss_type=args.loss_type,
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  num_display_imgs = args.num_display_imgs,
                  checkpoint_every_nth = args.checkpoint_every_nth)

if __name__=='__main__':
    main()
