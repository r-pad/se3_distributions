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
from generic_pose.utils.display_pose import makeDisplayImages, renderQuaternions, makeHistogramImages
from generic_pose.training.utils import to_np, evaluateLoopReg
from generic_pose.training.max_angle_utils import evaluateMaxedMultQuat

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    
    
class MultiTrainer(object):
    def __init__(self, 
                 train_data_folder,
                 valid_class_folder = None,
                 valid_model_folder = None,
                 valid_pose_folder = None,
                 img_size = (227,227),
                 max_orientation_offset = None,
                 max_orientation_iters = 200,
                 batch_size = 32,
                 loop_length = 2,
                 num_workers = 4,
                 model_filenames = None,
                 background_filenames = None,
                 distance_sigma = 0.1,
                 render_distance = 2,
                 cross_model_eval = True,
                 loop_training = False,
                 curriculum_threshold = 20.0*np.pi/180,
                 angular_curriculum_limits = None,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.class_loss = ViewpointLoss()
        self.render_distance = render_distance
        self.loop_training = loop_training
        self.cross_model_eval = cross_model_eval
        
        self.angular_curriculum_limits = angular_curriculum_limits
        self.curriculum_threshold = curriculum_threshold        
        if(angular_curriculum_limits is not None):
            max_orientation_offset = angular_curriculum_limits[0]
            
        self.train_loader = DataLoader(PoseImageDataSet(data_folders=train_data_folder,
                                               img_size = img_size,
                                               max_orientation_offset = max_orientation_offset,
                                               max_orientation_iters = max_orientation_iters,
                                               model_filenames=model_filenames,
                                               background_filenames = background_filenames,
                                               classification=False,
                                               num_bins=(1,1,1),
                                               distance_sigma=distance_sigma),
                                       num_workers=num_workers, 
                                       batch_size=batch_size, 
                                       shuffle=True)
        self.train_loader.dataset.loop_truth = [1,] + [0 for _ in range(loop_length-1)]
        
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
                                                   distance_sigma=distance_sigma),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_class_loader.dataset.loop_truth = [1,] + [0 for _ in range(loop_length-1)]
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
                                                   distance_sigma=distance_sigma),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_model_loader.dataset.loop_truth = [1,] + [0 for _ in range(loop_length-1)]
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
                                                   distance_sigma=distance_sigma),
                                           num_workers=int(num_workers/2), 
                                           batch_size=int(batch_size/2), 
                                           shuffle=True)
            self.valid_pose_loader.dataset.loop_truth = [1,] + [0 for _ in range(loop_length-1)]
            self.valid_types.append('valid_pose')
            self.valid_loaders.append(self.valid_pose_loader)
        
    def setAngleLimit(self, angle_limit):
        self.train_loader.dataset.max_orientation_offset = angle_limit
        for loader in self.valid_loaders:
            loader.dataset.max_orientation_offset = angle_limit

    def train(self, model, results_dir,
              loop_truth,
              num_epochs = 100000,
              log_every_nth = 10,
              checkpoint_every_nth = 1000,
              lr = 1e-5,
              optimizer = 'SGD',
              num_display_imgs=1,
              clip=None):
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
        curriculum_idx = 0
        min_loss_quat = float('inf')
        
        if(self.angular_curriculum_limits is not None):
            train_logger.scalar_summary('max_orientation_offset', self.angular_curriculum_limits[curriculum_idx]*180/np.pi, cumulative_batch_idx+1)

        print('Starting Training')
        
        #epoch_digits = len(str(num_epochs+1))
        #batch_digits = 8#len(str(len(self.train_loader)))        
        train_errors = []
        train_angles = []
#        train_loop_errors = []

        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (images, trans, quats, models, model_files) in enumerate(self.train_loader):
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                train_results = evaluateMaxedMultQuat(model, images[0], images[1], 
                                                  quats[0], quats[1],
                                                  optimizer = self.optimizer, 
                                                  disp_metrics = True)

                if(self.cross_model_eval):
                    train_cross = evaluateMaxedMultQuat(model, images[1], images[2], 
                                                  quats[0], quats[1],
                                                  optimizer = None, disp_metrics = True)

                if(self.loop_training):
                    train_loop = evaluateLoopReg(model, images, trans, loop_truth,
                                                 optimizer = self.optimizer, 
                                                 disp_metrics = True,
                                                 clip = clip)

                if('errs_vec' in train_results.keys()):
                    train_errors.append(train_results['errs_vec'])
                    train_angles.append(train_results['diff_vec'])

#                if('errs_vec' in train_loop.keys()):
#                    train_loop_errors.append(train_loop['errs_vec'])
                    
                if log_data:
                    print("epoch {} :: cumulative_batch_idx {}".format(epoch_idx, cumulative_batch_idx + 1))
                    
                    train_info = {}
                    for k,v in train_results.items():
                        if('vec' not in k):
                            train_info[k] = v
                    
                    if(self.cross_model_eval):
                        for k,v in train_cross.items():
                            if('vec' not in k):
                                train_info['cross_' + k] = v

                    if(self.loop_training):
                        for k,v in train_loop.items():
                            if('vec' not in k):
                                train_info[k] = v

                    for tag, value in train_info.items():
                        train_logger.scalar_summary(tag, value, cumulative_batch_idx+1)

                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        train_logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                        train_logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)
                                
                    if(len(train_errors) > 0):
                        train_logger.histo_summary('errs_vec',np.concatenate(train_errors), cumulative_batch_idx+1)
#                    if(len(train_loop_errors) > 0):
#                        train_logger.histo_summary('errs_vec',np.concatenate(train_loop_errors), cumulative_batch_idx+1)
#                        train_loop_errors = []                                        

                    if(num_display_imgs > 0):
    
                        train_origin_imgs = to_np(images[0].view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                        train_query_imgs  = to_np(images[1].view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                        train_quat_true   = to_np(trans[0][:num_display_imgs])                    
                        train_quat_est = to_np(train_results['quat_vec'][:num_display_imgs])
    
    
                        train_render_imgs = renderQuaternions(train_origin_imgs, train_query_imgs, 
                                                              train_quat_true, train_quat_est,
                                                              origin_quats = quats[0][:num_display_imgs],
                                                              model_files=model_files[0][:num_display_imgs],
                                                              camera_dist = self.render_distance)
    
                        train_info = {'renders':train_render_imgs}
                        if(self.cross_model_eval):
                            train_cross_imgs = to_np(images[2].view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                            train_cross_true  = to_np(trans[1][:num_display_imgs])
                            train_cross_est   = to_np(train_cross['quat_vec'][:num_display_imgs])
                        
                            train_render_cross_imgs = renderQuaternions(train_query_imgs, train_cross_imgs,
                                                                  train_cross_true, train_cross_est,
                                                                  origin_quats = quats[1][:num_display_imgs],
                                                                  model_files=model_files[1][:num_display_imgs],
                                                                  camera_dist = self.render_distance)
        
                            train_info['cross'] = train_render_cross_imgs
                                              
                        train_disp_imgs = makeDisplayImages(train_origin_imgs, train_query_imgs,
                                                            None, train_quat_true,
                                                            None, train_quat_est, 
                                                            num_bins = (1,1,1))
                        train_info['display'] = train_disp_imgs
    
                        if(len(train_errors) > 0):
                            train_mean_hist, train_error_hist, train_count_hist = makeHistogramImages(np.concatenate(train_errors), np.concatenate(train_angles))
                            train_info['mean_hist'] = train_mean_hist
                            train_info['error_hist'] = train_error_hist
                            train_info['count_hist'] = train_count_hist
                            train_errors = []
                            train_angles = []                                
    
                        for tag, images in train_info.items():
                            train_logger.image_summary(tag, images, cumulative_batch_idx+1)

                    self.optimizer.zero_grad()

                    #########################################
                    ############ VALIDATION SETS ############
                    #########################################
                    for valid_logger, valid_loader in zip(valid_logger_list, self.valid_loaders):
                        v_images, v_trans, v_quats, v_models, v_model_files = next(iter(valid_loader))
                        
                        valid_results = evaluateMaxedMultQuat(model, v_images[0], v_images[1], 
                                                          v_quats[0], v_quats[1],
                                                          optimizer = None, disp_metrics = True)

                        if(self.cross_model_eval):
                            valid_cross = evaluateMaxedMultQuat(model, v_images[1], v_images[2], 
                                                            v_quats[0], v_quats[1],
                                                            optimizer = None, disp_metrics = True)
                        if(self.loop_training):
                            valid_loop = evaluateLoopReg(model, v_images, v_trans, loop_truth,
                                                         optimizer = None, disp_metrics = True)

                        valid_info = {}
                        for k,v in valid_results.items():
                            if('vec' not in k):
                                valid_info[k] = v
                        if(self.cross_model_eval):    
                            for k,v in valid_cross.items():
                                if('vec' not in k):
                                    valid_info['cross_' + k] = v
                        if(self.loop_training):   
                            for k,v in valid_loop.items():
                                if('vec' not in k):
                                    valid_info[k] = v    
                        for tag, value in valid_info.items():
                            valid_logger.scalar_summary(tag, value, cumulative_batch_idx+1)
    
                        if('errs_vec' in valid_results.keys()):
                            valid_logger.histo_summary('errs_vec',valid_results['errs_vec'], cumulative_batch_idx+1)
    #                    if('errs_vec' in valid_loop.keys()):
    #                        valid_logger.histo_summary('errs_vec',valid_loop['errs_vec'], cumulative_batch_idx+1)
                        if(num_display_imgs > 0):
                            valid_origin_imgs = to_np(v_images[0].view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                            valid_query_imgs  = to_np(v_images[1].view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                            valid_quat_true   = to_np(v_trans[0][:num_display_imgs])
                            valid_quat_est    = to_np(valid_results['quat_vec'][:num_display_imgs])
                            
                            valid_render_imgs = renderQuaternions(valid_origin_imgs, valid_query_imgs, 
                                                                  valid_quat_true, valid_quat_est,
                                                                  origin_quats = v_quats[0][:num_display_imgs],
                                                                  model_files = v_model_files[0][:num_display_imgs],
                                                                  camera_dist = self.render_distance)
        
                            valid_info = {'renders':valid_render_imgs}
                            
                            if(self.cross_model_eval):
                                valid_cross_imgs  = to_np(v_images[2].view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                                valid_cross_true  = to_np(v_trans[1][:num_display_imgs])
                                valid_cross_est   = to_np(valid_cross['quat_vec'][:num_display_imgs])
                                valid_render_cross_imgs = renderQuaternions(valid_query_imgs, valid_cross_imgs,
                                                                     valid_cross_true, valid_cross_est,
                                                                     origin_quats = v_quats[1][:num_display_imgs],
                                                                     model_files = v_model_files[1][:num_display_imgs],
                                                                     camera_dist = self.render_distance)                    
                                valid_info['cross'] = valid_render_cross_imgs
                            
                            valid_disp_imgs = makeDisplayImages(valid_origin_imgs, valid_query_imgs,
                                                                None, valid_quat_true,
                                                                None, valid_quat_est, 
                                                                num_bins = (1,1,1))
                    
                            valid_info['display'] = valid_disp_imgs
        
                            if('errs_vec' in valid_results.keys()):
                                valid_mean_hist, valid_error_hist, valid_count_hist = makeHistogramImages(valid_results['errs_vec'], valid_results['diff_vec'])
                                valid_info['mean_hist'] = valid_mean_hist
                                valid_info['error_hist'] = valid_error_hist
                                valid_info['count_hist'] = valid_count_hist
                    
                            for tag, images in valid_info.items():
                                valid_logger.image_summary(tag, images, cumulative_batch_idx+1)
                        
                    if('loss_quat' in valid_results and valid_results['loss_quat'] < min_loss_quat):
                        min_loss_quat = min(min_loss_quat, valid_results.setdefault('loss_quat', -float('inf')))

#                        
#                                              format(epoch_idx, cumulative_batch_idx + 1, 
#                                              valid_results['loss_quat'], 
#                                              valid_results['loss_binned'],
#                                              epoch_digits, batch_digits))
                        weights_filename = os.path.join(weights_dir, 'best_quat.pth')
                        print("saving model ", weights_filename)
                        torch.save(model.state_dict(), weights_filename)                        

                    weights_filename = os.path.join(weights_dir, 'latest.pth')
                    torch.save(model.state_dict(), weights_filename)
                    print(valid_results['mean_err']*np.pi/180, self.curriculum_threshold)
                    if(self.angular_curriculum_limits is not None and 
                       curriculum_idx < len(self.angular_curriculum_limits)-1 and
                       valid_results['mean_err']*np.pi/180 < self.curriculum_threshold):
                        curriculum_idx += 1
                        self.setAngleLimit(self.angular_curriculum_limits[curriculum_idx])
                        train_logger.scalar_summary('curriculum_idx', curriculum_idx, cumulative_batch_idx+1)
                        train_logger.scalar_summary('max_orientation_offset', self.angular_curriculum_limits[curriculum_idx]*180/np.pi, cumulative_batch_idx+1)
                        print('curriculum_idx: ', curriculum_idx)
                        curr_weights_filename = os.path.join(weights_dir, 'curr_{}.pth'.format(cumulative_batch_idx))                        
                        print("saving model ", curr_weights_filename)
                        torch.save(model.state_dict(), curr_weights_filename) 
                        cumulative_batch_idx += 1                        
                        break
                    
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

    parser.add_argument('--max_orientation_offset', type=float, default=None)
    parser.add_argument('--max_orientation_iters', type=int, default=200)
    parser.add_argument('--angular_curriculum_limits', nargs='+', default=None)
    parser.add_argument('--curriculum_threshold', type=float, default=np.pi/9.0)

    parser.add_argument('--loop_training', dest='loop_training', action='store_true')    
    parser.add_argument('--cross_model_eval', dest='cross_model_eval', action='store_true')    

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loop_length', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--distance_sigma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip_gradients', type=float, default=None)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='vgg16')
    parser.add_argument('--compare_type', type=str, default='basic')
    parser.add_argument('--output_type', type=str, default='regression')

    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=50)
    parser.add_argument('--checkpoint_every_nth', type=int, default=1000)
    parser.add_argument('--num_display_imgs', type=int, default=1)

    args = parser.parse_args()

    assert args.output_type.lower() in ['regression', 'classification'], \
        'Invalid output_type {}, Must be regression or classification'.format(args.output_type.lower())

    
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

    if(args.angular_curriculum_limits is not None):
        args.angular_curriculum_limits = np.array(args.angular_curriculum_limits, dtype=float)
        
    trainer = MultiTrainer(train_data_folder = args.train_data_folder,
                                       valid_class_folder = args.valid_class_folder,
                                       valid_model_folder = args.valid_model_folder,
                                       valid_pose_folder = args.valid_pose_folder,
                                       img_size = (args.width,args.height),
                                       max_orientation_offset = args.max_orientation_offset,
                                       max_orientation_iters = args.max_orientation_iters,
                                       batch_size = args.batch_size,
                                       loop_length = args.loop_length,
                                       num_workers = args.num_workers,
                                       model_filenames = model_filenames,
                                       background_filenames = background_filenames,
                                       distance_sigma = args.distance_sigma, 
                                       seed = args.random_seed,
                                       render_distance = render_distance,
                                       cross_model_eval = args.cross_model_eval,
                                       loop_training = args.loop_training,
                                       curriculum_threshold = args.curriculum_threshold,
                                       angular_curriculum_limits = args.angular_curriculum_limits,
                                       )


    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    model = gen_pose_net(args.model_type.lower(), 
                         args.compare_type.lower(), 
                         output_dim = 4,
                         pretrained = args.pretrained)

    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    if(args.loop_length == 2):
        loop_truth = [0,0]
    else:
        loop_truth = [1,] + [0 for _ in range(args.loop_length-1)]

    trainer.train(model, results_dir,
                  loop_truth = loop_truth, 
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer = args.optimizer,
                  clip = args.clip_gradients,
                  num_display_imgs = args.num_display_imgs,
                  checkpoint_every_nth = args.checkpoint_every_nth)

if __name__=='__main__':
    main()
