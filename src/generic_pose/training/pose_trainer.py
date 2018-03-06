# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from logger import Logger

import torch
import numpy as np

from torch.autograd import Variable
from torch.optim import Adam, Adadelta, SGD
from torch.utils.data import DataLoader

import os

from generic_pose.datasets.image_dataset import PoseImageDataSet
from generic_pose.datasets.image_pair_dataset import PoseImagePairsDataSet

from generic_pose.losses.viewpoint_loss import ViewpointLoss, denseViewpointError
from generic_pose.losses.quaternion_loss import quaternionLoss, quaternionError
from generic_pose.utils.display_pose import makeDisplayImages, renderTopRotations, renderQuaternions
from generic_pose.training.utils import to_np, to_var, evaluatePairReg, evaluatePairCls, evaluateLoopReg
    
class PoseTrainer(object):
    def __init__(self, 
                 train_data_folders,
                 valid_data_folders,
                 img_size = (227,227),
                 batch_size = 32,
                 num_workers = 4,
                 model_filenames = None,
                 background_filenames = None,
                 classification = True,
                 locked_pairs = False,
                 num_bins = (50,50,25),
                 distance_sigma = 0.1,
                 render_distance = 2,
                 seed = 0):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.num_bins = num_bins
        self.img_size = img_size
        self.class_loss = ViewpointLoss()
        self.render_distance = render_distance
        self.classification = classification
        if(locked_pairs):
            DataSet = PoseImagePairsDataSet
        else:
            DataSet = PoseImageDataSet
            
        self.train_loader = DataLoader(DataSet(data_folders=train_data_folders,
                                               img_size = img_size,
                                               model_filenames=model_filenames,
                                               background_filenames = background_filenames,
                                               classification = self.classification,
                                               num_bins=self.num_bins,
                                               distance_sigma=distance_sigma),
                                       num_workers=batch_size, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    
        self.valid_loader = DataLoader(DataSet(data_folders=valid_data_folders,
                                               img_size = img_size,
                                               model_filenames=model_filenames,
                                               background_filenames = background_filenames,
                                               classification = self.classification,
                                               num_bins=self.num_bins,
                                               distance_sigma=distance_sigma),
                                       num_workers=num_workers, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    
    def train(self, model, results_dir, 
              num_epochs = 100000,
              log_every_nth = 10,
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
        
        valid_log_dir = os.path.join(log_dir,'valid')
        if not os.path.exists(valid_log_dir):
            os.makedirs(valid_log_dir)  
            
        weights_dir = os.path.join(results_dir,'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            
        train_logger = Logger(train_log_dir)
        valid_logger = Logger(valid_log_dir)
    
        cumulative_batch_idx = 0
        min_loss_quat = float('inf')
        min_loss_binned = float('inf')

        print('Starting Training')
        
        #epoch_digits = len(str(num_epochs+1))
        #batch_digits = 8#len(str(len(self.train_loader)))        

        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (origin, query, quat_true, class_true, origin_quat, model_file) in enumerate(self.train_loader):
                log_data = not((cumulative_batch_idx+1) % log_every_nth)
                if(self.classification):
                    train_results = evaluatePairCls(model, origin, query, class_true,
                                                    num_bins = self.num_bins,
                                                    optimizer = self.optimizer, 
                                                    disp_metrics = log_data)
                else:
                    train_results = evaluatePairReg(model, origin, query, quat_true,
                                                    optimizer = self.optimizer, 
                                                    disp_metrics = log_data)
                                                    
                if log_data:
                    v_origin, v_query, v_quat_true, v_class_true, v_origin_quat, v_model_file = next(iter(self.valid_loader))
                    
                    if(self.classification):
                        valid_results = evaluatePairCls(model, v_origin, v_query, v_class_true,
                                                        num_bins = self.num_bins,
                                                        optimizer = None, 
                                                        disp_metrics = True)
                    else:
                        valid_results = evaluatePairReg(model, v_origin, v_query, v_quat_true,
                                                        optimizer = None, 
                                                        disp_metrics = True)
                    
                    print("epoch {} :: cumulative_batch_idx {}".format(epoch_idx, cumulative_batch_idx + 1))
                    
                    train_info = {}
                    
                    for k,v in train_results.items():
                        if('est' not in k):
                            train_info[k] = v

                    valid_info = {}

                    for k,v in valid_results.items():
                        if('est' not in k):
                            valid_info[k] = v
                            
                    for tag, value in train_info.items():
                        train_logger.scalar_summary(tag, value, cumulative_batch_idx+1)
                    
                    for tag, value in valid_info.items():
                        valid_logger.scalar_summary(tag, value, cumulative_batch_idx+1)
                    
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        train_logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                        train_logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)
            
            
                    train_origin_imgs = to_np(origin.view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                    train_query_imgs = to_np(query.view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                    train_quat_true = to_np(quat_true[:num_display_imgs])
                    
                    if('quat_est' in train_results):
                        train_quat_est = to_np(train_results['quat_est'][:num_display_imgs])
                        train_render_imgs = renderQuaternions(train_origin_imgs, train_query_imgs, 
                                                              train_quat_true, train_quat_est,
                                                              origin_quats = origin_quat[:num_display_imgs],
                                                              model_files=model_file[:num_display_imgs],
                                                              camera_dist = self.render_distance)
                    else:
                        train_quat_est = None
                    
                    if('class_est' in train_results):
                        train_class_true = to_np(class_true[:num_display_imgs])
                        train_class_est = to_np(train_results['class_est'][:num_display_imgs])
                        train_render_imgs = renderTopRotations(train_class_true, train_class_est, 
                                                               self.num_bins, 
                                                               origin_quats = origin_quat[:num_display_imgs],
                                                               model_files=model_file[:num_display_imgs],
                                                               camera_dist = self.render_distance)

                    else:
                        train_class_true = None
                        train_class_est = None

                    train_info = {'renders':train_render_imgs}
                                      
                    train_disp_imgs = makeDisplayImages(train_origin_imgs, train_query_imgs,
                                                        train_class_true, train_quat_true,
                                                        train_class_est, train_quat_est, 
                                                        num_bins = self.num_bins)

                    
                    valid_origin_imgs = to_np(v_origin.view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                    valid_query_imgs  = to_np(v_query.view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                    valid_quat_true   = to_np(v_quat_true[:num_display_imgs])
                    
                    if('quat_est' in valid_results):
                        valid_quat_est = to_np(valid_results['quat_est'][:num_display_imgs])
                        valid_render_imgs = renderQuaternions(valid_origin_imgs, valid_query_imgs, 
                                                              valid_quat_true, valid_quat_est,
                                                              origin_quats = v_origin_quat[:num_display_imgs],
                                                              model_files = v_model_file[:num_display_imgs],
                                                              camera_dist = self.render_distance)
                    else:
                        valid_quat_est = None
                    
                    if('class_est' in valid_results):
                        valid_class_true = to_np(v_class_true[:num_display_imgs])
                        valid_class_est = to_np(valid_results['class_est'][:num_display_imgs])
                        valid_render_imgs = renderTopRotations(valid_class_true, valid_class_est, 
                                                               self.num_bins, 
                                                               origin_quats = v_origin_quat[:num_display_imgs],
                                                               model_files=v_model_file[:num_display_imgs],
                                                               camera_dist = self.render_distance)
                    else:
                        valid_class_true = None
                        valid_class_est = None

                    valid_info = {'renders':valid_render_imgs}
                    
                    valid_disp_imgs = makeDisplayImages(valid_origin_imgs, valid_query_imgs,
                                                        valid_class_true, valid_quat_true,
                                                        valid_class_est, valid_quat_est, 
                                                        num_bins = self.num_bins)
            
                    train_info['display'] = train_disp_imgs
            
                    for tag, images in train_info.items():
                        train_logger.image_summary(tag, images, cumulative_batch_idx+1)
                    
                    valid_info['display'] = valid_disp_imgs
            
                    for tag, images in valid_info.items():
                        valid_logger.image_summary(tag, images, cumulative_batch_idx+1)
                    
                    self.optimizer.zero_grad()

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
                    if('loss_binned' in valid_results and valid_results['loss_binned'] < min_loss_binned):
                        min_loss_binned = min(min_loss_binned, valid_results.setdefault('loss_binned', -float('inf')))
                                                                           
                        weights_filename = os.path.join(weights_dir, 'best_binned.pth')
                        print("saving model ", weights_filename)
                        torch.save(model.state_dict(), weights_filename)
                        

                    weights_filename = os.path.join(weights_dir, 'latest.pth')
                    torch.save(model.state_dict(), weights_filename)
                cumulative_batch_idx += 1

def main():
    import datetime
    from argparse import ArgumentParser
    from generic_pose.models.pose_networks import gen_pose_net
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_folders', type=str, default=None)
    parser.add_argument('--valid_data_folders', type=str, default=None)
    
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--model_data_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--locked_pairs', dest='locked_pairs', action='store_true')
    parser.add_argument('--single_model', dest='single_model', action='store_true')
    parser.add_argument('--render_distance', type=float, default=2.0)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--distance_sigma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='vgg16')
    parser.add_argument('--compare_type', type=str, default='basic')
    parser.add_argument('--output_type', type=str, default='regression')

    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=50)

    args = parser.parse_args()

    assert args.output_type.lower() in ['regression', 'classification'], \
        'Invalid output_type {}, Must be regression or classification'.format(args.output_type.lower())

    classification = args.output_type.lower() == 'classification'

    if(args.train_data_folders[-4:] == '.txt'):
        with open(args.train_data_folders, 'r') as f:    
            train_data_folders = f.read().split()
    else:
        train_data_folders = args.train_data_folders

    if(args.valid_data_folders[-4:] == '.txt'):
        with open(args.valid_data_folders, 'r') as f:    
            valid_data_folders = f.read().split()
    else:
        valid_data_folders = args.valid_data_folders
    
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

    num_bins = (50,50,25)
    trainer = PoseTrainer(train_data_folders = train_data_folders,
                          valid_data_folders = valid_data_folders,
                          img_size = (args.width,args.height),
                          batch_size = args.batch_size,
                          num_workers = args.num_workers,
                          model_filenames = model_filenames,
                          background_filenames = background_filenames,
                          locked_pairs = args.locked_pairs,
                          classification = classification, 
                          num_bins = num_bins,
                          distance_sigma = args.distance_sigma, 
                          seed = args.random_seed,
                          render_distance = render_distance)


    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    if(classification):
        output_dim = num_bins
    else:
        output_dim = 4
    
    model = gen_pose_net(args.model_type.lower(), 
                         args.compare_type.lower(), 
                         output_dim = output_dim,
                         pretrained = args.pretrained)

    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    trainer.train(model, results_dir, 
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer=args.optimizer)

if __name__=='__main__':
    main()
