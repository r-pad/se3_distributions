# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from logger import Logger

import torch
import numpy as np

import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam, Adadelta
from torch.utils.data import DataLoader

import os

from renderer_dataset import PoseRendererDataSet

from viewpoint_loss import ViewpointLoss, viewpointAccuracy
from quaternion_loss import quaternionLoss
from display_pose import makeDisplayImage
    
def to_np(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    else:
        return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

class PoseTrainer(object):
    def __init__(self, 
                 train_filenames,
                 valid_filenames,
                 img_size = (227,227),
                 batch_size = 32,
                 num_workers = 4,
                 background_filenames = None,
                 max_orientation_offset = None,
                 prerender = True,
                 num_model_imgs = 250000,
                 train_data_folder = None,
                 valid_data_folder = None,
                 save_data = False,
                 seed = 0):
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.img_size = img_size
        self.class_loss = ViewpointLoss()
        
        self.train_loader = DataLoader(PoseRendererDataSet(model_filenames = train_filenames,
                                                           img_size = img_size,
                                                           background_filenames = background_filenames,
                                                           max_orientation_offset = max_orientation_offset,
                                                           prerender=prerender,
                                                           num_model_imgs=num_model_imgs,
                                                           data_folder=train_data_folder,
                                                           save_data=save_data),
                                       num_workers=batch_size, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    
        self.valid_loader = DataLoader(PoseRendererDataSet(model_filenames = valid_filenames,
                                                           img_size = img_size,
                                                           background_filenames = background_filenames,
                                                           max_orientation_offset = max_orientation_offset,
                                                           prerender=prerender,
                                                           num_model_imgs=num_model_imgs,
                                                           data_folder=valid_data_folder,
                                                           save_data=save_data),
                                       num_workers=num_workers, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    
    def evaluateModel(self, model, origin, query, 
                      quat_true, u0_true, u1_true, u2_true,
                      backward=True):

        origin = to_var(origin)
        query = to_var(query)
        quat_true = to_var(quat_true)
        u0_true = to_var(u0_true)
        u1_true = to_var(u1_true)
        u2_true = to_var(u2_true)

        results = {}

        if(backward):
            self.optimizer.zero_grad()
            
        if(model.features_regression is not None):
            origin_features_regression = model.featuresRegression(origin)
            query_features_regression = model.featuresRegression(query)
            quat_est = model.compareRegression(origin_features_regression, 
                                               query_features_regression)

            loss_quat = quaternionLoss(quat_est, quat_true)
            if(backward):
                loss_quat.backward(retain_graph=True)

            results['quat_est'] = quat_est
            results['loss_quat'] = loss_quat.data[0]
            results['mean_origin_features_regression'] = np.mean(np.abs(to_np(origin_features_regression)))
            results['mean_query_features_regression'] = np.mean(np.abs(to_np(query_features_regression)))

        if(model.features_classification is not None):
            origin_features_classification = model.featuresClassification(origin)
            query_features_classification = model.featuresClassification(query)
            u0_est, u1_est, u2_est = model.compareClassification(origin_features_classification,
                                                                 query_features_classification)
             
            loss_u0 = self.class_loss(u0_est, u0_true)
            loss_u1 = self.class_loss(u1_est, u1_true)
            loss_u2 = self.class_loss(u2_est, u2_true)
            loss_binned = loss_u0 + loss_u1 + loss_u2
        
            err_u0 = viewpointAccuracy(u0_est, u0_true)
            err_u1 = viewpointAccuracy(u1_est, u1_true)
            err_u2 = viewpointAccuracy(u2_est, u2_true)

            if(backward):
                loss_binned.backward()
        
            results['u0_est'] = u0_est
            results['u1_est'] = u1_est
            results['u2_est'] = u2_est
            results['loss_u0'] = loss_u0.data[0]
            results['loss_u1'] = loss_u1.data[0]
            results['loss_u2'] = loss_u2.data[0]
            results['loss_binned'] = loss_binned.data[0]
            results['err_u0'] = err_u0
            results['err_u1'] = err_u1
            results['err_u2'] = err_u2

            results['mean_origin_features_classification'] = np.mean(np.abs(to_np(origin_features_classification)))
            results['mean_query_features_classification'] = np.mean(np.abs(to_np(query_features_classification)))        
        
        if(backward):
            self.optimizer.step()

        return results
    
    def train(self, model, results_dir, 
              num_epochs = 100000,
              log_every_nth = 10,
              lr = 1e-5):
        model.train()
        model.cuda()
        self.optimizer = Adam(model.parameters(), lr=lr)
 

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        log_dir = os.path.join(results_dir,'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        weights_dir = os.path.join(results_dir,'weights')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            
        logger = Logger(log_dir)
    
        cumulative_batch_idx = 0
        min_loss_quat = float('inf')
        min_loss_binned = float('inf')

        print('Starting Training')
        
        epoch_digits = len(str(num_epochs+1))
        batch_digits = 8#len(str(len(self.train_loader)))        
        
        for epoch_idx in range(1, num_epochs+1):
            for batch_idx, (origin, query, quat_true, u0_true, u1_true, u2_true, u_bins) in enumerate(self.train_loader):
                
                train_results = self.evaluateModel(model, origin, query, 
                                                   quat_true, u0_true, u1_true, u2_true,
                                                   backward=True)
                      
                if not((cumulative_batch_idx+1) % log_every_nth):
                    v_origin, v_query, v_quat_true, v_u0_true, v_u1_true, v_u2_true, v_u_bins = next(iter(self.valid_loader))
                    
                    valid_results = self.evaluateModel(model, v_origin, v_query, 
                                                       v_quat_true, v_u0_true, v_u1_true, v_u2_true,
                                                       backward=False)
                    
                    print("epoch {} :: cumulative_batch_idx {}".format(epoch_idx, cumulative_batch_idx + 1))
                    
                    info = {}
                    
                    for k,v in train_results.items():
                        if('est' not in k):
                            info['train_' + k] = v

                    for k,v in valid_results.items():
                        if('est' not in k):
                            info['valid_' + k] = v
                            
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, cumulative_batch_idx+1)
            
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                        logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)
            
                    if('quat_est' in train_results):
                        train_quat_est = to_np(train_results['quat_est'][:10])
                    else:
                        train_quat_est = None
                    
                    if('u0_est' in train_results):
                        train_u0_est = to_np(train_results['u0_est'][:10])
                        train_u1_est = to_np(train_results['u1_est'][:10])
                        train_u2_est = to_np(train_results['u2_est'][:10])
                    else:
                        train_u0_est = None
                        train_u1_est = None
                        train_u2_est = None
                        
                    train_disp_imgs = makeDisplayImage(to_np(origin.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(query.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(u0_true[:10]), to_np(u1_true[:10]), 
                                                       to_np(u2_true[:10]), to_np(quat_true[:10]),
                                                       train_u0_est, train_u1_est, train_u2_est, train_quat_est)

                    if('quat_est' in valid_results):
                        valid_quat_est = to_np(valid_results['quat_est'][:10])
                    else:
                        valid_quat_est = None
                    
                    if('u0_est' in valid_results):
                        valid_u0_est = to_np(valid_results['u0_est'][:10])
                        valid_u1_est = to_np(valid_results['u1_est'][:10])
                        valid_u2_est = to_np(valid_results['u2_est'][:10])
                    else:
                        valid_u0_est = None
                        valid_u1_est = None
                        valid_u2_est = None
                    
                    valid_disp_imgs = makeDisplayImage(to_np(v_origin.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(v_query.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(v_u0_true[:10]), to_np(v_u1_true[:10]), 
                                                       to_np(v_u2_true[:10]), to_np(v_quat_true[:10]),
                                                       valid_u0_est, valid_u1_est, valid_u2_est, valid_quat_est)
            
                    info = {
                        'train': train_disp_imgs,
                        'valid': valid_disp_imgs,
                    }
            
                    for tag, images in info.items():
                        logger.image_summary(tag, images, cumulative_batch_idx+1)
                    
                    if(('loss_quat' in valid_results and valid_results['loss_quat'] < min_loss_quat) \
                        or ('loss_binned' in valid_results and valid_results['loss_binned'] < min_loss_binned)):

                        min_loss_quat = min(min_loss_quat, valid_results.setdefault('loss_quat', -float('inf')))
                        min_loss_binned = min(min_loss_binned, valid_results.setdefault('loss_binned', -float('inf')))

                        weights_filename = os.path.join(weights_dir, 'epoch_{0:0{4}d}_batch_{1:0{5}d}_lquat_{2:.4f}_lbinned_{3:.4f}.pth'.\
                                              format(epoch_idx, cumulative_batch_idx + 1, 
                                              valid_results['loss_quat'], 
                                              valid_results['loss_binned'],
                                              epoch_digits, batch_digits))
                                                                           
                        print("saving model ", weights_filename)
                        torch.save(model.state_dict(), weights_filename)
                        

                    weights_filename = os.path.join(weights_dir, 'latest.pth')
                    torch.save(model.state_dict(), weights_filename)
                cumulative_batch_idx += 1

def main():
    import datetime
    from argparse import ArgumentParser
    from gen_pose_net import gen_pose_net_alexnet, gen_pose_net_vgg16, gen_pose_net_resnet101, gen_pose_net_resnet50
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_file', type=str, default=None)
    parser.add_argument('--valid_data_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)
    parser.add_argument('--max_orientation_offset', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=227)
    parser.add_argument('--width', type=int, default=227)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    
    parser.add_argument('--model_type', type=str, default='resnet101')
    parser.add_argument('--train_classification', dest='classification', action='store_true')
    parser.add_argument('--train_regression', dest='regression', action='store_true')
    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--render_live',  dest='prerender', action='store_false')
    parser.add_argument('--num_model_imgs', type=int, default=25000)
    parser.add_argument('--train_data_folder', type=str, default=None)
    parser.add_argument('--valid_data_folder', type=str, default=None)
    parser.add_argument('--save_renders',  dest='save_data', action='store_true')
    parser.add_argument('--random_seed', type=int, default=0)


    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=10)

    args = parser.parse_args()

    if(args.train_data_file is not None):
        with open(args.train_data_file, 'r') as f:    
            train_filenames = f.read().split()
    else:
        train_filenames = None
    
    if(args.valid_data_file is not None):
        with open(args.valid_data_file, 'r') as f:    
            valid_filenames = f.read().split()    
    else:
        valid_filenames = None
        
    if(args.background_data_file is not None):
        with open(args.background_data_file, 'r') as f:    
            background_filenames = f.read().split()
    else:
        background_filenames = None

    trainer = PoseTrainer(train_filenames = train_filenames,
                          valid_filenames = valid_filenames,
                          img_size = (args.width,args.height),
                          batch_size = args.batch_size,
                          num_workers = args.num_workers,
                          background_filenames = background_filenames,
                          max_orientation_offset = args.max_orientation_offset,
                          prerender = args.prerender,
                          num_model_imgs = args.num_model_imgs,
                          train_data_folder = args.train_data_folder,
                          valid_data_folder = args.valid_data_folder,
                          save_data = args.save_data and args.prerender,
                          seed = args.random_seed)


    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if(args.model_type == 'alexnet'):
        model = gen_pose_net_alexnet(pretrained=args.pretrained)
    elif(args.model_type == 'vgg'):
        model = gen_pose_net_vgg16(pretrained=args.pretrained)
    elif(args.model_type == 'resnet101'):
        model = gen_pose_net_resnet101(classification = args.classification, 
                                       regression = args.regression, 
                                       pretrained=args.pretrained)
    elif(args.model_type == 'resnet50'):
        model = gen_pose_net_resnet50(classification = args.classification, 
                                      regression = args.regression, 
                                      pretrained=args.pretrained)
    else:
        raise AssertionError('Model type {} not supported, alexnet, vgg and resnet are only valid types'.format(args.model_type))
        
    trainer.train(model, results_dir, 
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr)

if __name__=='__main__':
    main()
