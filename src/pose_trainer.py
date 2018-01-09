# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from logger import Logger

import torch

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

        if(backward):
            self.optimizer.zero_grad()
            
        quat_est, u0_est, u1_est, u2_est = model.forward(origin, query)

        loss_quat = quaternionLoss(quat_est, quat_true)
        
        loss_u0 = self.class_loss(u0_est, u0_true)
        loss_u1 = self.class_loss(u1_est, u1_true)
        loss_u2 = self.class_loss(u2_est, u2_true)
        loss_binned = loss_u0 + loss_u1 + loss_u2

        if(backward):
            loss_quat.backward(retain_graph=True)
            loss_binned.backward()
        
            self.optimizer.step()

        err_u0 = viewpointAccuracy(u0_est, u0_true)
        err_u1 = viewpointAccuracy(u1_est, u1_true)
        err_u2 = viewpointAccuracy(u2_est, u2_true)
        
        results = {'quat_est':quat_est, 
                   'u0_est':u0_est,
                   'u1_est':u1_est, 
                   'u2_est':u2_est,
                   'loss_quat':loss_quat, 
                   'loss_u0':loss_u0, 
                   'loss_u1':loss_u1, 
                   'loss_u2':loss_u2, 
                   'loss_binned':loss_binned, 
                   'err_u0':err_u0, 
                   'err_u1':err_u1, 
                   'err_u2':err_u2}        
        
        return results
    
    def train(self, model, results_dir, 
              num_epochs = 100000,
              log_every_nth = 10):
        model.train()
        model.cuda()
        self.optimizer = Adadelta(model.parameters())
 

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

                    info = {
                        'train_loss_quat': train_results['loss_quat'].data[0],
                        'train_loss_u0': train_results['loss_u0'].data[0],
                        'train_loss_u1': train_results['loss_u1'].data[0],
                        'train_loss_u2': train_results['loss_u2'].data[0],
                        'train_loss_binned': train_results['loss_binned'].data[0],
                        'train_err_u0': train_results['err_u0'],
                        'train_err_u1': train_results['err_u1'],
                        'train_err_u2': train_results['err_u2'],
                        'valid_loss_quat': valid_results['loss_quat'].data[0],
                        'valid_loss_u0': valid_results['loss_u0'].data[0],
                        'valid_loss_u1': valid_results['loss_u1'].data[0],
                        'valid_loss_u2': valid_results['loss_u2'].data[0],
                        'valid_loss_binned': valid_results['loss_binned'].data[0],
                        'valid_err_u0': valid_results['err_u0'],
                        'valid_err_u1': valid_results['err_u1'],
                        'valid_err_u2': valid_results['err_u2'],
                    }
            
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, cumulative_batch_idx+1)
            
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                        logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)
            
                    train_disp_imgs = makeDisplayImage(to_np(origin.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(query.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(u0_true[:10]), to_np(u1_true[:10]), 
                                                       to_np(u2_true[:10]), to_np(quat_true[:10]),
                                                       to_np(train_results['u0_est'][:10]), to_np(train_results['u1_est'][:10]),
                                                       to_np(train_results['u2_est'][:10]), to_np(train_results['quat_est'][:10]))
                    valid_disp_imgs = makeDisplayImage(to_np(v_origin.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(v_query.view(-1, 3, self.img_size[0], self.img_size[1])[:10]),
                                                       to_np(v_u0_true[:10]), to_np(v_u1_true[:10]), 
                                                       to_np(v_u2_true[:10]), to_np(v_quat_true[:10]),
                                                       to_np(valid_results['u0_est'][:10]), to_np(valid_results['u1_est'][:10]),
                                                       to_np(valid_results['u2_est'][:10]), to_np(valid_results['quat_est'][:10]))
            
                    info = {
                        'train': train_disp_imgs,
                        'valid': valid_disp_imgs,
                    }
            
                    for tag, images in info.items():
                        logger.image_summary(tag, images, cumulative_batch_idx+1)
                    
                    if(valid_results['loss_quat'].data[0] < min_loss_quat or valid_results['loss_binned'].data[0] < min_loss_binned):

                        min_loss_quat = min(min_loss_quat, valid_results['loss_quat'].data[0])
                        min_loss_binned = min(min_loss_binned, valid_results['loss_binned'].data[0])

                        weights_filename = os.path.join(weights_dir, 'epoch_{0:0{4}d}_batch_{1:0{5}d}_lquat_{2:.4f}_lbinned_{3:.4f}.pth'.\
                                              format(epoch_idx, cumulative_batch_idx + 1, 
                                              valid_results['loss_quat'].data[0], 
                                              valid_results['loss_binned'].data[0],
                                              epoch_digits, batch_digits))
                                                                           
                        print("saving model ", weights_filename)
                        torch.save(model.state_dict(), weights_filename)

                cumulative_batch_idx += 1

def main():
    import datetime
    from argparse import ArgumentParser
    from gen_pose_net import gen_pose_net
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_file', type=str, default=None)
    parser.add_argument('--valid_data_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)
    parser.add_argument('--max_orientation_offset', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=227)
    parser.add_argument('--width', type=int, default=227)
    
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
    
    model = gen_pose_net(pretrained=True)    
    
    trainer.train(model, results_dir, 
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth)

if __name__=='__main__':
    main()
