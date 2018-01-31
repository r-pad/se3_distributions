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
from torch.optim import Adam, Adadelta, SGD
from torch.utils.data import DataLoader

import os

from generic_pose.datasets.image_dataset import PoseImageDataSet

from generic_pose.losses.viewpoint_loss import ViewpointLoss, denseViewpointError
from generic_pose.losses.quaternion_loss import quaternionLoss, quaternionError
from generic_pose.utils.display_pose import makeDisplayImages, renderTopRotations, renderQuaternions
    
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
                 train_data_folders,
                 valid_data_folders,
                 img_size = (227,227),
                 batch_size = 32,
                 num_workers = 4,
                 model_filenames = None,
                 background_filenames = None,
                 classification = True,
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
        self.train_loader = DataLoader(PoseImageDataSet(data_folders=train_data_folders,
                                                        img_size = img_size,
                                                        model_filenames=model_filenames,
                                                        background_filenames = background_filenames,
                                                        classification = classification,
                                                        num_bins=self.num_bins,
                                                        distance_sigma=distance_sigma),
                                       num_workers=batch_size, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    
        self.valid_loader = DataLoader(PoseImageDataSet(data_folders=valid_data_folders,
                                                        img_size = img_size,
                                                        model_filenames=model_filenames,
                                                        background_filenames = background_filenames,
                                                        classification = classification,
                                                        num_bins=self.num_bins,
                                                        distance_sigma=distance_sigma),
                                       num_workers=num_workers, 
                                       batch_size=batch_size, 
                                       shuffle=True)
    
    def evaluateModel(self, model, origin, query, 
                      quat_true, class_true,
                      backward=True, disp_metrics=False):

        origin = to_var(origin)
        query = to_var(query)
        quat_true = to_var(quat_true)
        class_true = to_var(class_true)

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
            
            if(disp_metrics):
                err_quat = quaternionError(quat_est, quat_true)
                results['err_quat'] = err_quat
                results['mean_origin_features_regression'] = np.mean(np.abs(to_np(origin_features_regression)))
                results['mean_query_features_regression'] = np.mean(np.abs(to_np(query_features_regression)))

        if(model.features_classification is not None):
            origin_features_classification = model.featuresClassification(origin)
            query_features_classification = model.featuresClassification(query)
            class_est = model.compareClassification(origin_features_classification,
                                                    query_features_classification)
            
            loss_binned = self.class_loss(class_est, class_true)
            

            if(backward):
                loss_binned.backward()
        
            results['class_est'] = class_est
            results['loss_binned'] = loss_binned.data[0]
            if(disp_metrics):
                err_binned, err_idx = denseViewpointError(class_est, class_true, self.num_bins)            
                results['err_binned'] = err_binned
                results['err_idx'] = err_idx
                results['mean_origin_features_classification'] = np.mean(np.abs(to_np(origin_features_classification)))
                results['mean_query_features_classification'] = np.mean(np.abs(to_np(query_features_classification)))        
        
        if(backward):
            self.optimizer.step()

        return results
    
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
                loq_data = not((cumulative_batch_idx+1) % log_every_nth)
                train_results = self.evaluateModel(model, origin, query, 
                                                   quat_true, class_true,
                                                   backward=True, disp_metrics = loq_data)
                      
                if loq_data:
                    v_origin, v_query, v_quat_true, v_class_true, v_origin_quat, v_model_file = next(iter(self.valid_loader))
                    
                    valid_results = self.evaluateModel(model, v_origin, v_query, 
                                                       v_quat_true, v_class_true,
                                                       backward=False, disp_metrics=True)
                    
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
                    valid_query_imgs = to_np(v_query.view(-1, 3, self.img_size[0], self.img_size[1])[:num_display_imgs])
                    valid_quat_true = to_np(v_quat_true[:num_display_imgs])
                    
                    if('quat_est' in valid_results):
                        valid_quat_est = to_np(valid_results['quat_est'][:num_display_imgs])
                        valid_render_imgs = renderQuaternions(valid_origin_imgs, valid_query_imgs, 
                                                              valid_quat_true, valid_quat_est,
                                                              origin_quats = v_origin_quat[:num_display_imgs],
                                                              model_files=v_model_file[:num_display_imgs],
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
    from generic_pose.models.generic_pose_net import gen_pose_net_alexnet, gen_pose_net_vgg16, gen_pose_net_resnet101, gen_pose_net_resnet50
    
    parser = ArgumentParser()

    parser.add_argument('--train_data_folders', type=str, default=None)
    parser.add_argument('--valid_data_folders', type=str, default=None)
    
    parser.add_argument('--weight_file', type=str, default=None)
    parser.add_argument('--model_data_file', type=str, default=None)
    parser.add_argument('--background_data_file', type=str, default=None)

    parser.add_argument('--robot', dest='robot', action='store_true')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    
    parser.add_argument('--distance_sigma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--model_type', type=str, default='resnet101')
    parser.add_argument('--train_classification', dest='classification', action='store_true')
    parser.add_argument('--train_regression', dest='regression', action='store_true')
    parser.add_argument('--random_init', dest='pretrained', action='store_false')    
    
    parser.add_argument('--random_seed', type=int, default=0)


    parser.add_argument('--results_dir', type=str, default='results/') 
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--log_every_nth', type=int, default=50)

    args = parser.parse_args()

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
    
    render_distance = 2
    if(args.model_data_file is not None):
        if(args.robot):
            render_distance = 0.5
            model_filenames = args.model_data_file
            
        else:
            model_filenames = {}
            with open(args.model_data_file, 'r') as f:    
                filenames = f.read().split()
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

    trainer = PoseTrainer(train_data_folders = train_data_folders,
                          valid_data_folders = valid_data_folders,
                          img_size = (args.width,args.height),
                          batch_size = args.batch_size,
                          num_workers = args.num_workers,
                          model_filenames = model_filenames,
                          background_filenames = background_filenames,
                          classification = args.classification, 
                          num_bins = (50,50,25),
                          distance_sigma = args.distance_sigma, 
                          seed = args.random_seed,
                          render_distance = render_distance)


    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)    
    
    if args.weight_file is not None:
        args.pretrained = False

    if(args.model_type.lower() == 'alexnet'):
        model = gen_pose_net_alexnet(pretrained=args.pretrained)
    elif(args.model_type.lower() == 'vgg'):
        model = gen_pose_net_vgg16(classification = args.classification, 
                                   regression = args.regression, 
                                   pretrained = args.pretrained)
    elif(args.model_type.lower() == 'resnet101'):
        model = gen_pose_net_resnet101(classification = args.classification, 
                                       regression = args.regression, 
                                       pretrained = args.pretrained)
    elif(args.model_type.lower() == 'resnet50'):
        model = gen_pose_net_resnet50(classification = args.classification, 
                                      regression = args.regression, 
                                      pretrained = args.pretrained)
    else:
        raise AssertionError('Model type {} not supported, alexnet, vgg and resnet are only valid types'.format(args.model_type))

    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))

    trainer.train(model, results_dir, 
                  num_epochs = args.num_epochs,
                  log_every_nth = args.log_every_nth,
                  lr = args.lr,
                  optimizer=args.optimizer)

if __name__=='__main__':
    main()
