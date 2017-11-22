# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 02:13:18 2017

@author: bokorn
"""

import torch
import torch.optim as optim
from torch.autograd import Variable
from argparse import ArgumentParser
from torch.optim import Adam, Adadelta
from torch.utils.data import DataLoader
import numpy as np
import os
import datetime, os

from pose_data_loader import PoseDataLoader
from gen_pose_net import gen_pose_net, gen_pose_net_stacked
from viewpoint_loss import ViewpointLoss
from quaternion_loss import quaternionLoss

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
    


#def my_collate(batch):
#    batch = filter (lambda x:x is not None, batch)
#    return default_collate(batch)

def train_with_dataloader(args):#, model):
    if(args.model == 'stacked'):
        model = gen_pose_net_stacked()
    else:
        model = gen_pose_net(pretrained=True)
        
    model.train()
    model.cuda()
    loader = DataLoader(PoseDataLoader(data_dir=args.train_data_dir,
                                       img_size= (args.width, args.height)),
                        num_workers=args.num_workers, 
                        batch_size=args.batch_size, 
                        shuffle=True)

    optimizer = Adadelta(model.parameters())

    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # init empty loss files
    running_loss_file = os.path.join(results_dir, 'running_loss.txt')
    instantaneous_loss_file = os.path.join(results_dir, 'instantaneous_loss.txt')
    open(os.path.join(results_dir, 'running_loss.txt'), 'a').close()
    open(os.path.join(results_dir, 'instantaneous_loss.txt'), 'a').close()

    cumulative_batch_idx = 0
    running_loss_quat = 0.0
    running_loss_azim = 0.0
    running_loss_elev = 0.0
    running_loss_tilt = 0.0

    euler_loss = ViewpointLoss()

    for epoch_idx in range(1, args.num_epochs+1):
        for batch_idx, (origin, query, conj_q, d_azim, d_elev, d_tilt) in enumerate(loader):

            if torch.cuda.is_available():
                origin = Variable(origin.cuda())
                query = Variable(query.cuda())
                conj_q = Variable(conj_q.cuda())
                d_azim = Variable(d_azim.cuda())
                d_elev = Variable(d_elev.cuda())
                d_tilt = Variable(d_tilt.cuda())
            else:                
                origin = Variable(origin)
                query = Variable(query)
                conj_q = Variable(conj_q)
                d_azim = Variable(d_azim)
                d_elev = Variable(d_elev)
                d_tilt = Variable(d_tilt)

            optimizer.zero_grad()
            quat_est, azim_est, elev_est, tilt_est = model.forward(origin, query)
            loss_quat = quaternionLoss(quat_est, conj_q)
            
            loss_azim = euler_loss(azim_est, d_azim)
            loss_elev = euler_loss(elev_est, d_elev)
            loss_tilt = euler_loss(tilt_est, d_tilt)
            loss_euler = loss_azim + loss_elev + loss_tilt
            
            loss_quat.backward(retain_graph=True)
            loss_euler.backward()
            
            optimizer.step()
            
            running_loss_quat += loss_quat.data[0]
            running_loss_azim += loss_azim.data[0]
            running_loss_elev += loss_elev.data[0]
            running_loss_tilt += loss_tilt.data[0]
            if cumulative_batch_idx>0 and not(cumulative_batch_idx % args.print_loss_every_nth):
                print("epoch {} :: cumulative_batch_idx {} :: \nloss quat {}, \nloss azim {}, \nloss elev {}, \nloss tilt {}".format(epoch_idx, cumulative_batch_idx + 1, 
                      running_loss_quat / args.print_loss_every_nth, 
                      running_loss_azim / args.print_loss_every_nth,
                      running_loss_elev / args.print_loss_every_nth,
                      running_loss_tilt / args.print_loss_every_nth))
                with open(instantaneous_loss_file, "a") as f:
                    f.write(str(cumulative_batch_idx) 
                        + ", " + str(loss_quat.data[0]) 
                        + ", " + str(loss_azim.data[0]) 
                        + ", " + str(loss_elev.data[0])
                        + ", " + str(loss_tilt.data[0])
                        + '\n')
                with open(running_loss_file, "a") as f:
                    f.write(str(cumulative_batch_idx) 
                        + ", " + str(running_loss_quat / args.print_loss_every_nth) 
                        + ", " + str(running_loss_azim / args.print_loss_every_nth) 
                        + ", " + str(running_loss_elev / args.print_loss_every_nth) 
                        + ", " + str(running_loss_tilt / args.print_loss_every_nth) 
                        + '\n')
                running_loss_quat = 0.0
                running_loss_azim = 0.0
                running_loss_elev = 0.0
                running_loss_tilt = 0.0

            if cumulative_batch_idx>0 and not(cumulative_batch_idx % args.save_every_nth): 
                print("saving model ", os.path.join(results_dir, 'epoch_{}_batch_{}_lq_{:.4f}_la_{:.4f}_le_{:.4f}_lt_{:.4f}.pth'.\
                                                            format(epoch_idx, str(cumulative_batch_idx).zfill(8), 
                                                                   loss_quat.data[0], loss_azim.data[0], loss_elev.data[0], loss_tilt.data[0])))
                torch.save(model.state_dict(), os.path.join(results_dir, 'epoch_{}_batch_{}_lq_{:.4f}_la_{:.4f}_le_{:.4f}_lt_{:.4f}.pth'.\
                                                            format(epoch_idx, str(cumulative_batch_idx).zfill(8), 
                                                                   loss_quat.data[0], loss_azim.data[0], loss_elev.data[0], loss_tilt.data[0])))
            cumulative_batch_idx += 1

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--train_data_dir', type=str, default='/home/bokorn/src/render_cnn/RenderForCNN/data/syn_images/')
    parser.add_argument('--results_dir', type=str, default='/media/bokorn/ExtraDrive1/posenet_results/') 
    parser.add_argument('--save_every_nth', type=int, default=500)
    parser.add_argument('--print_loss_every_nth', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10000000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=227)
    parser.add_argument('--width', type=int, default=227)
    parser.add_argument('--optim', type=str, default='adam')
    # finetuning from pretrained_weight_file
    parser.add_argument('--pretrained_weight_file', type=str)
    # resume training from the most recent weight file
    parser.add_argument('--resume', action='store_true', default=False) 

    args = parser.parse_args()
    train_with_dataloader(args)

if __name__=='__main__':
    main()