# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 02:13:18 2017

@author: bokorn
"""
from logger import Logger

import torch
import torch.optim as optim
from torch.autograd import Variable
from argparse import ArgumentParser
from torch.optim import Adam, Adadelta
from torch.utils.data import DataLoader
import numpy as np
import os
import datetime, os

from pose_data_loader import PoseDirDataSet, PoseFileDataSet
from gen_pose_net import gen_pose_net, gen_pose_net_stacked
from viewpoint_loss import ViewpointLoss, viewpointAccuracy
from quaternion_loss import quaternionLoss

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
    
def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    
    
def train_with_dataloader(args):#, model):
    if(args.model == 'stacked'):
        model = gen_pose_net_stacked()
    else:
        model = gen_pose_net(pretrained=True)


    
    model.train()
    model.cuda()
#    loader = DataLoader(PoseDirDataSet(data_dir = args.train_data_dir,
#                                       img_size = (args.width, args.height)),
#                        num_workers=args.num_workers, 
#                        batch_size=args.batch_size, 
#                        shuffle=True)
    
    with open(args.train_data_file, 'r') as f:    
        train_filenames = f.read().split()
    with open(args.valid_data_file, 'r') as f:    
        valid_filenames = f.read().split()                
    
    train_loader = DataLoader(PoseFileDataSet(render_filenames = train_filenames,
                                              img_size = (args.width, args.height),
                                              max_orientation_offset = args.max_orientation_offset),
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True)

    valid_loader = DataLoader(PoseFileDataSet(render_filenames = valid_filenames,
                                              img_size = (args.width, args.height),
                                              max_orientation_offset = args.max_orientation_offset),
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True)
                              
    optimizer = Adadelta(model.parameters())

    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(args.results_dir,current_timestamp)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    log_dir = os.path.join(results_dir,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = Logger(log_dir)

    # init empty loss files
    running_loss_file = os.path.join(results_dir, 'running_loss.txt')
    instantaneous_loss_file = os.path.join(results_dir, 'instantaneous_loss.txt')
    binned_loss_file = os.path.join(results_dir, 'binned_loss.txt')
    open(os.path.join(results_dir, 'running_loss.txt'), 'a').close()
    open(os.path.join(results_dir, 'instantaneous_loss.txt'), 'a').close()
    open(os.path.join(results_dir, 'binned_loss.txt'), 'a').close()

    cumulative_batch_idx = 0
    running_loss_quat = 0.0
    running_loss_azim = 0.0
    running_loss_elev = 0.0
    running_loss_tilt = 0.0
    running_err_azim = 0.0
    running_err_elev = 0.0
    running_err_tilt = 0.0
    
    binned_loss_quat = np.zeros(360)
    
    binned_loss_azim = np.zeros(360)
    binned_loss_elev = np.zeros(360)
    binned_loss_tilt = np.zeros(360)

    binned_err_azim = np.zeros(360)
    binned_err_elev = np.zeros(360)
    binned_err_tilt = np.zeros(360)
    
    binned_count_quat = np.zeros(360)
    binned_count_azim = np.zeros(360)
    binned_count_elev = np.zeros(360)
    binned_count_tilt = np.zeros(360)
    
    euler_loss = ViewpointLoss()

    print('Starting Training')

    for epoch_idx in range(1, args.num_epochs+1):
        for batch_idx, (origin, query, conj_q, d_quat, d_azim, d_elev, d_tilt, d_euler) in enumerate(train_loader):
            origin = to_var(origin)
            query = to_var(query)
            conj_q = to_var(conj_q)
            d_quat = to_var(d_quat)
            d_azim = to_var(d_azim)
            d_elev = to_var(d_elev)
            d_tilt = to_var(d_tilt)
            #d_euler = Variable(d_euler)

            optimizer.zero_grad()
            quat_est, azim_est, elev_est, tilt_est = model.forward(origin, query)
            #import IPython; IPython.embed()

            loss_quat = quaternionLoss(quat_est, d_quat)
            
            loss_azim = euler_loss(azim_est, d_azim)
            loss_elev = euler_loss(elev_est, d_elev)
            loss_tilt = euler_loss(tilt_est, d_tilt)
            loss_euler = loss_azim + loss_elev + loss_tilt
            
            loss_quat.backward(retain_graph=True)
            loss_euler.backward()
            
            optimizer.step()
            
#            running_loss_quat += loss_quat.mean().data[0]
#            running_loss_azim += loss_azim.sum().data[0]
#            running_loss_elev += loss_elev.sum().data[0]
#            running_loss_tilt += loss_tilt.sum().data[0]
            running_loss_quat += loss_quat.data[0]
            running_loss_azim += loss_azim.data[0]
            running_loss_elev += loss_elev.data[0]
            running_loss_tilt += loss_tilt.data[0]
            
            err_azim = viewpointAccuracy(azim_est, d_azim)
            err_elev = viewpointAccuracy(elev_est, d_elev)
            err_tilt = viewpointAccuracy(tilt_est, d_tilt)

            running_err_azim += err_azim
            running_err_elev += err_elev
            running_err_tilt += err_tilt
            
#            running_err_azim += err_azim.sum() / err_azim.size(0)
#            running_err_elev += err_elev.sum() / err_elev.size(0)
#            running_err_tilt += err_tilt.sum() / err_tilt.size(0)
#            
#            q_idx = (2 * np.arccos(to_np(d_quat[:,-1])) *180./np.pi).astype(int)
#            a_idx = to_np(d_azim.max(1)[1])
#            e_idx = to_np(d_elev.max(1)[1])
#            t_idx = to_np(d_tilt.max(1)[1])
#                                                   
#            for i, (q_i, a_i, e_i, t_i) in enumerate(zip(q_idx, a_idx, e_idx, t_idx)):
##                binned_loss_quat[q_i] += loss_quat[i].data[0]
##                binned_loss_azim[a_i] += loss_azim[i].data[0]
##                binned_loss_elev[e_i] += loss_elev[i].data[0]
##                binned_loss_tilt[t_i] += loss_tilt[i].data[0]
#                binned_err_azim[a_i] += err_azim[i]
#                binned_err_elev[e_i] += err_elev[i]
#                binned_err_tilt[e_i] += err_tilt[i]
#
#                binned_count_quat[q_i] += 1
#                binned_count_azim[a_i] += 1
#                binned_count_elev[e_i] += 1
#                binned_count_tilt[t_i] += 1            
            
            if cumulative_batch_idx>0 and not(cumulative_batch_idx % args.print_loss_every_nth):
                v_origin, v_query, v_conj_q, v_d_quat, v_d_azim, v_d_elev, v_d_tilt, v_d_euler = next(iter(valid_loader))
                v_origin = to_var(v_origin)
                v_query = to_var(v_query)
                v_conj_q = to_var(v_conj_q)
                v_d_quat = to_var(v_d_quat)
                v_d_azim = to_var(v_d_azim)
                v_d_elev = to_var(v_d_elev)
                v_d_tilt = to_var(v_d_tilt)
            
                v_quat_est, v_azim_est, v_elev_est, v_tilt_est = model.forward(v_origin, v_query)
                
                v_loss_quat = quaternionLoss(v_quat_est, v_d_quat)

                v_loss_azim = euler_loss(v_azim_est, v_d_azim)
                v_loss_elev = euler_loss(v_elev_est, v_d_elev)
                v_loss_tilt = euler_loss(v_tilt_est, v_d_tilt)
                v_loss_euler = v_loss_azim + v_loss_elev + v_loss_tilt
                
                v_err_azim = viewpointAccuracy(v_azim_est, v_d_azim)
                v_err_elev = viewpointAccuracy(v_elev_est, v_d_elev)
                v_err_tilt = viewpointAccuracy(v_tilt_est, v_d_tilt)

                print("epoch {} :: cumulative_batch_idx {}".format(epoch_idx, cumulative_batch_idx + 1))
                
#                print("epoch {} :: cumulative_batch_idx {} :: \nloss quat {}, \nloss azim {}, \nloss elev {}, \nloss tilt {}, \nacc azim {}, \nacc elev {}, \nacc tilt {}".format(epoch_idx, cumulative_batch_idx + 1, 
#                      running_loss_quat / args.print_loss_every_nth, 
#                      running_loss_azim / args.print_loss_every_nth,
#                      running_loss_elev / args.print_loss_every_nth,
#                      running_loss_tilt / args.print_loss_every_nth,
#                      running_err_azim / args.print_loss_every_nth,
#                      running_err_elev / args.print_loss_every_nth,
#                      running_err_tilt / args.print_loss_every_nth))
                      
#                print(torch.stack([d_azim.max(1)[1].data, azim_est.max(1)[1].data, 
#                                   d_elev.max(1)[1].data, elev_est.max(1)[1].data, 
#                                   d_tilt.max(1)[1].data, tilt_est.max(1)[1].data]).transpose(0,1))
                        
#                with open(instantaneous_loss_file, "a") as f:
#                    f.write(str(cumulative_batch_idx) 
#                        + ", " + str(loss_quat.data[0]) 
#                        + ", " + str(loss_azim.data[0]) 
#                        + ", " + str(loss_elev.data[0])
#                        + ", " + str(loss_tilt.data[0])
#                        + '\n')
#                with open(running_loss_file, "a") as f:
#                    f.write(str(cumulative_batch_idx) 
#                        + ", " + str(running_loss_quat / args.print_loss_every_nth) 
#                        + ", " + str(running_loss_azim / args.print_loss_every_nth) 
#                        + ", " + str(running_loss_elev / args.print_loss_every_nth) 
#                        + ", " + str(running_loss_tilt / args.print_loss_every_nth) 
#                        + '\n')
#                import IPython; IPython.embed()
#                with open(binned_loss_file, "a") as f:
#                    f.write(str(cumulative_batch_idx) 
#                        + ", " + str((binned_loss_quat / binned_count_quat).tolist()).replace('[','').replace(']','')
##                        + ", " + str((binned_loss_azim / binned_count_azim).tolist()).replace('[','').replace(']','')
##                        + ", " + str((binned_loss_elev / binned_count_elev).tolist()).replace('[','').replace(']','')
##                        + ", " + str((binned_loss_tilt / binned_count_tilt).tolist()).replace('[','').replace(']','')
#                        + ", " + str((binned_err_azim / binned_count_azim).tolist()).replace('[','').replace(']','')
#                        + ", " + str((binned_err_elev / binned_count_elev).tolist()).replace('[','').replace(']','')
#                        + ", " + str((binned_err_tilt / binned_count_tilt).tolist()).replace('[','').replace(']','')
#                        + '\n')
#                        
                running_loss_quat = 0.0
                running_loss_azim = 0.0
                running_loss_elev = 0.0
                running_loss_tilt = 0.0
                running_err_azim = 0.0
                running_err_elev = 0.0
                running_err_tilt = 0.0
                binned_loss_quat = np.zeros(360)
                
#                binned_loss_azim = np.zeros(360)
#                binned_loss_elev = np.zeros(360)
#                binned_loss_tilt = np.zeros(360)
            
                binned_err_azim = np.zeros(360)
                binned_err_elev = np.zeros(360)
                binned_err_tilt = np.zeros(360)
                
                binned_count_quat = np.zeros(360)
                binned_count_azim = np.zeros(360)
                binned_count_elev = np.zeros(360)
                binned_count_tilt = np.zeros(360)
                                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'train_loss_quat': loss_quat.data[0],
                    'train_loss_azim': loss_azim.data[0],
                    'train_loss_elev': loss_elev.data[0],
                    'train_loss_tilt': loss_tilt.data[0],
                    'train_loss_euler': loss_euler.data[0],
                    'train_err_azim': err_azim,
                    'train_err_elev': err_elev,
                    'train_err_tilt': err_tilt,
                    'valid_loss_quat': v_loss_quat.data[0],
                    'valid_loss_azim': v_loss_azim.data[0],
                    'valid_loss_elev': v_loss_elev.data[0],
                    'valid_loss_tilt': v_loss_tilt.data[0],
                    'valid_loss_euler': v_loss_euler.data[0],
                    'valid_err_azim': v_err_azim,
                    'valid_err_elev': v_err_elev,
                    'valid_err_tilt': v_err_tilt,
                }
        
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, cumulative_batch_idx+1)
        
                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), cumulative_batch_idx+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), cumulative_batch_idx+1)
        
                # (3) Log the images
                info = {
                    'train_origin': to_np(origin.view(-1, 3, args.width, args.height)[:10]),
                    'train_query': to_np(query.view(-1, 3, args.width, args.height)[:10]),                                         
                    'valid_origin': to_np(v_origin.view(-1, 3, args.width, args.height)[:10]),
                    'valid_query': to_np(v_query.view(-1, 3, args.width, args.height)[:10]),
                }
        
                for tag, images in info.items():
                    logger.image_summary(tag, images, cumulative_batch_idx+1)

            if cumulative_batch_idx>0 and not(cumulative_batch_idx % args.save_every_nth): 
#                print("saving model ", os.path.join(results_dir, 'epoch_{}_batch_{}_lq_{:.4f}_la_{:.4f}_le_{:.4f}_lt_{:.4f}.pth'.\
#                                                            format(epoch_idx, str(cumulative_batch_idx).zfill(8), 
#                                                                   loss_quat.data[0], loss_azim.data[0], loss_elev.data[0], loss_tilt.data[0])))
#                torch.save(model.state_dict(), os.path.join(results_dir, 'epoch_{}_batch_{}_lq_{:.4f}_la_{:.4f}_le_{:.4f}_lt_{:.4f}.pth'.\
#                                                            format(epoch_idx, str(cumulative_batch_idx).zfill(8), 
#                                                                   loss_quat.data[0], loss_azim.data[0], loss_elev.data[0], loss_tilt.data[0])))
#                                                                   print("saving model ", os.path.join(results_dir, 'epoch_{}_batch_{}_lq_{:.4f}_la_{:.4f}_le_{:.4f}_lt_{:.4f}.pth'.\
#                                                            format(epoch_idx, str(cumulative_batch_idx).zfill(8), 
#                                                                   loss_quat.data[0], loss_azim.data[0], loss_elev.data[0], loss_tilt.data[0])))
                print("saving model")
                torch.save(model.state_dict(), os.path.join(results_dir, 'weights.pth'))
            cumulative_batch_idx += 1

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--train_data_dir', type=str, default='/media/bokorn/ExtraDrive1/render_cnn_data/model_images/syn_images/')
    parser.add_argument('--train_data_file', type=str, default='/home/bokorn/src/generic_pose/generic_pose/src/chairs_100_train.txt')
    parser.add_argument('--valid_data_file', type=str, default='/home/bokorn/src/generic_pose/generic_pose/src/chairs_100_valid.txt')
    parser.add_argument('--max_orientation_offset', type=float, default=float('inf'))
    parser.add_argument('--results_dir', type=str, default='/media/bokorn/ExtraDrive2/posenet_results/') 
    #parser.add_argument('--log_dir', type=str, default='/media/bokorn/ExtraDrive1/posenet_logs/') 
    parser.add_argument('--save_every_nth', type=int, default=500)
    parser.add_argument('--print_loss_every_nth', type=int, default=10)
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