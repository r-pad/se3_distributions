# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 05:35:19 2018

@author: bokorn
"""

import numpy as np
import cv2
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_folder = '/home/bokorn/results/result_imgs/iter_full/' 
train_models = np.load(os.path.join(model_folder, 'train_models.npy'))
train_names = [fn.split('/')[-2] for fn in train_models]
valid_models = np.load(os.path.join(model_folder, 'valid_models.npy'))
valid_names = [fn.split('/')[-2] for fn in valid_models]

#dot_q_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_quat/2018-04-11_06-22-46/weights'
#dot_a_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_axis/2018-04-11_06-25-46/weights'
#mult_q_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_quat/2018-04-11_06-28-02/weights'
#mult_a_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_axis/2018-04-11_06-39-32/weights'
#true_q_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_quat/2018-04-11_06-52-10/weights'
#true_a_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_axis/2018-04-11_06-50-11/weights'
#
#quat_weight_paths = [dot_q_path,mult_q_path,true_q_path]
#
#experiment_name = []
#for path in quat_weight_paths:
#    experiment_name.append(path.split('/')[4])
#
#experiment_name.append('alex_reg_car42')
#experiment_name.append('alex_reg_car42_90deg')
#experiment_name.append('alex_reg_car42_45deg')

#results_folder = '/home/bokorn/results/result_imgs/iter10x10/'
#results_folder = '/home/bokorn/results/result_imgs/iter_sub45/'

#results_folder = '/home/bokorn/results/result_imgs/iter_full_45/' 
#results_folder = '/home/bokorn/results/result_imgs/iter_full/' 
#results_folder = '/home/bokorn/results/result_imgs/iter_full_noclip/' 
results_folder = '/home/bokorn/results/result_imgs/iter_full_all_car/' 
#results_folder = '/home/bokorn/results/result_imgs/iter_full_shapenet_45/' 

HAS_VCLASS = False
mean_train_errs = []
mean_valid_errs = []
if(HAS_VCLASS):
    mean_vclass_errs = []

#labels = ['data_45', 
#          #'data_90',  
#          'baseline',
#          'max_est',  
#          #'max_true',
#          ]
#          
#experiment_name = ['alex_reg_car42_45deg', 
#                   #'alex_reg_car42_90deg', 
#                   'alex_reg_car42', 
#                   'alex_reg_car42_maxdot_45deg_quat', 
#                   #'alex_reg_car42_maxtrue_45deg_quat'
#                   ]

experiment_name = ['alex_reg_all_car',
                   'alex_reg_all_car_45deg',
                   'alex_reg_all_car_90deg',
                   'alex_reg_all_car_maxest_45deg',
                   #'alex_reg_all_car_maxest_90deg',
                   #'alex_reg_all_car_maxtrue_45deg'
                   ]

labels = ['baseline',
          'data_45', 
          'data_90',
          'max_est_45',
          #'max_est_90',
          #'max_true_45'
          ]
          
#experiment_name = ['alexnet_reg_shapenet',
#                   'alexnet_reg_shapenet_45deg',
#                   'alexnet_reg_shapenet_90deg',
#                   'alex_reg_shapenet_maxest_45deg']
#labels = ['baseline',
#          'data_45', 
#          'data_90',
#          'max_est_45']
                
for j, name in enumerate(experiment_name):
    filename = os.path.join(results_folder, name+'_errs.npz')
    print(filename)
    data = np.load(filename)
    
    train_errs=data['train_errs']
#    train_imgs=data['train_imgs']
#    train_tgts=data['train_tgts']
    valid_errs=data['valid_errs']
#    valid_imgs=data['valid_imgs']
#    valid_tgts=data['valid_tgts']
    
    train_mean = train_errs.mean(2)
    valid_mean = valid_errs.mean(2)
    plt.plot(train_mean.T*180/np.pi)
    #plt.legend(train_names)
    plt.title(labels[j]+'_train')

    plt.savefig(results_folder + labels[j]+'_train.png')
    plt.gcf().clear()

    plt.plot(valid_mean.T*180/np.pi)
    #plt.legend(valid_names)
    plt.title(labels[j]+'_valid')

    plt.savefig(results_folder + labels[j]+'_valid.png')
    plt.gcf().clear()

    if(HAS_VCLASS):
        vclass_errs=data['valid_class_errs']
        vclass_mean = vclass_errs.mean(2)
        plt.plot(vclass_mean.T*180/np.pi)
        #plt.legend(valid_names)
        plt.title(labels[j]+'_vclass')
    
        plt.savefig(results_folder + labels[j]+'_vclass.png')
        plt.gcf().clear()


    plt.plot(train_mean.mean(0)*180/np.pi, label=labels[j]+'_train')
    plt.plot(valid_mean.mean(0)*180/np.pi, label=labels[j]+'_valid')

    if(HAS_VCLASS):
        plt.plot(vclass_mean.mean(0)*180/np.pi, label=labels[j]+'_vclass')

    plt.legend()
    plt.savefig(results_folder + labels[j]+'_mean.png')
    plt.gcf().clear()
    
    mean_train_errs.append(train_errs.mean(2).mean(0))
    mean_valid_errs.append(valid_errs.mean(2).mean(0))

    if(HAS_VCLASS):
        mean_vclass_errs.append(vclass_errs.mean(2).mean(0))
    
#    filename = os.path.join(results_folder, name+'_errs.npz')
#    np.savez(filename, 
#             train_errs=train_errs,
#             valid_errs=valid_errs)
#    filename = os.path.join(results_folder, name+'_imgs.npz')
#    np.savez(filename, 
#             train_imgs=train_imgs,
#             train_tgts=train_tgts,
#             valid_imgs=valid_imgs,
#             valid_tgts=valid_tgts)
#             
    #cv2.imwrite("/home/bokorn/results/test/imgs/disp_render.png", np.transpose(render_imgs[0], [1,2,0])*255)
    
    
for name, train_mean in zip(labels, mean_train_errs):
    plt.plot(train_mean*180/np.pi, label=name+'_train')
    #plt.plot(valid_mean*180/np.pi, label=name+'_valid')

plt.legend()
plt.savefig(results_folder +'train_results.png')
plt.gcf().clear()

for name, valid_mean in zip(labels, mean_valid_errs):
    #plt.plot(train_mean*180/np.pi, label=name+'_train')
    plt.plot(valid_mean*180/np.pi, label=name+'_valid')

plt.legend()
plt.savefig(results_folder +'valid_results.png')
plt.gcf().clear()

if(HAS_VCLASS):
    for name, vclass_mean in zip(labels, mean_vclass_errs):
        #plt.plot(train_mean*180/np.pi, label=name+'_train')
        plt.plot(vclass_mean*180/np.pi, label=name+'_vclass')
    
    plt.legend()
    plt.savefig(results_folder +'vclass_results.png')
    plt.gcf().clear()

#import IPython; IPython.embed()