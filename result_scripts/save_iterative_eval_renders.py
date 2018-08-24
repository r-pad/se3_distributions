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


dot_q_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_quat/2018-04-11_06-22-46/weights'
dot_a_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_axis/2018-04-11_06-25-46/weights'
mult_q_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_quat/2018-04-11_06-28-02/weights'
mult_a_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_axis/2018-04-11_06-39-32/weights'
true_q_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_quat/2018-04-11_06-52-10/weights'
true_a_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_axis/2018-04-11_06-50-11/weights'

quat_weight_paths = [dot_q_path,mult_q_path,true_q_path]

experiment_name = []
for path in quat_weight_paths:
    experiment_name.append(path.split('/')[4])

experiment_name.append('alex_reg_car42')
experiment_name.append('alex_reg_car42_90deg')
experiment_name.append('alex_reg_car42_45deg')

results_folder = '/home/bokorn/results/result_imgs/iter10x10/'

mean_train_errs = []
mean_valid_errs = []

labels = ['max_dot', 'max_mult', 'max_true', 'baseline', 'data_90', 'data_45']

for j, name in enumerate(experiment_name):
    filename = os.path.join(results_folder, name+'_imgs.npz')
    print(filename)
    data = np.load(filename)
    

    train_imgs=data['train_imgs']
    train_tgts=data['train_tgts']

    valid_imgs=data['valid_imgs']
    valid_tgts=data['valid_tgts']

    if not os.path.exists(results_folder+labels[j]):
        os.mkdir(results_folder+labels[j])
    for model_num in range(train_imgs.shape[0]):
        model_folder = results_folder+labels[j]+'/train_model_{}'.format(model_num)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        for init_num in range(train_imgs.shape[2]):
            init_folder = model_folder+'/init_{}'.format(init_num)
            if not os.path.exists(init_folder):
                os.mkdir(init_folder)
            cv2.imwrite(init_folder+'/target.png', train_tgts[model_num, init_num])
            for k in range(train_imgs.shape[1]):
                cv2.imwrite(init_folder+'/img{:02d}.png'.format(k), train_imgs[model_num, k, init_num])
                
    for model_num in range(valid_imgs.shape[0]):
        model_folder = results_folder+labels[j]+'/valid_model_{}'.format(model_num)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        for init_num in range(valid_imgs.shape[2]):
            init_folder = model_folder+'/init_{}'.format(init_num)
            if not os.path.exists(init_folder):
                os.mkdir(init_folder)
            cv2.imwrite(init_folder+'/target.png', valid_tgts[model_num, init_num])
            for k in range(valid_imgs.shape[1]):
                cv2.imwrite(init_folder+'/img{:02d}.png'.format(k), valid_imgs[model_num, k, init_num])
                

import IPython; IPython.embed()