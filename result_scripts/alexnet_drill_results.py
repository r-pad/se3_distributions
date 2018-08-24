# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:52:26 2018

@author: bokorn
"""

import numpy as np
import scipy
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generic_pose.utils.tensorboard_utils as tb_utils

filter_size = 10

#baseline_train = '/home/bokorn/results/alexnet_reg_baseline_drill/2018-03-15_20-59-53/logs/train/events.out.tfevents.1521147653.compute-0-9.local'
#baseline_valid = '/home/bokorn/results/alexnet_reg_baseline_drill/2018-03-15_20-59-53/logs/valid/events.out.tfevents.1521147653.compute-0-9.local'
#baseline_train_data = np.array(tb_utils.getSummaryData(baseline_train, ['mean_err'])['mean_err'])
#baseline_valid_data = np.array(tb_utils.getSummaryData(baseline_valid, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/drill/alexnet_reg_baseline_drill.npz',
#         train = baseline_train_data,
#         valid = baseline_valid_data)

res_baseline = np.load('/home/bokorn/results/result_imgs/drill/alexnet_reg_baseline_drill.npz')
baseline_train_data = res_baseline['train']
baseline_valid_data = res_baseline['valid']

###############################
### Symetric Training Drill ###
###############################
#both_train = '/home/bokorn/results/alexnet_reg_both_drill/2018-03-25_04-36-15/logs/train/events.out.tfevents.1521952701.compute-0-9.local'
#both_valid = '/home/bokorn/results/alexnet_reg_both_drill/2018-03-25_04-36-15/logs/valid/events.out.tfevents.1521952701.compute-0-9.local'
#both_train_data = np.array(tb_utils.getSummaryData(both_train, ['mean_err'])['mean_err'])
#both_valid_data = np.array(tb_utils.getSummaryData(both_valid, ['mean_err'])['mean_err'])
#
#split_train = '/home/bokorn/results/alexnet_reg_split_drill/2018-03-25_04-13-27/logs/train/events.out.tfevents.1521951281.compute-0-9.local'
#split_valid = '/home/bokorn/results/alexnet_reg_split_drill/2018-03-25_04-13-27/logs/valid/events.out.tfevents.1521951281.compute-0-9.local'
#split_train_data = np.array(tb_utils.getSummaryData(split_train, ['mean_err'])['mean_err'])
#split_valid_data = np.array(tb_utils.getSummaryData(split_valid, ['mean_err'])['mean_err'])
#
#sym_train = '/home/bokorn/results/alexnet_reg_sym_drill/2018-03-25_04-11-45/logs/train/events.out.tfevents.1521951192.compute-0-7.local'
#sym_valid = '/home/bokorn/results/alexnet_reg_sym_drill/2018-03-25_04-11-45/logs/valid/events.out.tfevents.1521951192.compute-0-7.local'
#sym_train_data = np.array(tb_utils.getSummaryData(sym_train, ['mean_err'])['mean_err'])
#sym_valid_data = np.array(tb_utils.getSummaryData(sym_valid, ['mean_err'])['mean_err'])

#plt.plot(baseline_train_data[:600,0], scipy.ndimage.uniform_filter(baseline_train_data[:600,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_data[:600,0], scipy.ndimage.uniform_filter(baseline_valid_data[:600,1], size=filter_size), label='baseline_valid')
#
#plt.plot(both_train_data[:,0], scipy.ndimage.uniform_filter(both_train_data[:,1], size=filter_size), label='both_train')
#plt.plot(both_valid_data[:,0], scipy.ndimage.uniform_filter(both_valid_data[:,1], size=filter_size), label='both_valid')
#
#plt.plot(split_train_data[:,0], scipy.ndimage.uniform_filter(split_train_data[:,1], size=filter_size), label='split_train')
#plt.plot(split_valid_data[:,0], scipy.ndimage.uniform_filter(split_valid_data[:,1], size=filter_size), label='split_valid')
#
#plt.plot(sym_train_data[:,0], scipy.ndimage.uniform_filter(sym_train_data[:,1], size=filter_size), label='sym_train')
#plt.plot(sym_valid_data[:,0], scipy.ndimage.uniform_filter(sym_valid_data[:,1], size=filter_size), label='sym_valid')

#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/sym_tests.png")
#plt.gcf().clear()

#####################################
### Skip Connection Compare Drill ###
#####################################
#skip_train = '/home/bokorn/results/alexnet_reg_skip_drill/2018-03-25_00-57-41/logs/train/events.out.tfevents.1521939545.compute-0-7.local'
#skip_valid = '/home/bokorn/results/alexnet_reg_skip_drill/2018-03-25_00-57-41/logs/valid/events.out.tfevents.1521939545.compute-0-7.local'
#skip_train_data = np.array(tb_utils.getSummaryData(skip_train, ['mean_err'])['mean_err'])
#skip_valid_data = np.array(tb_utils.getSummaryData(skip_valid, ['mean_err'])['mean_err'])

#plt.plot(baseline_train_data[:600,0], scipy.ndimage.uniform_filter(baseline_train_data[:600,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_data[:600,0], scipy.ndimage.uniform_filter(baseline_valid_data[:600,1], size=filter_size), label='baseline_valid')
#
#plt.plot(skip_train_data[:,0], scipy.ndimage.uniform_filter(skip_train_data[:,1], size=filter_size), label='skip_train')
#plt.plot(skip_valid_data[:,0], scipy.ndimage.uniform_filter(skip_valid_data[:,1], size=filter_size), label='skip_valid')

#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/skip_tests.png")
#plt.gcf().clear()

############################
### Fixed Features Drill ###
############################
#fixed_train = '/home/bokorn/results/alexnet_reg_fixed_features_drill/2018-03-25_03-52-28/logs/train/events.out.tfevents.1521949981.compute-0-9.local'
#fixed_valid = '/home/bokorn/results/alexnet_reg_fixed_features_drill/2018-03-25_03-52-28/logs/valid/events.out.tfevents.1521949981.compute-0-9.local'
#fixed_train_data = np.array(tb_utils.getSummaryData(fixed_train, ['mean_err'])['mean_err'])
#fixed_valid_data = np.array(tb_utils.getSummaryData(fixed_valid, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/drill/alexnet_reg_fixed_features_drill.npz',
#         train = fixed_train_data,
#         valid = fixed_valid_data)
#
#plt.plot(baseline_train_data[:600,0], scipy.ndimage.uniform_filter(baseline_train_data[:600,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_data[:600,0], scipy.ndimage.uniform_filter(baseline_valid_data[:600,1], size=filter_size), label='baseline_valid')
#
#plt.plot(fixed_train_data[:,0], scipy.ndimage.uniform_filter(fixed_train_data[:,1], size=filter_size), label='fixed_train')
#plt.plot(fixed_valid_data[:,0], scipy.ndimage.uniform_filter(fixed_valid_data[:,1], size=filter_size), label='fixed_valid')
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/alex_fixed.png")
#plt.gcf().clear()

######################################
### Fully Connected Features Drill ###
######################################
#linear_train = '/home/bokorn/results/alexnet_reg_linear_drill/2018-03-25_00-57-40/logs/train/events.out.tfevents.1521939547.compute-0-7.local'
#linear_valid = '/home/bokorn/results/alexnet_reg_linear_drill/2018-03-25_00-57-40/logs/valid/events.out.tfevents.1521939547.compute-0-7.local'
#linear_train_data = np.array(tb_utils.getSummaryData(linear_train, ['mean_err'])['mean_err'])
#linear_valid_data = np.array(tb_utils.getSummaryData(linear_valid, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/drill/alexnet_reg_linear_drill.npz',
#         train = linear_train_data,
#         valid = linear_valid_data)
#
#plt.plot(baseline_train_data[:600,0], scipy.ndimage.uniform_filter(baseline_train_data[:600,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_data[:600,0], scipy.ndimage.uniform_filter(baseline_valid_data[:600,1], size=filter_size), label='baseline_valid')
#
#plt.plot(linear_train_data[:,0], scipy.ndimage.uniform_filter(linear_train_data[:,1], size=filter_size), label='linear_train')
#plt.plot(linear_valid_data[:,0], scipy.ndimage.uniform_filter(linear_valid_data[:,1], size=filter_size), label='linear_valid')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/alex_linear.png")
#plt.gcf().clear()

###############################
### Clipped Gradients Drill ###
###############################
clip_path = '/home/bokorn/results/alexnet_reg_clip_drill/'
clip_train_path = glob.glob(clip_path + '**/train/**/events.*', recursive=True)[-1]
#clip_valid_class_path = glob.glob(clip_path + '**/valid_class/**/events.*', recursive=True)[-1]
#clip_valid_model_path = glob.glob(clip_path + '**/valid_model/**/events.*', recursive=True)[-1]
clip_valid_pose_path = glob.glob(clip_path + '**/valid_pose/**/events.*', recursive=True)[-1]
clip_train_data = np.array(tb_utils.getSummaryData(clip_train_path, ['mean_err'])['mean_err'])
#clip_valid_class_data = np.array(tb_utils.getSummaryData(clip_valid_class_path, ['mean_err'])['mean_err'])
#clip_valid_model_data = np.array(tb_utils.getSummaryData(clip_valid_model_path, ['mean_err'])['mean_err'])
clip_valid_pose_data = np.array(tb_utils.getSummaryData(clip_valid_pose_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/drill/alexnet_reg_clip_drill.npz',
         train = clip_train_data,
#         valid_class = clip_valid_class_data,
         #valid_model = clip_valid_model_data,
         valid_pose = clip_valid_pose_data,
         )
plt.plot(baseline_train_data[:600,0], scipy.ndimage.uniform_filter(baseline_train_data[:600,1], size=filter_size), label='baseline_train')
plt.plot(baseline_valid_data[:600,0], scipy.ndimage.uniform_filter(baseline_valid_data[:600,1], size=filter_size), label='baseline_valid')

plt.plot(clip_train_data[:,0], scipy.ndimage.uniform_filter(clip_train_data[:,1], size=filter_size), label='linear_train')
plt.plot(clip_valid_pose_data[:,0], scipy.ndimage.uniform_filter(clip_valid_pose_data[:,1], size=filter_size), label='clip_valid')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/drill/alex_clip.png")
plt.gcf().clear()

loop_path = '/home/bokorn/results/alexnet_reg_loop_drill/'
loop_train_path = glob.glob(loop_path + '**/train/**/events.*', recursive=True)[-1]
#clip_valid_class_path = glob.glob(clip_path + '**/valid_class/**/events.*', recursive=True)[-1]
#clip_valid_model_path = glob.glob(clip_path + '**/valid_model/**/events.*', recursive=True)[-1]
loop_valid_pose_path = glob.glob(loop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
loop_train_data = np.array(tb_utils.getSummaryData(loop_train_path, ['mean_err'])['mean_err'])
#clip_valid_class_data = np.array(tb_utils.getSummaryData(clip_valid_class_path, ['mean_err'])['mean_err'])
#clip_valid_model_data = np.array(tb_utils.getSummaryData(clip_valid_model_path, ['mean_err'])['mean_err'])
loop_valid_pose_data = np.array(tb_utils.getSummaryData(loop_valid_pose_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/drill/alexnet_reg_loop_drill.npz',
         train = loop_train_data,
#         valid_class = clip_valid_class_data,
         #valid_model = clip_valid_model_data,
         valid_pose = loop_valid_pose_data,
         )
plt.plot(baseline_train_data[:600,0], scipy.ndimage.uniform_filter(baseline_train_data[:600,1], size=filter_size), label='baseline_train')
plt.plot(baseline_valid_data[:600,0], scipy.ndimage.uniform_filter(baseline_valid_data[:600,1], size=filter_size), label='baseline_valid')

plt.plot(loop_train_data[:,0], scipy.ndimage.uniform_filter(loop_train_data[:,1], size=filter_size), label='loop_train')
plt.plot(loop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(loop_valid_pose_data[:,1], size=filter_size), label='loop_valid')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/drill/alex_loop.png")
plt.gcf().clear()

import IPython; IPython.embed()