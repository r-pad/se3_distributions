# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:52:26 2018

@author: bokorn
"""

import numpy as np
import scipy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob

import generic_pose.utils.tensorboard_utils as tb_utils

filter_size = 100

noloop_path = '/home/bokorn/results/alexnet_reg_shapenet_noloop/'

noloop_train_path = glob.glob(noloop_path + '**/train/**/events.*', recursive=True)[-1]
noloop_valid_class_path = glob.glob(noloop_path + '**/valid_class/**/events.*', recursive=True)[-1]
noloop_valid_model_path = glob.glob(noloop_path + '**/valid_model/**/events.*', recursive=True)[-1]
noloop_valid_pose_path = glob.glob(noloop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
noloop_train_data = np.array(tb_utils.getSummaryData(noloop_train_path, ['mean_err'])['mean_err'])
noloop_valid_class_data = np.array(tb_utils.getSummaryData(noloop_valid_class_path, ['mean_err'])['mean_err'])
noloop_valid_model_data = np.array(tb_utils.getSummaryData(noloop_valid_model_path, ['mean_err'])['mean_err'])
noloop_valid_pose_data = np.array(tb_utils.getSummaryData(noloop_valid_pose_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/alexnet_reg_shapenet_noloop.npz',
         train = noloop_train_data,
         valid_class = noloop_valid_class_data,
         valid_model = noloop_valid_model_data,
         valid_pose = noloop_valid_pose_data)

loop_path = '/home/bokorn/results/alexnet_reg_shapenet/'
loop_train_path = glob.glob(loop_path + '**/train/**/events.*', recursive=True)[-1]
loop_valid_class_path = glob.glob(loop_path + '**/valid_class/**/events.*', recursive=True)[-1]
loop_valid_model_path = glob.glob(loop_path + '**/valid_model/**/events.*', recursive=True)[-1]
loop_valid_pose_path = glob.glob(loop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
loop_train_data = np.array(tb_utils.getSummaryData(loop_train_path, ['mean_err'])['mean_err'])
loop_valid_class_data = np.array(tb_utils.getSummaryData(loop_valid_class_path, ['mean_err'])['mean_err'])
loop_valid_model_data = np.array(tb_utils.getSummaryData(loop_valid_model_path, ['mean_err'])['mean_err'])
loop_valid_pose_data = np.array(tb_utils.getSummaryData(loop_valid_pose_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/alexnet_reg_shapenet_loop.npz',
         train = loop_train_data,
         valid_class = loop_valid_class_data,
         valid_model = loop_valid_model_data,
         valid_pose = loop_valid_pose_data)

import IPython; IPython.embed()

plt.plot(loop_train_data[:,0], scipy.ndimage.uniform_filter(loop_train_data[:,1], size=filter_size), marker='', markevery=100, label='loop_train')
plt.plot(loop_valid_class_data[:,0], scipy.ndimage.uniform_filter(loop_valid_class_data[:,1], size=filter_size), marker='', markevery=100, label='loop_valid_class')
plt.plot(loop_valid_model_data[:,0], scipy.ndimage.uniform_filter(loop_valid_model_data[:,1], size=filter_size), marker='', markevery=100, label='loop_valid_model')
plt.plot(loop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(loop_valid_pose_data[:,1], size=filter_size), marker='', markevery=100, label='loop_valid_pose')

plt.plot(noloop_train_data[:,0], scipy.ndimage.uniform_filter(noloop_train_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_train')
plt.plot(noloop_valid_class_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_class_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_class')
plt.plot(noloop_valid_model_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_model_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_model')
plt.plot(noloop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_pose_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/alex_car3_angle.png")
plt.gcf().clear()

import IPython; IPython.embed()