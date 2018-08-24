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
np.savez('/home/bokorn/results/result_imgs/shapenet/alexnet_reg_shapenet_noloop.npz',
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
np.savez('/home/bokorn/results/result_imgs/shapenet/alexnet_reg_shapenet_loop.npz',
         train = loop_train_data,
         valid_class = loop_valid_class_data,
         valid_model = loop_valid_model_data,
         valid_pose = loop_valid_pose_data)

a45_path = '/home/bokorn/results/alexnet_reg_shapenet_loop_45deg/'
a45_train_path = glob.glob(a45_path + '**/train/**/events.*', recursive=True)[-1]
a45_valid_class_path = glob.glob(a45_path + '**/valid_class/**/events.*', recursive=True)[-1]
a45_valid_model_path = glob.glob(a45_path + '**/valid_model/**/events.*', recursive=True)[-1]
a45_valid_pose_path = glob.glob(a45_path + '**/valid_pose/**/events.*', recursive=True)[-1]
a45_train_data = np.array(tb_utils.getSummaryData(a45_train_path, ['mean_err'])['mean_err'])
a45_valid_class_data = np.array(tb_utils.getSummaryData(a45_valid_class_path, ['mean_err'])['mean_err'])
a45_valid_model_data = np.array(tb_utils.getSummaryData(a45_valid_model_path, ['mean_err'])['mean_err'])
a45_valid_pose_data = np.array(tb_utils.getSummaryData(a45_valid_pose_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/shapenet/alexnet_reg_shapenet_loop_45deg.npz',
         train = a45_train_data,
         valid_class = a45_valid_class_data,
         valid_model = a45_valid_model_data,
         valid_pose = a45_valid_pose_data)

plt.plot(loop_train_data[:,0], scipy.ndimage.uniform_filter(loop_train_data[:,1], size=filter_size), marker='', markevery=100, label='loop_train')
plt.plot(loop_valid_class_data[:,0], scipy.ndimage.uniform_filter(loop_valid_class_data[:,1], size=filter_size), marker='', markevery=100, label='loop_valid_class')
plt.plot(loop_valid_model_data[:,0], scipy.ndimage.uniform_filter(loop_valid_model_data[:,1], size=filter_size), marker='', markevery=100, label='loop_valid_model')
plt.plot(loop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(loop_valid_pose_data[:,1], size=filter_size), marker='', markevery=100, label='loop_valid_pose')

plt.plot(noloop_train_data[:,0], scipy.ndimage.uniform_filter(noloop_train_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_train')
plt.plot(noloop_valid_class_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_class_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_class')
plt.plot(noloop_valid_model_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_model_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_model')
plt.plot(noloop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_pose_data[:,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/shapenet/alex_shapenet_loop.png")
plt.gcf().clear()

a45_len = a45_valid_class_data.shape[0]
filter_size = 10

plt.plot(noloop_train_data[:a45_len,0], scipy.ndimage.uniform_filter(noloop_train_data[:a45_len,1], size=filter_size), marker='x', markevery=100, label='no_loop_train')
plt.plot(noloop_valid_class_data[:a45_len,0], scipy.ndimage.uniform_filter(noloop_valid_class_data[:a45_len,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_class')
plt.plot(noloop_valid_model_data[:a45_len,0], scipy.ndimage.uniform_filter(noloop_valid_model_data[:a45_len,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_model')
plt.plot(noloop_valid_pose_data[:a45_len,0], scipy.ndimage.uniform_filter(noloop_valid_pose_data[:a45_len,1], size=filter_size), marker='x', markevery=100, label='no_loop_valid_pose')

plt.plot(a45_train_data[:,0], scipy.ndimage.uniform_filter(a45_train_data[:,1], size=filter_size), marker='', markevery=100, label='a45_train')
plt.plot(a45_valid_class_data[:,0], scipy.ndimage.uniform_filter(a45_valid_class_data[:,1], size=filter_size), marker='', markevery=100, label='a45_valid_class')
plt.plot(a45_valid_model_data[:,0], scipy.ndimage.uniform_filter(a45_valid_model_data[:,1], size=filter_size), marker='', markevery=100, label='a45_valid_model')
plt.plot(a45_valid_pose_data[:,0], scipy.ndimage.uniform_filter(a45_valid_pose_data[:,1], size=filter_size), marker='', markevery=100, label='a45_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/shapenet/alex_shapenet_45.png")
plt.gcf().clear()


import IPython; IPython.embed()