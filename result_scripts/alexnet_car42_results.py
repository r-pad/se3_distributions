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

res_baseline = np.load('/home/bokorn/results/result_imgs/car42/alexnet_reg_car42_noloop.npz')
baseline_train_data = res_baseline['train']
baseline_valid_model_data = res_baseline['valid_model']

#################
### Loop Test ###
#################
#noloop_path = '/home/bokorn/results/alex_reg_car42/'
#noloop_train_path = glob.glob(noloop_path + '**/train/**/events.*', recursive=True)[-1]
##noloop_valid_class_path = glob.glob(noloop_path + '**/valid_class/**/events.*', recursive=True)[-1]
#noloop_valid_model_path = glob.glob(noloop_path + '**/valid/**/events.*', recursive=True)[-1]
##noloop_valid_pose_path = glob.glob(noloop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#noloop_train_data = np.array(tb_utils.getSummaryData(noloop_train_path, ['mean_err'])['mean_err'])
##noloop_valid_class_data = np.array(tb_utils.getSummaryData(noloop_valid_class_path, ['mean_err'])['mean_err'])
#noloop_valid_model_data = np.array(tb_utils.getSummaryData(noloop_valid_model_path, ['mean_err'])['mean_err'])
##noloop_valid_pose_data = np.array(tb_utils.getSummaryData(noloop_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alexnet_reg_car42_noloop.npz',
#         train = noloop_train_data,
###         valid_class = noloop_valid_class_data,
#         valid_model = noloop_valid_model_data,
##         valid_pose = noloop_valid_pose_data,
#         )
#
#loop_path = '/home/bokorn/results/alex_reg_car42_loop_new/2018-03-27_03-16-53/'
#loop_train_path = glob.glob(loop_path + '**/train/**/events.*', recursive=True)[-1]
##loop_valid_class_path = glob.glob(loop_path + '**/valid_class/**/events.*', recursive=True)[-1]
#loop_valid_model_path = glob.glob(loop_path + '**/valid_model/**/events.*', recursive=True)[-1]
#loop_valid_pose_path = glob.glob(loop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#loop_train_data = np.array(tb_utils.getSummaryData(loop_train_path, ['mean_err'])['mean_err'])
##loop_valid_class_data = np.array(tb_utils.getSummaryData(loop_valid_class_path, ['mean_err'])['mean_err'])
#loop_valid_model_data = np.array(tb_utils.getSummaryData(loop_valid_model_path, ['mean_err'])['mean_err'])
#loop_valid_pose_data = np.array(tb_utils.getSummaryData(loop_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alexnet_reg_car42_loop.npz',
#         train = loop_train_data,
##         valid_class = loop_valid_class_data,
#         valid_model = loop_valid_model_data,
#         valid_pose = loop_valid_pose_data,
#         )
#
#plt.plot(loop_train_data[:,0], scipy.ndimage.uniform_filter(loop_train_data[:,1], size=filter_size), label='loop_train')
#plt.plot(loop_valid_model_data[:,0], scipy.ndimage.uniform_filter(loop_valid_model_data[:,1], size=filter_size), label='loop_valid_model')
#plt.plot(loop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(loop_valid_pose_data[:,1], size=filter_size), label='loop_valid_pose')
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='no_loop_train')
#plt.plot(baseline_valid_model_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:,1], size=filter_size), label='no_loop_valid_model')
##plt.plot(noloop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_pose_data[:,1], size=filter_size), label='no_loop_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_loop.png")
#plt.gcf().clear()

#######################
### Angle Loop Test ###
#######################
#a90_loop_path = '/home/bokorn/results/alex_reg_car42_loop_90deg/'
#a90_loop_train_path = glob.glob(a90_loop_path + '**/train/**/events.*', recursive=True)[-1]
#a90_loop_valid_model_path = glob.glob(a90_loop_path + '**/valid_model/**/events.*', recursive=True)[-1]
#a90_loop_valid_pose_path = glob.glob(a90_loop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#a90_loop_train_data = np.array(tb_utils.getSummaryData(a90_loop_train_path, ['mean_err'])['mean_err'])
#a90_loop_valid_model_data = np.array(tb_utils.getSummaryData(a90_loop_valid_model_path, ['mean_err'])['mean_err'])
#a90_loop_valid_pose_data = np.array(tb_utils.getSummaryData(a90_loop_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_loop_90deg.npz',
#         train = a90_loop_train_data,
#         valid_model = a90_loop_valid_model_data,
#         valid_pose = a90_loop_valid_pose_data)
#
#a45_loop_path = '/home/bokorn/results/alex_reg_car42_loop_45deg/'
#a45_loop_train_path = glob.glob(a45_loop_path + '**/train/**/events.*', recursive=True)[-1]
#a45_loop_valid_model_path = glob.glob(a45_loop_path + '**/valid_model/**/events.*', recursive=True)[-1]
#a45_loop_valid_pose_path = glob.glob(a45_loop_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#a45_loop_train_data = np.array(tb_utils.getSummaryData(a45_loop_train_path, ['mean_err'])['mean_err'])
#a45_loop_valid_model_data = np.array(tb_utils.getSummaryData(a45_loop_valid_model_path, ['mean_err'])['mean_err'])
#a45_loop_valid_pose_data = np.array(tb_utils.getSummaryData(a45_loop_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_loop_45deg.npz',
#         train = a45_loop_train_data,
#         valid_model = a45_loop_valid_model_data,
#         valid_pose = a45_loop_valid_pose_data)
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_model_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:,1], size=filter_size), label='baseline_valid_model')
##plt.plot(baseline_valid_pose_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_pose_data[:,1], size=filter_size), label='baseline_valid_pose')
#
#plt.plot(a90_loop_train_data[:,0], scipy.ndimage.uniform_filter(a90_loop_train_data[:,1], size=filter_size), label='a90_loop_train')
#plt.plot(a90_loop_valid_model_data[:,0], scipy.ndimage.uniform_filter(a90_loop_valid_model_data[:,1], size=filter_size), label='a90_loop_valid_model')
#plt.plot(a90_loop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(a90_loop_valid_pose_data[:,1], size=filter_size), label='a90_loop_valid_pose')
#
#plt.plot(a45_loop_train_data[:,0], scipy.ndimage.uniform_filter(a45_loop_train_data[:,1], size=filter_size), label='a45_loop_train')
#plt.plot(a45_loop_valid_model_data[:,0], scipy.ndimage.uniform_filter(a45_loop_valid_model_data[:,1], size=filter_size), label='a45_loop_valid_model')
#plt.plot(a45_loop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(a45_loop_valid_pose_data[:,1], size=filter_size), label='a45_loop_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_loop_angle.png")
#plt.gcf().clear()

#######################
### Angle Loop Test ###
#######################
#a90_path = '/home/bokorn/results/alex_reg_car42_90deg/'
#a90_train_path = glob.glob(a90_path + '**/train/**/events.*', recursive=True)[-1]
#a90_valid_model_path = glob.glob(a90_path + '**/valid_model/**/events.*', recursive=True)[-1]
#a90_valid_pose_path = glob.glob(a90_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#a90_train_data = np.array(tb_utils.getSummaryData(a90_train_path, ['mean_err'])['mean_err'])
#a90_valid_model_data = np.array(tb_utils.getSummaryData(a90_valid_model_path, ['mean_err'])['mean_err'])
#a90_valid_pose_data = np.array(tb_utils.getSummaryData(a90_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_loop_90deg.npz',
#         train = a90_train_data,
#         valid_model = a90_valid_model_data,
#         valid_pose = a90_valid_pose_data)
#
#a45_path = '/home/bokorn/results/alex_reg_car42_45deg/'
#a45_train_path = glob.glob(a45_path + '**/train/**/events.*', recursive=True)[-1]
#a45_valid_model_path = glob.glob(a45_path + '**/valid_model/**/events.*', recursive=True)[-1]
#a45_valid_pose_path = glob.glob(a45_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#a45_train_data = np.array(tb_utils.getSummaryData(a45_train_path, ['mean_err'])['mean_err'])
#a45_valid_model_data = np.array(tb_utils.getSummaryData(a45_valid_model_path, ['mean_err'])['mean_err'])
#a45_valid_pose_data = np.array(tb_utils.getSummaryData(a45_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_loop_45deg.npz',
#         train = a45_train_data,
#         valid_model = a45_valid_model_data,
#         valid_pose = a45_valid_pose_data)
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_model_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:,1], size=filter_size), label='baseline_valid_model')
##plt.plot(baseline_valid_pose_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_pose_data[:,1], size=filter_size), label='baseline_valid_pose')
#
#plt.plot(a90_train_data[:,0], scipy.ndimage.uniform_filter(a90_train_data[:,1], size=filter_size), label='a90_train')
#plt.plot(a90_valid_model_data[:,0], scipy.ndimage.uniform_filter(a90_valid_model_data[:,1], size=filter_size), label='a90_valid_model')
#plt.plot(a90_valid_pose_data[:,0], scipy.ndimage.uniform_filter(a90_valid_pose_data[:,1], size=filter_size), label='a90_valid_pose')
#
#plt.plot(a45_train_data[:,0], scipy.ndimage.uniform_filter(a45_train_data[:,1], size=filter_size), label='a45_train')
#plt.plot(a45_valid_model_data[:,0], scipy.ndimage.uniform_filter(a45_valid_model_data[:,1], size=filter_size), label='a45_valid_model')
#plt.plot(a45_valid_pose_data[:,0], scipy.ndimage.uniform_filter(a45_valid_pose_data[:,1], size=filter_size), label='a45_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_angle.png")
#plt.gcf().clear()

###################
### Linear Test ###
###################
#linear_path = '/home/bokorn/results/alex_reg_car42_loop_new/2018-03-27_20-52-40/'
#linear_train_path = glob.glob(linear_path + '**/train/**/events.*', recursive=True)[-1]
##linear_valid_class_path = glob.glob(linear_path + '**/valid_class/**/events.*', recursive=True)[-1]
#linear_valid_model_path = glob.glob(linear_path + '**/valid_model/**/events.*', recursive=True)[-1]
#linear_valid_pose_path = glob.glob(linear_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#linear_train_data = np.array(tb_utils.getSummaryData(linear_train_path, ['mean_err'])['mean_err'])
##linear_valid_class_data = np.array(tb_utils.getSummaryData(linear_valid_class_path, ['mean_err'])['mean_err'])
#linear_valid_model_data = np.array(tb_utils.getSummaryData(linear_valid_model_path, ['mean_err'])['mean_err'])
#linear_valid_pose_data = np.array(tb_utils.getSummaryData(linear_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alexnet_reg_car42_linear.npz',
#         train = linear_train_data,
##         valid_class = linear_valid_class_data,
#         valid_model = linear_valid_model_data,
#         valid_pose = linear_valid_pose_data,
#         )
#
#plt.plot(linear_train_data[:,0], scipy.ndimage.uniform_filter(linear_train_data[:,1], size=filter_size), label='linear_train')
#plt.plot(linear_valid_model_data[:,0], scipy.ndimage.uniform_filter(linear_valid_model_data[:,1], size=filter_size), label='linear_valid_model')
#plt.plot(linear_valid_pose_data[:,0], scipy.ndimage.uniform_filter(linear_valid_pose_data[:,1], size=filter_size), label='linear_valid_pose')
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='no_loop_train')
#plt.plot(baseline_valid_model_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:,1], size=filter_size), label='no_loop_valid_model')
##plt.plot(noloop_valid_pose_data[:,0], scipy.ndimage.uniform_filter(noloop_valid_pose_data[:,1], size=filter_size), label='no_loop_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_linear.png")
#plt.gcf().clear()

###########################
### Angle Cirriculumn 3 ###
###########################
#ac3_path = '/home/bokorn/results/alex_reg_car42_ang_cur_3/'
#ac3_train_path = glob.glob(ac3_path + '**/train/**/events.*', recursive=True)[-1]
#ac3_valid_model_path = glob.glob(ac3_path + '**/valid_model/**/events.*', recursive=True)[-1]
#ac3_valid_pose_path = glob.glob(ac3_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#ac3_train_data = np.array(tb_utils.getSummaryData(ac3_train_path, ['mean_err'])['mean_err'])
#ac3_valid_model_data = np.array(tb_utils.getSummaryData(ac3_valid_model_path, ['mean_err'])['mean_err'])
#ac3_valid_pose_data = np.array(tb_utils.getSummaryData(ac3_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_ang_cur_3.npz',
#         train = ac3_train_data,
#         valid_model = ac3_valid_model_data,
#         valid_pose = ac3_valid_pose_data)
#
#ac10_path = '/home/bokorn/results/alex_reg_car42_ang_cur_10/'
#ac10_train_path = glob.glob(ac10_path + '**/train/**/events.*', recursive=True)[-1]
#ac10_valid_model_path = glob.glob(ac10_path + '**/valid_model/**/events.*', recursive=True)[-1]
#ac10_valid_pose_path = glob.glob(ac10_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#ac10_train_data = np.array(tb_utils.getSummaryData(ac10_train_path, ['mean_err'])['mean_err'])
#ac10_valid_model_data = np.array(tb_utils.getSummaryData(ac10_valid_model_path, ['mean_err'])['mean_err'])
#ac10_valid_pose_data = np.array(tb_utils.getSummaryData(ac10_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_ang_cur_10.npz',
#         train = ac10_train_data,
#         valid_model = ac10_valid_model_data,
#         valid_pose = ac10_valid_pose_data)
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_model_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:,1], size=filter_size), label='baseline_valid_model')
##plt.plot(baseline_valid_pose_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_pose_data[:,1], size=filter_size), label='baseline_valid_pose')
#
#plt.plot(ac3_train_data[:,0], scipy.ndimage.uniform_filter(ac3_train_data[:,1], size=filter_size), label='ac3_train')
#plt.plot(ac3_valid_model_data[:,0], scipy.ndimage.uniform_filter(ac3_valid_model_data[:,1], size=filter_size), label='ac3_valid_model')
#plt.plot(ac3_valid_pose_data[:,0], scipy.ndimage.uniform_filter(ac3_valid_pose_data[:,1], size=filter_size), label='ac3_valid_pose')
#
#plt.plot(ac10_train_data[:,0], scipy.ndimage.uniform_filter(ac10_train_data[:,1], size=filter_size), label='ac10_train')
#plt.plot(ac10_valid_model_data[:,0], scipy.ndimage.uniform_filter(ac10_valid_model_data[:,1], size=filter_size), label='ac10_valid_model')
#plt.plot(ac10_valid_pose_data[:,0], scipy.ndimage.uniform_filter(ac10_valid_pose_data[:,1], size=filter_size), label='ac10_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_cur.png")
#plt.gcf().clear()
#
#cur_len = max(ac3_train_data.shape[0],ac10_train_data.shape[0])
#filter_size = 50
#plt.plot(baseline_train_data[:cur_len,0], scipy.ndimage.uniform_filter(baseline_train_data[:cur_len,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_model_data[:cur_len,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:cur_len,1], size=filter_size), label='baseline_valid_model')
##plt.plot(baseline_valid_pose_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_pose_data[:,1], size=filter_size), label='baseline_valid_pose')
#
#plt.plot(ac3_train_data[:,0], scipy.ndimage.uniform_filter(ac3_train_data[:,1], size=filter_size), label='ac3_train')
#plt.plot(ac3_valid_model_data[:,0], scipy.ndimage.uniform_filter(ac3_valid_model_data[:,1], size=filter_size), label='ac3_valid_model')
#plt.plot(ac3_valid_pose_data[:,0], scipy.ndimage.uniform_filter(ac3_valid_pose_data[:,1], size=filter_size), label='ac3_valid_pose')
#
#plt.plot(ac10_train_data[:,0], scipy.ndimage.uniform_filter(ac10_train_data[:,1], size=filter_size), label='ac10_train')
#plt.plot(ac10_valid_model_data[:,0], scipy.ndimage.uniform_filter(ac10_valid_model_data[:,1], size=filter_size), label='ac10_valid_model')
#plt.plot(ac10_valid_pose_data[:,0], scipy.ndimage.uniform_filter(ac10_valid_pose_data[:,1], size=filter_size), label='ac10_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_cur_zoom.png")
#plt.gcf().clear()

#maxc3_path = '/home/bokorn/results/alex_reg_car42_maxang_cur_3/'
#maxc3_path = '/home/bokorn/results/alex_reg_car42_maxang_cur_3/2018-04-06_06-12-24/'
#maxc3_train_path = glob.glob(maxc3_path + '**/train/**/events.*', recursive=True)[-1]
#maxc3_valid_model_path = glob.glob(maxc3_path + '**/valid_model/**/events.*', recursive=True)[-1]
#maxc3_valid_pose_path = glob.glob(maxc3_path + '**/valid_pose/**/events.*', recursive=True)[-1]
#maxc3_train_data = np.array(tb_utils.getSummaryData(maxc3_train_path, ['mean_err'])['mean_err'])
#maxc3_valid_model_data = np.array(tb_utils.getSummaryData(maxc3_valid_model_path, ['mean_err'])['mean_err'])
#maxc3_valid_pose_data = np.array(tb_utils.getSummaryData(maxc3_valid_pose_path, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/car42/alex_reg_car42_maxang_cur_3.npz',
#         train = maxc3_train_data,
#         valid_model = maxc3_valid_model_data,
#         valid_pose = maxc3_valid_pose_data)
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_model_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:,1], size=filter_size), label='baseline_valid_model')
##plt.plot(baseline_valid_pose_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_pose_data[:,1], size=filter_size), label='baseline_valid_pose')
#
#plt.plot(maxc3_train_data[:,0], scipy.ndimage.uniform_filter(maxc3_train_data[:,1], size=filter_size), label='maxc3_train')
#plt.plot(maxc3_valid_model_data[:,0], scipy.ndimage.uniform_filter(maxc3_valid_model_data[:,1], size=filter_size), label='maxc3_valid_model')
#plt.plot(maxc3_valid_pose_data[:,0], scipy.ndimage.uniform_filter(maxc3_valid_pose_data[:,1], size=filter_size), label='maxc3_valid_pose')
#['diff_vec', 'errs_vec']
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_maxang_cur_3.png")
#plt.gcf().clear()
#
#cur_len = maxc3_train_data.shape[0]
#filter_size = 10
#plt.plot(baseline_train_data[:cur_len,0], scipy.ndimage.uniform_filter(baseline_train_data[:cur_len,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_model_data[:cur_len,0], scipy.ndimage.uniform_filter(baseline_valid_model_data[:cur_len,1], size=filter_size), label='baseline_valid_model')
##plt.plot(baseline_valid_pose_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_pose_data[:,1], size=filter_size), label='baseline_valid_pose')
#
#plt.plot(maxc3_train_data[:,0], scipy.ndimage.uniform_filter(maxc3_train_data[:,1], size=filter_size), label='maxc3_train')
#plt.plot(maxc3_valid_model_data[:,0], scipy.ndimage.uniform_filter(maxc3_valid_model_data[:,1], size=filter_size), label='maxc3_valid_model')
#plt.plot(maxc3_valid_pose_data[:,0], scipy.ndimage.uniform_filter(maxc3_valid_pose_data[:,1], size=filter_size), label='maxc3_valid_pose')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_maxang_cur_3_zoom.png")
#plt.gcf().clear()

dot_q_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_quat/2018-04-11_06-22-46/'
dot_a_path = '/home/bokorn/results/alex_reg_car42_maxdot_45deg_axis/2018-04-11_06-25-46/'
mult_q_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_quat/2018-04-11_06-28-02/'
mult_a_path = '/home/bokorn/results/alex_reg_car42_maxmult_45deg_axis/2018-04-11_06-39-32/'
true_q_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_quat/2018-04-11_06-52-10/'
true_a_path = '/home/bokorn/results/alex_reg_car42_maxtrue_45deg_axis/2018-04-11_06-50-11/'

dot_q_train_path = glob.glob(dot_q_path + '**/train/**/events.*', recursive=True)[-1]
dot_q_valid_model_path = glob.glob(dot_q_path + '**/valid_model/**/events.*', recursive=True)[-1]
dot_q_valid_pose_path = glob.glob(dot_q_path + '**/valid_pose/**/events.*', recursive=True)[-1]
dot_q_train_data = np.array(tb_utils.getSummaryData(dot_q_train_path, ['mean_err'])['mean_err'])
dot_q_valid_model_data = np.array(tb_utils.getSummaryData(dot_q_valid_model_path, ['mean_err'])['mean_err'])
dot_q_valid_pose_data = np.array(tb_utils.getSummaryData(dot_q_valid_pose_path, ['mean_err'])['mean_err'])

dot_a_train_path = glob.glob(dot_a_path + '**/train/**/events.*', recursive=True)[-1]
dot_a_valid_model_path = glob.glob(dot_a_path + '**/valid_model/**/events.*', recursive=True)[-1]
dot_a_valid_pose_path = glob.glob(dot_a_path + '**/valid_pose/**/events.*', recursive=True)[-1]
dot_a_train_data = np.array(tb_utils.getSummaryData(dot_a_train_path, ['mean_err'])['mean_err'])
dot_a_valid_model_data = np.array(tb_utils.getSummaryData(dot_a_valid_model_path, ['mean_err'])['mean_err'])
dot_a_valid_pose_data = np.array(tb_utils.getSummaryData(dot_a_valid_pose_path, ['mean_err'])['mean_err'])

mult_q_train_path = glob.glob(mult_q_path + '**/train/**/events.*', recursive=True)[-1]
mult_q_valid_model_path = glob.glob(mult_q_path + '**/valid_model/**/events.*', recursive=True)[-1]
mult_q_valid_pose_path = glob.glob(mult_q_path + '**/valid_pose/**/events.*', recursive=True)[-1]
mult_q_train_data = np.array(tb_utils.getSummaryData(mult_q_train_path, ['mean_err'])['mean_err'])
mult_q_valid_model_data = np.array(tb_utils.getSummaryData(mult_q_valid_model_path, ['mean_err'])['mean_err'])
mult_q_valid_pose_data = np.array(tb_utils.getSummaryData(mult_q_valid_pose_path, ['mean_err'])['mean_err'])

mult_a_train_path = glob.glob(mult_a_path + '**/train/**/events.*', recursive=True)[-1]
mult_a_valid_model_path = glob.glob(mult_a_path + '**/valid_model/**/events.*', recursive=True)[-1]
mult_a_valid_pose_path = glob.glob(mult_a_path + '**/valid_pose/**/events.*', recursive=True)[-1]
mult_a_train_data = np.array(tb_utils.getSummaryData(mult_a_train_path, ['mean_err'])['mean_err'])
mult_a_valid_model_data = np.array(tb_utils.getSummaryData(mult_a_valid_model_path, ['mean_err'])['mean_err'])
mult_a_valid_pose_data = np.array(tb_utils.getSummaryData(mult_a_valid_pose_path, ['mean_err'])['mean_err'])

true_q_train_path = glob.glob(true_q_path + '**/train/**/events.*', recursive=True)[-1]
true_q_valid_model_path = glob.glob(true_q_path + '**/valid_model/**/events.*', recursive=True)[-1]
true_q_valid_pose_path = glob.glob(true_q_path + '**/valid_pose/**/events.*', recursive=True)[-1]
true_q_train_data = np.array(tb_utils.getSummaryData(true_q_train_path, ['mean_err'])['mean_err'])
true_q_valid_model_data = np.array(tb_utils.getSummaryData(true_q_valid_model_path, ['mean_err'])['mean_err'])
true_q_valid_pose_data = np.array(tb_utils.getSummaryData(true_q_valid_pose_path, ['mean_err'])['mean_err'])

true_a_train_path = glob.glob(true_a_path + '**/train/**/events.*', recursive=True)[-1]
true_a_valid_model_path = glob.glob(true_a_path + '**/valid_model/**/events.*', recursive=True)[-1]
true_a_valid_pose_path = glob.glob(true_a_path + '**/valid_pose/**/events.*', recursive=True)[-1]
true_a_train_data = np.array(tb_utils.getSummaryData(true_a_train_path, ['mean_err'])['mean_err'])
true_a_valid_model_data = np.array(tb_utils.getSummaryData(true_a_valid_model_path, ['mean_err'])['mean_err'])
true_a_valid_pose_data = np.array(tb_utils.getSummaryData(true_a_valid_pose_path, ['mean_err'])['mean_err'])

#####

plt.plot(dot_q_train_data[:,0], scipy.ndimage.uniform_filter(dot_q_train_data[:,1], size=filter_size), label='dot_quat_train')
plt.plot(dot_q_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_model_data[:,1], size=filter_size), label='dot_quat_valid_model')
plt.plot(dot_q_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_pose_data[:,1], size=filter_size), label='dot_quat_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_quat.png")
plt.gcf().clear()

plt.plot(dot_q_train_data[:,0], scipy.ndimage.uniform_filter(dot_q_train_data[:,1], size=filter_size)-126, label='dot_quat_train')
plt.plot(dot_q_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_model_data[:,1], size=filter_size)-126, label='dot_quat_valid_model')
plt.plot(dot_q_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_pose_data[:,1], size=filter_size)-126, label='dot_quat_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_quat_improv.png")
plt.gcf().clear()

plt.plot(dot_a_train_data[:,0], scipy.ndimage.uniform_filter(dot_a_train_data[:,1], size=filter_size), label='dot_axis_train')
plt.plot(dot_a_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_model_data[:,1], size=filter_size), label='dot_axis_valid_model')
plt.plot(dot_a_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_pose_data[:,1], size=filter_size), label='dot_axis_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_axis.png")
plt.gcf().clear()

plt.plot(dot_a_train_data[:,0], scipy.ndimage.uniform_filter(dot_a_train_data[:,1], size=filter_size)-126, label='dot_axis_train')
plt.plot(dot_a_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_model_data[:,1], size=filter_size)-126, label='dot_axis_valid_model')
plt.plot(dot_a_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_pose_data[:,1], size=filter_size)-126, label='dot_axis_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_axis_improv.png")
plt.gcf().clear()


######

plt.plot(dot_q_train_data[:,0], scipy.ndimage.uniform_filter(dot_q_train_data[:,1], size=filter_size), label='dot_quat_train')
plt.plot(dot_q_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_model_data[:,1], size=filter_size), label='dot_quat_valid_model')
plt.plot(dot_q_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_pose_data[:,1], size=filter_size), label='dot_quat_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_quat.png")
plt.gcf().clear()

plt.plot(dot_q_train_data[:,0], scipy.ndimage.uniform_filter(dot_q_train_data[:,1], size=filter_size)-126, label='dot_quat_train')
plt.plot(dot_q_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_model_data[:,1], size=filter_size)-126, label='dot_quat_valid_model')
plt.plot(dot_q_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_pose_data[:,1], size=filter_size)-126, label='dot_quat_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_quat_improv.png")
plt.gcf().clear()

plt.plot(dot_a_train_data[:,0], scipy.ndimage.uniform_filter(dot_a_train_data[:,1], size=filter_size), label='dot_axis_train')
plt.plot(dot_a_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_model_data[:,1], size=filter_size), label='dot_axis_valid_model')
plt.plot(dot_a_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_pose_data[:,1], size=filter_size), label='dot_axis_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_axis.png")
plt.gcf().clear()

plt.plot(dot_a_train_data[:,0], scipy.ndimage.uniform_filter(dot_a_train_data[:,1], size=filter_size)-126, label='dot_axis_train')
plt.plot(dot_a_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_model_data[:,1], size=filter_size)-126, label='dot_axis_valid_model')
plt.plot(dot_a_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_pose_data[:,1], size=filter_size)-126, label='dot_axis_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_axis_improv.png")
plt.gcf().clear()

########

plt.plot(dot_q_train_data[:,0], scipy.ndimage.uniform_filter(dot_q_train_data[:,1], size=filter_size), label='dot_quat_train')
plt.plot(dot_q_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_model_data[:,1], size=filter_size), label='dot_quat_valid_model')
plt.plot(dot_q_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_pose_data[:,1], size=filter_size), label='dot_quat_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_quat.png")
plt.gcf().clear()

plt.plot(dot_q_train_data[:,0], scipy.ndimage.uniform_filter(dot_q_train_data[:,1], size=filter_size)-126, label='dot_quat_train')
plt.plot(dot_q_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_model_data[:,1], size=filter_size)-126, label='dot_quat_valid_model')
plt.plot(dot_q_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_q_valid_pose_data[:,1], size=filter_size)-126, label='dot_quat_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_quat_improv.png")
plt.gcf().clear()

plt.plot(dot_a_train_data[:,0], scipy.ndimage.uniform_filter(dot_a_train_data[:,1], size=filter_size), label='dot_axis_train')
plt.plot(dot_a_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_model_data[:,1], size=filter_size), label='dot_axis_valid_model')
plt.plot(dot_a_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_pose_data[:,1], size=filter_size), label='dot_axis_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_axis.png")
plt.gcf().clear()

plt.plot(dot_a_train_data[:,0], scipy.ndimage.uniform_filter(dot_a_train_data[:,1], size=filter_size)-126, label='dot_axis_train')
plt.plot(dot_a_valid_model_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_model_data[:,1], size=filter_size)-126, label='dot_axis_valid_model')
plt.plot(dot_a_valid_pose_data[:,0], scipy.ndimage.uniform_filter(dot_a_valid_pose_data[:,1], size=filter_size)-126, label='dot_axis_valid_pose')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/car42/alex_car42_dot_axis_improv.png")
plt.gcf().clear()



import IPython; IPython.embed()