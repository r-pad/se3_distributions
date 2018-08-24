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

filter_size = 50

#baseline_train = '/home/bokorn/results/vgg_reg_lr_1e-4_drill/2018-02-24_07-38-39/logs/train/events.out.tfevents.1519457965.compute-0-11.local'
#baseline_valid = '/home/bokorn/results/vgg_reg_lr_1e-4_drill/2018-02-24_07-38-39/logs/valid/events.out.tfevents.1519457965.compute-0-11.local'
#baseline_train_data = np.array(tb_utils.getSummaryData(baseline_train, ['err_quat'])['err_quat'])
#baseline_valid_data = np.array(tb_utils.getSummaryData(baseline_valid, ['err_quat'])['err_quat'])
#np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-4_drill.npz',
#         train = baseline_train_data,
#         valid = baseline_valid_data)
#res_baseline = np.load('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-4_drill.npz')
#lr4_train_data = res_baseline['train']
#lr4_valid_data = res_baseline['valid']

#lr2_path = '/home/bokorn/results/vgg_reg_lr_1e-2_drill/'
#lr2_train_path = glob.glob(lr2_path + '**/train/**/events.*', recursive=True)[-1]
#lr2_valid_path = glob.glob(lr2_path + '**/valid/**/events.*', recursive=True)[-1]
#lr2_train_data = np.array(tb_utils.getSummaryData(lr2_train_path, ['err_quat'])['err_quat'])
#lr2_valid_data = np.array(tb_utils.getSummaryData(lr2_valid_path, ['err_quat'])['err_quat'])
#np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-2_drill.npz',
#         train = lr2_train_data,
#         valid = lr2_valid_data)
#
#baseline_path = '/home/bokorn/results/vgg_reg_lr_1e-3_drill/'
#baseline_train_path = glob.glob(baseline_path + '**/train/**/events.*', recursive=True)[-1]
#baseline_valid_path = glob.glob(baseline_path + '**/valid/**/events.*', recursive=True)[-1]
#baseline_train_data = np.array(tb_utils.getSummaryData(baseline_train_path, ['err_quat'])['err_quat'])
#baseline_valid_data = np.array(tb_utils.getSummaryData(baseline_valid_path, ['err_quat'])['err_quat'])
#np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-3_drill.npz',
#         train = baseline_train_data,
#         valid = baseline_valid_data)
res_baseline = np.load('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-3_drill.npz')
baseline_train_data = res_baseline['train']
baseline_valid_data = res_baseline['valid']

##########################
### Learning Rate Test ###
##########################
#plt.plot(lr2_train_data[:,0], scipy.ndimage.uniform_filter(lr2_train_data[:,1], size=filter_size), label='lr2_train')
#plt.plot(lr2_valid_data[:,0], scipy.ndimage.uniform_filter(lr2_valid_data[:,1], size=filter_size), label='lr2_valid')
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='lr3_train')
#plt.plot(baseline_valid_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_data[:,1], size=filter_size), label='lr3_valid')
#
#plt.plot(lr4_train_data[:,0], scipy.ndimage.uniform_filter(lr4_train_data[:,1], size=filter_size), label='lr4_train')
#plt.plot(lr4_valid_data[:,0], scipy.ndimage.uniform_filter(lr4_valid_data[:,1], size=filter_size), label='lr4_valid')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/vgg_baselines.png")
#plt.gcf().clear()

################################
### Angle Test Drill LR 1e-4 ###
################################
#a45_train = '/home/bokorn/results/vgg_reg_drill_45deg/2018-03-25_00-48-01/logs/train/events.out.tfevents.1521938939.compute-0-7.local'
#a45_valid = '/home/bokorn/results/vgg_reg_drill_45deg/2018-03-25_00-48-01/logs/valid/events.out.tfevents.1521938939.compute-0-7.local'
#a45_train_data = np.array(tb_utils.getSummaryData(a45_train, ['mean_err'])['mean_err'])
#a45_valid_data = np.array(tb_utils.getSummaryData(a45_valid, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/test/imgs/vgg_reg_lr_1e-4_drill_45deg.npz',
#         train = a45_train_data,
#         valid = a45_valid_data)

#res_a45 = np.load('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-4_drill_45deg.npz')
#a45_train_data = res_a45['train']
#a45_valid_data = res_a45['valid']

#a90_train = '/home/bokorn/results/vgg_reg_drill_90deg/2018-03-24_23-47-47/logs/train/events.out.tfevents.1521935328.compute-0-7.local'
#a90_valid = '/home/bokorn/results/vgg_reg_drill_90deg/2018-03-24_23-47-47/logs/valid/events.out.tfevents.1521935328.compute-0-7.local'
#a90_train_data = np.array(tb_utils.getSummaryData(a90_train, ['mean_err'])['mean_err'])
#a90_valid_data = np.array(tb_utils.getSummaryData(a90_valid, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-4_drill_90deg.npz',
#         train = a90_train_data,
#         valid = a90_valid_data)
        
#res_a90 = np.load('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-4_drill_90deg.npz')
#a90_train_data = res_a90['train']
#a90_valid_data = res_a90['valid']
#
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_data[:,1], size=filter_size), label='baseline_valid')
#
#plt.plot(a45_train_data[:,0], scipy.ndimage.uniform_filter(a45_train_data[:,1], size=filter_size), label='a45_train')
#plt.plot(a45_valid_data[:,0], scipy.ndimage.uniform_filter(a45_valid_data[:,1], size=filter_size), label='a45_valid')
#
#plt.plot(a90_train_data[:,0], scipy.ndimage.uniform_filter(a90_train_data[:,1], size=filter_size), label='a90_train')
#plt.plot(a90_valid_data[:,0], scipy.ndimage.uniform_filter(a90_valid_data[:,1], size=filter_size), label='a90_valid')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/vgg_angle.png")
#plt.gcf().clear()

####################################
### Fixed Features Drill LR 1e-4 ###
####################################
#fixed_train = '/home/bokorn/results/vgg_reg_fixed_features_drill/2018-03-25_04-04-56/logs/train/events.out.tfevents.1521950735.compute-0-11.local'
#fixed_valid = '/home/bokorn/results/vgg_reg_fixed_features_drill/2018-03-25_04-04-56/logs/valid/events.out.tfevents.1521950735.compute-0-11.local'
#fixed_train_data = np.array(tb_utils.getSummaryData(fixed_train, ['mean_err'])['mean_err'])
#fixed_valid_data = np.array(tb_utils.getSummaryData(fixed_valid, ['mean_err'])['mean_err'])
#np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_fixed_features_drill.npz',
#         train = fixed_train_data,
#         valid = fixed_valid_data)
#        
#plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
#plt.plot(baseline_valid_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_data[:,1], size=filter_size), label='baseline_valid')
#
#plt.plot(fixed_train_data[:,0], scipy.ndimage.uniform_filter(fixed_train_data[:,1], size=filter_size), label='fixed_train')
#plt.plot(fixed_valid_data[:,0], scipy.ndimage.uniform_filter(fixed_valid_data[:,1], size=filter_size), label='fixed_valid')
#
#plt.legend()
#plt.savefig("/home/bokorn/results/result_imgs/drill/vgg_fixed_features.png")
#plt.gcf().clear()         

################################
### Angle Test Drill LR 1e-3 ###
################################
a90_path = '/home/bokorn/results/vgg_reg_lr_1e-3_drill_90deg/'
a90_train_path = glob.glob(a90_path + '**/train/**/events.*', recursive=True)[-1]
a90_valid_path = glob.glob(a90_path + '**/valid/**/events.*', recursive=True)[-1]
a90_train_data = np.array(tb_utils.getSummaryData(a90_train_path, ['mean_err'])['mean_err'])
a90_valid_data = np.array(tb_utils.getSummaryData(a90_valid_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-3_drill_90deg.npz',
         train = a90_train_data,
         valid = a90_valid_data)

a45_path = '/home/bokorn/results/vgg_reg_lr_1e-3_drill_45deg/'
a45_train_path = glob.glob(a45_path + '**/train/**/events.*', recursive=True)[-1]
a45_valid_path = glob.glob(a45_path + '**/valid/**/events.*', recursive=True)[-1]
a45_train_data = np.array(tb_utils.getSummaryData(a45_train_path, ['mean_err'])['mean_err'])
a45_valid_data = np.array(tb_utils.getSummaryData(a45_valid_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-3_drill_45deg.npz',
         train = a45_train_data,
         valid = a45_valid_data)

plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
plt.plot(baseline_valid_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_data[:,1], size=filter_size), label='baseline_valid')

plt.plot(a45_train_data[:,0], scipy.ndimage.uniform_filter(a45_train_data[:,1], size=filter_size), label='a45_train')
plt.plot(a45_valid_data[:,0], scipy.ndimage.uniform_filter(a45_valid_data[:,1], size=filter_size), label='a45_valid')

plt.plot(a90_train_data[:,0], scipy.ndimage.uniform_filter(a90_train_data[:,1], size=filter_size), label='a90_train')
plt.plot(a90_valid_data[:,0], scipy.ndimage.uniform_filter(a90_valid_data[:,1], size=filter_size), label='a90_valid')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/drill/vgg_angle.png")
plt.gcf().clear()

####################################
### Fixed Features Drill LR 1e-3 ###
####################################  
fixed_path = '/home/bokorn/results/vgg_reg_lr_1e-3_fixed_features_drill/'
fixed_train_path = glob.glob(fixed_path + '**/train/**/events.*', recursive=True)[-1]
fixed_valid_path = glob.glob(fixed_path + '**/valid/**/events.*', recursive=True)[-1]
fixed_train_data = np.array(tb_utils.getSummaryData(fixed_train_path, ['mean_err'])['mean_err'])
fixed_valid_data = np.array(tb_utils.getSummaryData(fixed_valid_path, ['mean_err'])['mean_err'])
np.savez('/home/bokorn/results/result_imgs/drill/vgg_reg_lr_1e-3_fixed_features_drill.npz',
         train = fixed_train_data,
         valid = fixed_valid_data)
        
plt.plot(baseline_train_data[:,0], scipy.ndimage.uniform_filter(baseline_train_data[:,1], size=filter_size), label='baseline_train')
plt.plot(baseline_valid_data[:,0], scipy.ndimage.uniform_filter(baseline_valid_data[:,1], size=filter_size), label='baseline_valid')

plt.plot(fixed_train_data[:,0], scipy.ndimage.uniform_filter(fixed_train_data[:,1], size=filter_size), label='fixed_train')
plt.plot(fixed_valid_data[:,0], scipy.ndimage.uniform_filter(fixed_valid_data[:,1], size=filter_size), label='fixed_valid')

plt.legend()
plt.savefig("/home/bokorn/results/result_imgs/drill/vgg_fixed_features.png")
plt.gcf().clear()      

import IPython; IPython.embed()