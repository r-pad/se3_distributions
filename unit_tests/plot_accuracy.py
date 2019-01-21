# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def main():
    results_dir = '/home/bokorn/results/test/'
    data = np.load('/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_trans_0.5/valid_metrics.npz')

    error_add = data['error_add']
    add_accuracy = []
    add_thresholds = np.linspace(0,0.1,1000)

    for th in add_thresholds:
        add_accuracy.append(np.mean(error_add < th))

    plt.plot(add_thresholds, add_accuracy)
    plt.xlabel('Average Distance Threshold in Meters (Non-Symetric)') 
    plt.ylabel('Accuracy') 
    plt.savefig(results_dir + 'add_accuracy.png')
    plt.gcf().clear()

    dist_top = data['dist_top']
    ang_accuracy = []
    ang_thresholds = np.linspace(0,180,1000)
    for th in ang_thresholds:
        ang_accuracy.append(np.mean(dist_top < th))

    plt.plot(ang_thresholds, ang_accuracy)
    plt.xlabel('Rotation Angle Threshold') 
    plt.ylabel('Accuracy') 
    plt.savefig(results_dir + 'ang_accuracy.png')
    plt.gcf().clear()
  
    import IPython; IPython.embed() 
    
if __name__=='__main__':
    main()

