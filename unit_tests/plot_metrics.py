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
    valid_dataset_metrics = np.load("/home/bokorn/results/ycb_finetune/01_002_master_chef_can/metrics/valid_metrics.npz")
    valid_performance_metrics = np.load("/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_trans_0.5/valid_metrics.npz")
    train_dataset_metrics = np.load("/home/bokorn/results/ycb_finetune/01_002_master_chef_can/metrics/train_metrics.npz")
    train_performance_metrics = np.load("/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_1.0/t_images/train_w_renders_metrics.npz")
    #import IPython; IPython.embed() 

    train_performance_metrics = dict(train_performance_metrics)
    #train_performance_metrics['rank_gt'] = np.delete(train_performance_metrics['rank_gt'], 5302)
    train_performance_metrics['rank_gt']  = train_performance_metrics['rank_gt'][:18566]

    results_dir = '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/random_renders_aug_all_trans_0.5/'
    
    plt.scatter(valid_dataset_metrics['min_distance'], valid_performance_metrics['rank_gt'], s=1, alpha = 0.1)
    v_hist_count, v_edges = np.histogram(valid_dataset_metrics['min_distance'], bins=100)
    v_hist_rank, v_edges = np.histogram(valid_dataset_metrics['min_distance'], bins=100, 
                                    weights=valid_performance_metrics['rank_gt'])
    v_hist = v_hist_rank/v_hist_count
    plt.plot(v_edges[1:], v_hist, c='c')
    plt.xlabel('Min distance to grid indice')
    plt.ylabel('Rank of GT Bin')
    plt.savefig(results_dir + 'distance.png')
    plt.gcf().clear()
   

    t_hist_count, t_edges = np.histogram(train_dataset_metrics['occlusion'], bins=100)
    t_hist_rank, t_edges = np.histogram(train_dataset_metrics['occlusion'], bins=100, 
                                    weights=train_performance_metrics['rank_gt'])
    t_hist = t_hist_rank/t_hist_count
    v_hist_count, v_edges = np.histogram(valid_dataset_metrics['occlusion'], bins=100)
    v_hist_rank, v_edges = np.histogram(valid_dataset_metrics['occlusion'], bins=100, 
                                    weights=valid_performance_metrics['rank_gt'])
    v_hist = v_hist_rank/v_hist_count

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    ax2.hist(train_dataset_metrics['occlusion'], alpha=0.2, color='r')
    ax2.hist(valid_dataset_metrics['occlusion'], alpha=0.2, color='g')
    ax1.scatter(train_dataset_metrics['occlusion'], train_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='r')
    ax1.scatter(valid_dataset_metrics['occlusion'], valid_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='g')
    ax1.plot(t_edges[1:], t_hist, c='m')
    ax1.plot(v_edges[1:], v_hist, c='c')
    plt.xlabel('Percent Occluded')
    plt.ylabel('Rank of GT Bin')
    plt.savefig(results_dir + 'occlusion.png')
    fig.clear()
    plt.gcf().clear()

    fig, ax1 = plt.subplots()
    sc = ax1.scatter(valid_dataset_metrics['occlusion'], valid_dataset_metrics['min_distance'], 
            s=1, alpha = 0.1, c=valid_performance_metrics['rank_gt'])
    plt.colorbar(sc, label='Rank of GT Bin')
    plt.xlabel('Percent Occluded')
    plt.ylabel('Min distance to grid indice')
    plt.savefig(results_dir + 'occlusion_vs_dist.png')
    fig.clear()
    plt.gcf().clear()
    
    fig, ax1 = plt.subplots()
    sc = ax1.scatter(valid_dataset_metrics['occlusion'], valid_dataset_metrics['min_distance'], 
            s=1, alpha = 0.1, c=np.arange(len(valid_dataset_metrics['occlusion'])))
    plt.colorbar(sc, label='Dataset Idx')
    plt.xlabel('Percent Occluded')
    plt.ylabel('Min distance to grid indice')
    plt.savefig(results_dir + 'occlusion_vs_dist_idx.png')
    fig.clear()
    plt.gcf().clear()



    t_hist_count, t_edges = np.histogram(train_dataset_metrics['mean_h'], bins=100)
    t_hist_rank, t_edges = np.histogram(train_dataset_metrics['mean_h'], bins=100, 
                                    weights=train_performance_metrics['rank_gt'])
    t_hist = t_hist_rank/t_hist_count
    v_hist_count, v_edges = np.histogram(valid_dataset_metrics['mean_h'], bins=100)
    v_hist_rank, v_edges = np.histogram(valid_dataset_metrics['mean_h'], bins=100, 
                                    weights=valid_performance_metrics['rank_gt'])
    v_hist = v_hist_rank/v_hist_count

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    ax2.hist(train_dataset_metrics['mean_h'], alpha=0.2, color='r')
    ax2.hist(valid_dataset_metrics['mean_h'], alpha=0.2, color='g')
    ax1.scatter(train_dataset_metrics['mean_h'], train_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='r')
    ax1.scatter(valid_dataset_metrics['mean_h'], valid_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='g')
    ax1.plot(t_edges[1:], t_hist, c='m')
    ax1.plot(v_edges[1:], v_hist, c='c')
    plt.xlabel('Mean Hue')
    plt.ylabel('Rank of GT Bin')
    plt.savefig(results_dir + 'mean_h.png')
    fig.clear()
    plt.gcf().clear()
    
    t_hist_count, t_edges = np.histogram(train_dataset_metrics['mean_s'], bins=100)
    t_hist_rank, t_edges = np.histogram(train_dataset_metrics['mean_s'], bins=100,
                                    weights=train_performance_metrics['rank_gt'])
    t_hist = t_hist_rank/t_hist_count
    v_hist_count, v_edges = np.histogram(valid_dataset_metrics['mean_s'], bins=100)
    v_hist_rank, v_edges = np.histogram(valid_dataset_metrics['mean_s'], bins=100,
                                    weights=valid_performance_metrics['rank_gt'])
    v_hist = v_hist_rank/v_hist_count

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    ax2.hist(train_dataset_metrics['mean_s'], alpha=0.2, color='r')
    ax2.hist(valid_dataset_metrics['mean_s'], alpha=0.2, color='g')
    ax1.scatter(train_dataset_metrics['mean_s'], train_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='r')
    ax1.scatter(valid_dataset_metrics['mean_s'], valid_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='g')
    ax1.plot(t_edges[1:], t_hist, c='m')
    ax1.plot(v_edges[1:], v_hist, c='c')
    plt.xlabel('Mean Saturation')
    plt.ylabel('Rank of GT Bin')
    plt.savefig(results_dir + 'mean_s.png')
    fig.clear()
    plt.gcf().clear()
    
    t_hist_count, t_edges = np.histogram(train_dataset_metrics['mean_v'], bins=100) 
    t_hist_rank, t_edges = np.histogram(train_dataset_metrics['mean_v'], bins=100, 
                               weights=train_performance_metrics['rank_gt'])
    t_hist = t_hist_rank/t_hist_count
    v_hist_count, v_edges = np.histogram(valid_dataset_metrics['mean_v'], bins=100) 
    v_hist_rank, v_edges = np.histogram(valid_dataset_metrics['mean_v'], bins=100, 
                               weights=valid_performance_metrics['rank_gt'])
    v_hist = v_hist_rank/v_hist_count

    fig, ax1 = plt.subplots()
    ax2 = plt.twinx()
    ax2.hist(train_dataset_metrics['mean_v'], alpha=0.2, color='r')
    ax2.hist(valid_dataset_metrics['mean_v'], alpha=0.2, color='g')
    ax1.scatter(train_dataset_metrics['mean_v'], train_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='r')
    ax1.scatter(valid_dataset_metrics['mean_v'], valid_performance_metrics['rank_gt'], s=1, alpha = 0.1, c='g')
    ax1.plot(t_edges[1:], t_hist, c='m')
    ax1.plot(v_edges[1:], v_hist, c='c')
    plt.xlabel('Mean Value')
    plt.ylabel('Rank of GT Bin')
    plt.savefig(results_dir + 'mean_v.png')
    fig.clear()
    plt.gcf().clear()

    import IPython; IPython.embed() 
    
if __name__=='__main__':
    main()

