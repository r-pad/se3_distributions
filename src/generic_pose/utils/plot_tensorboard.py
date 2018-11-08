# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import numpy as np
import generic_pose.utils.tensorboard_utils as tb_utils

def ewma(data, alpha):
    alpha_rev = 1-alpha
    out = np.zeros(data.shape)
    N = data.shape[0]
    out[0] = data[0]
    for j in range(1,N):
        out[j] = alpha*data[j] + alpha_rev*out[j-1]
    return out

def plotScalar(tag, data, image_prefix, smoothing=0.0, wall_time = None,
               labels=None, xlabel=None, ylabel=None, title=None, 
               max_step=None):
    plt.gcf().clear()
    fig = plt.figure()
    ax = plt.subplot(111)
    print(ylabel)
    cell_text = []
    for d, lbl in zip(data, labels):
        if(max_step is not None):
            end = np.where(d[:,0] > max_step)[0]
            if end.size > 0:
                end = end[0]
            else:
                end = None
        else:
            end = None
        p = ax.plot(d[:end,0], ewma(d[:end,1], 1-smoothing), label=lbl, linewidth=0.5)
        ax.plot(d[:end,0], d[:end,1], color = p[-1].get_color(), alpha=0.1)
        print(lbl)
        print('Final Raw: {}'.format(d[-1,1]))                                      
        print('Final Smoothed: {}'.format(ewma(d[:end,1], 1-smoothing)[-1]))
        cell_text.append(['{:.2f}'.format(d[-1,1]) + ' ({:.2f})'.format(ewma(d[:end,1], 1-smoothing)[-1])])
    print()
    
    if(labels is not None):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=5)
    if(wall_time is not None):
        ax_wt = ax.twiny()
        wall_time = np.array(wall_time)[:end]
        wt_hrs = (wall_time - wall_time[0])*np.array([1,1/3600])
        ax_wt.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

        ax_wt.set_xlim([0, wt_hrs[-1,1]])
        ax_wt.set_xlabel("Relative Time (hrs)")
        #ax_wt.set_xlim(ax.get_xlim())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    fig.savefig(image_prefix+tag+'_plot.png')
    
    plt.gcf().clear()
    fig = plt.figure()
    ax = plt.subplot(111)
    #fig.patch.set_visible(False)
    ax.axis('off')
    #ax.axis('tight')
    plt.table(cellText=cell_text,
              rowLabels=labels,
              colLabels=[ylabel],
              loc='center',
              )
    #fig.tight_layout()
    plt.title(title)
    fig.savefig(image_prefix+tag+'_table.png')

def main():
    import os
    from argparse import ArgumentParser
    from itertools import cycle
    
    parser = ArgumentParser()
    parser.add_argument('--log_dirs', nargs='+', required=True)
    parser.add_argument('--log_labels', nargs='+')
    parser.add_argument('--scalar_tags', nargs='+', required=True)
    parser.add_argument('--xlabels', nargs='+', default=['Steps'])
    parser.add_argument('--ylabels', nargs='+', default=None)
    parser.add_argument('--titles', nargs='+', default=[''])
    parser.add_argument('--max_step', type=int, default=None)
    parser.add_argument('--wall_time', dest='wall_time', action='store_true')
    parser.add_argument('--image_prefix', type=str, default='')
    parser.add_argument('--smoothing', type=float, default=0.9)
 
    args = parser.parse_args()
    data = {}
    wall_time = None
    for tag in args.scalar_tags:
        data[tag] = []

    for log_dir in args.log_dirs:
        path = glob.glob(log_dir + '/**/events.*', recursive=True)[-1]
        log_data = tb_utils.getSummaryData(path, args.scalar_tags)
        for tag in args.scalar_tags:
            data[tag].append(np.array(log_data[tag]))
        if(args.wall_time and wall_time is None):
            wall_time = tb_utils.getWallTime(path)

    if(args.ylabels is None):
        args.ylabels = args.scalar_tags

    for tag, xlabel, ylabel, title in zip(args.scalar_tags, cycle(args.xlabels),
                                          cycle(args.ylabels), cycle(args.titles)):
        plotScalar(tag, data[tag], args.image_prefix, smoothing=args.smoothing, 
                   wall_time=wall_time, labels=args.log_labels, 
                   xlabel=xlabel, ylabel=ylabel, title=title,
                   max_step=args.max_step) 

if __name__=='__main__':
    main()

