# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def savePlot(err, baseline, max_threshold, results_prefix, xlabel="Threshold"):
    accuracy = []
    accuracy_baseline = []
    thresholds = np.linspace(0,max_threshold,1000)

    for th in thresholds:
        accuracy.append(np.mean(err < th))
        accuracy_baseline.append(np.mean(baseline < th))

    plt.plot(thresholds, accuracy)
    plt.plot(thresholds, accuracy_baseline, c='r')
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy') 
    plt.savefig(results_prefix + 'accuracy_{:.2f}_vs_{:.2f}.png'.format(np.mean(accuracy)*100, 
            np.mean(accuracy_baseline)*100))
    plt.gcf().clear()


def plotAccuracy(data, results_prefix):
    savePlot(data['error_add'], data['error_add_pcnn'], 0.1, results_prefix = results_prefix + 'add_', 
            xlabel = 'Average Distance Threshold in Meters (Non-Symetric)')
    savePlot(data['dist_top'], data['dist_pcnn'], 180, results_prefix = results_prefix + 'ang_', 
            xlabel = 'Rotation Angle Threshold')
    
