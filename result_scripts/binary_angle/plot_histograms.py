# -*- coding: utf-8 -*-
"""
Created on Someday Sometime

@author: bokorn
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

title_name = 'Class Validation'
data_folder = 'shapenet'
file_prefix = 'valid_class_0'

hist_data=np.load('/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/{}/{}_histograms.npz'.format(data_folder, file_prefix))
edges = hist_data['hist_edges']

#####################
### Accuracy Plot ###
#####################

accuracy_hist = hist_data['accuracy_hist']

plt.bar(edges[:-1], accuracy_hist, width=edges[1]-edges[0])
plt.axvline(45, color='r')

plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Accuracy w/ 0.5 Threshold')
plt.title('{} Set'.format(title_name))

plt.savefig('/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/{}.png'.format(file_prefix))
plt.gcf().clear()

#####################
#### Output Plot ####
#####################

data = hist_data['data']
binned_output = []

for j in range(len(edges)-1):
    binned_output.append(data[1,np.bitwise_and(data[0] >= edges[j], data[0] < edges[j+1])])

bin_width = edges[1]-edges[0]

bp = plt.boxplot(binned_output, 0, '.')#flierprops={'size':1,'marker':'.'})
for flier in bp['fliers']:
    flier.set(marker='.', markersize=1)

plt.xticks(np.arange(0,180,25)/bin_width, np.arange(0,180,25))
plt.axvline(45/bin_width, color='r')
plt.axhline(0.5, color='g')

plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Mean Output')
plt.title('{} Set'.format(title_name))

plt.savefig('/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/output_{}.png'.format(file_prefix))
plt.gcf().clear()

##################
#### PR Curve ####
##################

count_hist = hist_data['count_hist']
count_hist[0] = 0
plt.bar(edges[:-1], count_hist, width=edges[1]-edges[0])
plt.axvline(45, color='r')

plt.xlabel('Angle Difference (degrees)')
plt.ylabel('Number of Samples')
plt.title('{} Set'.format(title_name))

plt.savefig('/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/counts_{}.png'.format(file_prefix))
plt.gcf().clear()

truth = np.bitwise_and(data[0,:] < 45, data[0,:] > 0)
precision = []
recall = []
for v in np.linspace(0,1,101):
    selected = np.bitwise_and(data[1,:] > v, data[0,:] > 0)
    tp = np.sum(np.bitwise_and(truth, selected))
    p = tp/np.sum(selected)
    r = tp/np.sum(truth)
    precision.append(p)
    recall.append(r)

selected = data[1,:] > .5
tp = np.sum(np.bitwise_and(truth, selected))
p = tp/np.sum(selected)
r = tp/np.sum(truth)
print('PR at 0.5')
print('Precision: {:.3f}'.format(p))
print('Recall: {:.3f}'.format(r))

plt.plot(recall, precision)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('{} Set'.format(title_name))

plt.savefig('/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/pr_{}.png'.format(file_prefix))
plt.gcf().clear()
   
non_boundary = np.bitwise_not(np.isclose(data[0,:],45,atol=5))
truth = np.bitwise_and(np.bitwise_and(data[0,:] < 45, non_boundary), data[0,:] > 0)
precision = []
recall = []
for v in np.linspace(0,1,101):
    selected = np.bitwise_and(np.bitwise_and(data[1,:] > v, non_boundary), data[0,:] > 0)
    tp = np.sum(np.bitwise_and(truth, selected))
    p = tp/np.sum(selected)
    r = tp/np.sum(truth)
    precision.append(p)
    recall.append(r)
    
selected =np.bitwise_and(np.bitwise_and(data[1,:] > .5, non_boundary), data[0,:] > 0)
tp = np.sum(np.bitwise_and(truth, selected))
p = tp/np.sum(selected)
r = tp/np.sum(truth)
print('PR at 0.5 w/o Boundary')
print('Precision: {:.3f}'.format(p))
print('Recall: {:.3f}'.format(r))

plt.plot(recall, precision)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('{} Set w/o Boundry'.format(title_name))

plt.savefig('/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/pr_nob_{}.png'.format(file_prefix))
plt.gcf().clear()

