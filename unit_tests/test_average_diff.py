# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:13:43 2018

@author: bokorn
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quat_math import quatAngularDiff, quat2AxisAngle
from generic_pose.utils.transformations import random_quaternion

num_diffs = 100000
angles = []
single_angles = []
axis_angles = []

for j in range(num_diffs):
    q1 = random_quaternion()
    q2 = random_quaternion()
    angles += [quatAngularDiff(q1,q2)]
    single_angles += [quatAngularDiff(q1,np.array([0,0,0,1]))]
    axis_angles += [quat2AxisAngle(q1)[1]]


n, bins, patches = plt.hist(np.array(angles)*180/np.pi, 1000, density=True, facecolor='g', alpha=0.75)
plt.savefig("/home/bokorn/results/test/imgs/diff.png")
plt.gcf().clear()
n, bins, patches = plt.hist(np.array(single_angles)*180/np.pi, 1000, density=True, facecolor='g', alpha=0.75)
plt.savefig("/home/bokorn/results/test/imgs/single_angle.png")
plt.gcf().clear()
n, bins, patches = plt.hist(np.array(axis_angles)*180/np.pi, 1000, density=True, facecolor='g', alpha=0.75)
plt.savefig("/home/bokorn/results/test/imgs/angle.png")

import IPython; IPython.embed()
    
