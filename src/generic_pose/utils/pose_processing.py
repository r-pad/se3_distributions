# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
from quat_math import (quatDiff,
                       quatAngularDiff,
                       quaternion_multiply,
                       quaternion_inverse)

def viewpoint2Pose(q):
    delta_quat = np.array([.5,.5,.5,.5])
    q_flip = quaternion_inverse(q.copy())
    q_flip[2] *= -1
    return quaternion_multiply(delta_quat, q_flip)

def viewpointDiff(view_q, base_q):
    return quatDiff(viewpoint2Pose(view_q), base_q)

def viewpointDiffBatch(view_quats, base_quats):
    return np.array(list(map(viewpointDiff, view_quats, base_quats)))

def viewpointAngleDiff(q, base_vertices):
    q_adj = viewpoint2Pose(q)
    true_diff = []
    for v in base_vertices:
        true_diff.append(quatAngularDiff(q_adj, v))
    return np.array(true_diff)

