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

def pose2Viewpoint(q):
    delta_quat = np.array([-.5,-.5,-.5,.5])    
    q_flip = quaternion_multiply(delta_quat, q.copy())
    q_flip[2] *= -1
    return quaternion_inverse(q_flip)

def viewpoint2Pose(q):
    delta_quat = np.array([.5,.5,.5,.5])
    q_flip = quaternion_inverse(q.copy())
    q_flip[2] *= -1
    return quaternion_multiply(delta_quat, q_flip)

def quatDiffBatch(view_quats, base_quats):
    return np.array(list(map(quatDiff, view_quats, base_quats)))

def quatAngularDiffBatch(view_quats, base_quats):
    return np.array(list(map(quatAngularDiff, view_quats, base_quats)))


