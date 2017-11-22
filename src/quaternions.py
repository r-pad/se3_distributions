# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 01:40:50 2017

@author: bokorn
"""
import numpy as np
from pose_renderer import (objCentenedCameraPos, 
                           camPosToQuaternion, 
                           camRotQuaternion, 
                           quaternionProduct)
                           

    
def euler2quat(azimuth, elevation, tilt):
    cy = np.cos(azimuth * 0.5)
    sy = np.sin(azimuth * 0.5)
    cr = np.cos(tilt * 0.5)
    sr = np.sin(tilt * 0.5)
    cp = np.cos(elevation * 0.5)
    sp = np.sin(elevation * 0.5)
    return np.array([cy * cr * cp + sy * sr * sp, 
                     cy * sr * cp - sy * cr * sp,
                     cy * cr * sp + sy * sr * cp,
                     sy * cr * cp - cy * sr * sp])

def mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])
    
def conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])
