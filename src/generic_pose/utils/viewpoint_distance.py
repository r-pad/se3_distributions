# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:24:09 2018

@author: bokorn
"""

import numpy as np

def calcViewlossVec(size, sigma):
    band    = np.linspace(-1*size, size, 1 + 2*size, dtype=np.int16)
    vec     = np.linspace(-1*size, size, 1 + 2*size, dtype=np.float)
    prob    = np.exp(-1 * abs(vec) / sigma)
    prob    = prob / np.sum(prob)

    return band, prob

def label2Probs(angle, angle_bins = 360, band_width = 7, sigma=5):
    '''
    Returns three arrays for the viewpoint labels, one for each rotation axis.
    A label is given by angle
    :return:
    '''
    # Calculate object multiplier
    angle = int(angle)
    label = np.zeros(angle_bins, dtype=np.float)
    
    # calculate probabilities
    band, prob = calcViewlossVec(band_width, sigma)

    for i in band:
        ind = np.mod(angle + i + 360, 360)
        label[ind] = prob[i + band_width]

    return label

