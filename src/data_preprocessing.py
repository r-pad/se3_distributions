# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:24:09 2018

@author: bokorn
"""

import numpy as np
import cv2

import transformations as tf


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

def resizeAndPad(img, size, padColor=255.0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
    
def transparentOverlay(foreground, background=None, pos=(0,0),scale = 1):
    """
    :param foreground: transparent Image (BGRA)
    :param background: Input Color Background Image
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Overlayed image
    """
    
    if(scale != 1):
        foreground = cv2.resize(foreground, None,fx=scale,fy=scale)

    alpha = foreground[:,:,3:].astype(float)/255    
    
    if(background is None):
        background = alpha*foreground[:,:,:3] + 255.0*(1.0-alpha)
    else:
        h,w,_ = foreground.shape
        rows,cols,_ = background.shape
        
        y0,x0 = pos[0],pos[1]
        y1 = min(y0+h, rows)
        x1 = min(x0+w, cols)
        background[y0:y1,x0:x1,:] = alpha*foreground[:,:,:3] + (1.0-alpha)*background[y0:y1,x0:x1,:]
    
    return background

def randomQuatNear(init_quat, max_orientation_offset):
    offset_axis = np.random.randn(3)
    offset_axis /= np.linalg.norm(offset_axis)
    offset_angle = max_orientation_offset * np.random.rand()
    offset_quat = tf.quaternion_about_axis(offset_angle, offset_axis)
    near_quat = tf.quaternion_multiply(init_quat, offset_quat)
    return near_quat, offset_quat

def uniformRandomQuaternion():
    u = np.random.rand(3)
    return uniform2Quat(u)

def uniform2Quat(u):
    r1 = np.sqrt(1-u[0])
    r2 = np.sqrt(u[0])
    theta1 = 2.0*np.pi*u[1]
    theta2 = 2.0*np.pi*u[2]
    return np.array([r1*np.sin(theta1), r1*np.cos(theta1), r2*np.sin(theta2), r2*np.cos(theta2)])

def quat2Uniform(q):
    u1 = (q[2]**2 + q[3]**2)
    u2 = np.arctan2(q[0],q[1])/(2.0*np.pi)
    u3 = np.arctan2(q[2],q[3])/(2.0*np.pi)
    return np.array([u1, u2, u3])