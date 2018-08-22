# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:39:11 2017

@author: bokorn
"""

import cv2
import numpy as np
from .image_preprocessing import transparentOverlay
from quat_math import quat2AxisAngle, quatAngularDiff

import scipy
from scipy import ndimage as ndi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import quat_math.transformations as tf_trans
from model_renderer.syscall_renderer import renderView

def get_figure():
  fig = plt.figure(num=0, figsize=(6, 4), dpi=150)
  fig.clf()
  return fig

def fig2rgb_array(fig, expand=False):
  fig.canvas.draw()
  buf = fig.canvas.tostring_rgb()
  ncols, nrows = fig.canvas.get_width_height()
  shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
  return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))
    
def makeHistogramImages(angle_error, angle_truth, num_bins = 180):
    error_bins = np.zeros(num_bins)
    count_bins = np.zeros(num_bins)
    for err, truth in zip(angle_error, angle_truth):
        idx = int(truth*num_bins/180)
        error_bins[idx] += err
        count_bins[idx] += 1

    bin_vals = np.arange(num_bins)*180/num_bins
    fig = get_figure()
    ax = fig.gca()
    ax.bar(bin_vals, error_bins/(count_bins + (count_bins==0)))
    mean_hist = np.transpose(fig2rgb_array(fig, expand=True), (0,3,1,2))
    fig.clear()
    ax = fig.gca()
    ax.bar(bin_vals, count_bins)
    count_hist = np.transpose(fig2rgb_array(fig, expand=True), (0,3,1,2))
    fig.clear()
    ax = fig.gca()
    ax.hist(angle_error, num_bins, density=True, facecolor='g', alpha=0.75, range=(0, 180))
    error_hist = np.transpose(fig2rgb_array(fig, expand=True), (0,3,1,2))
    
    return mean_hist, error_hist, count_hist
    
def makeDisplayImages(origin_imgs, query_imgs, 
                          true_class, true_quats,
                          est_class, est_quats,
                          num_bins = (50, 50, 25),
                          text_height = 0,
                          display_width = 1250+50,
                          norm_mean = np.array([0.485, 0.456, 0.406]),
                          norm_std = np.array([0.229, 0.224, 0.225])):

    origin_imgs = np.transpose(origin_imgs, (0,2,3,1))
    query_imgs = np.transpose(query_imgs, (0,2,3,1))
    origin_imgs = np.minimum(np.maximum(origin_imgs*norm_std + norm_mean, 0.0), 1.0)
    query_imgs = np.minimum(np.maximum(query_imgs*norm_std + norm_mean, 0.0), 1.0)
    
    n, h, w, c = origin_imgs.shape
    disp_w = display_width
    disp_h = (h*disp_w)//(2*w)
    class_h = display_width//num_bins[2]
    
    if(est_class is not None):
        disp_imgs = np.zeros((n, disp_h + 2*class_h+2 + text_height, disp_w, c), dtype='float32')
    else:
        disp_imgs = np.zeros((n, disp_h+text_height, disp_w, c), dtype='float32')
    
    for j in range(n):
        disp_imgs[j, :disp_h, :disp_w//2, :] = cv2.resize(origin_imgs[j], (disp_w//2, disp_h))
        disp_imgs[j, :disp_h, disp_w//2:, :] = cv2.resize(query_imgs[j], (disp_w//2, disp_h))

        true_axis, true_angle = quat2AxisAngle(true_quats[j])
        display_string_true = 'True Angle: {:.3f}, True Axis: {}'.format(true_angle*180/np.pi, true_axis)
        text_height = disp_h-40
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
        text_height += 20

        if(est_quats is not None):
            est_q = est_quats[j]/np.linalg.norm(est_quats[j])
            est_q *= np.sign(est_q[3])
            est_axis, est_angle = quat2AxisAngle(est_q)
             
            display_string_est  = 'Est Angle:  {:.3f}, Est Axis:  {}'.format(est_angle*180/np.pi, est_axis)        
            cv2.putText(disp_imgs[j], display_string_est, (10, text_height),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
            cv2.putText(disp_imgs[j], display_string_est, (10, text_height),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
            text_height += 20
        if(est_class is not None):
            true_img = makeProbabilityMosaic(true_class[j], num_bins, plt.cm.Blues)
            est_img = makeProbabilityMosaic(np.exp(est_class[j]), num_bins, plt.cm.Reds)

            true_img = cv2.resize(true_img, (display_width, class_h))
            est_img = cv2.resize(est_img, (display_width, class_h))
            
            disp_col = disp_h + 1
            disp_imgs[j, disp_col:disp_col+class_h, :, :] = true_img
            disp_col += class_h + 1
            disp_imgs[j, disp_col:disp_col+class_h, :, :] = est_img
        
            #quat_img = np.zeros(text_height, disp_w, c)
            true_max_idx = np.unravel_index(np.argmax(true_class[j]), num_bins)
            est_max_idx = np.unravel_index(np.argmax(np.exp(est_class[j])), num_bins)

            display_string_class = 'True Max Bin: {}, Est Max Bin: {}'.format(tuple(true_max_idx), tuple(est_max_idx))
            cv2.putText(disp_imgs[j], display_string_class, (10, text_height),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
            cv2.putText(disp_imgs[j], display_string_class, (10, text_height),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
            
    disp_imgs = np.flip(np.transpose(disp_imgs, (0,3,1,2)),1)

    return disp_imgs

def makeProbabilityMosaic(class_values, num_bins, cmap = plt.cm.Blues, max_idx=True, max_radius=2):
    if(type(max_idx) not in [list, tuple, np.ndarray]):
        max_idx = np.unravel_index(np.argmax(class_values), num_bins)

    class_prob = class_values.reshape(num_bins)
    sm = class_prob.transpose([0,2,1]).reshape(num_bins[0], num_bins[1]*num_bins[2]) / class_prob.sum()
    sm_img = get_colors(sm, cmap)[:,:,:3]
    mosaic_list = []
    for j, img_slice in enumerate(np.split(sm_img, num_bins[2], axis=1)):
        if j == max_idx[2]:
            max_slice = img_slice.copy()
            cv2.circle(max_slice, (max_idx[1], max_idx[0]), max_radius, (0,1,0), -1) 
            img_slice = max_slice
        mosaic_list.append(img_slice)
        mosaic_list.append(np.zeros((num_bins[0],1,3)))    
    
    return np.concatenate(mosaic_list[:-1], axis=1)
    
def makeImageMosaicModes(class_values, num_bins, cmap = plt.cm.Blues,  filter_sigma=1, max_window = 5, num_modes=10):
    class_prob = class_values.reshape(num_bins)
    max_idx = np.unravel_index(np.argmax(class_values), num_bins)
    filtered_values = ndi.filters.gaussian_filter(class_prob, sigma=filter_sigma, mode='wrap')
    max_values = ndi.maximum_filter(filtered_values, size=max_window, mode='wrap')
    local_max_idxs = np.nonzero(max_values == filtered_values)
    local_max_vals = filtered_values[local_max_idxs]
    local_max_idxs = np.array(local_max_idxs).T

    modes = np.concatenate([np.expand_dims(local_max_vals,axis=1), local_max_idxs], axis=1)
    modes = np.sort(modes, axis=0)[::-1]
    modes = modes[:num_modes, 1:].astype(int)
    #sm = np.roll(filtered_values.reshape(num_bins[0], num_bins[1]*num_bins[2]) / class_values.sum(), num_bins[2]//2 *  num_bins[1])
    sm = filtered_values.transpose([0,2,1]).reshape(num_bins[0], num_bins[1]*num_bins[2]) / filtered_values.sum()
    sm_img = get_colors(sm, cmap)[:,:,:3]
    mosaic_list = []
    for j, img_slice in enumerate(np.split(sm_img, num_bins[2], axis=1)):
        for m in modes[modes[:,2] == j]:
            max_slice = img_slice.copy()
            cv2.circle(max_slice, (m[1], m[0]), 2, (1,0,1), -1) 
            img_slice = max_slice
        if j == max_idx[2]:
            max_slice = img_slice.copy()
            cv2.circle(max_slice, (max_idx[1], max_idx[0]), 1, (0,1,0), -1) 
            img_slice = max_slice
            
        mosaic_list.append(img_slice)
        mosaic_list.append(np.zeros((num_bins[0],1,3)))    
    
    return np.concatenate(mosaic_list[:-1], axis=1)

def renderQuaternions(origin_imgs, query_imgs, 
                      true_quats, est_quats,
                      model_files, origin_quats=None,
                      display_width = 1250, 
                      norm_mean = np.array([0.485, 0.456, 0.406]),
                      norm_std = np.array([0.229, 0.224, 0.225]), 
                      camera_dist = 2, render_gripper = False):

    origin_imgs = np.transpose(origin_imgs, (0,2,3,1))
    query_imgs = np.transpose(query_imgs, (0,2,3,1))
    origin_imgs = np.minimum(np.maximum(origin_imgs*norm_std + norm_mean, 0.0), 1.0)
    query_imgs = np.minimum(np.maximum(query_imgs*norm_std + norm_mean, 0.0), 1.0)
    
    n, h, w, c = origin_imgs.shape
    disp_w = 3*(display_width//3)
    disp_h = (h*disp_w)//(3*w)
    
    disp_imgs = np.zeros((n, disp_h, disp_w, c), dtype='float32')

    if(n > 0 and model_files[0].split('/')[-1] == 'electric_gripper_w_fingers.obj'):
        init_q = tf_trans.quaternion_inverse(tf_trans.quaternion_about_axis(3*np.pi/4, [2,-2,1]))
    else:
        init_q = np.array([ 0.0,  0.0,  0.0,  1.0])
        
    for j in range(n):
        est_q = est_quats[j]/np.linalg.norm(est_quats[j])
        est_q *= np.sign(est_q[3])
        diff = quatAngularDiff(est_q, true_quats[j])

        if(origin_quats is None):
            origin_quats = [np.array([0,0,0,1])]

        #q = tf_trans.quaternion_multiply(init_q, tf_trans.quaternion_multiply(est_q, origin_quats[j]))
        q = tf_trans.quaternion_multiply(est_q, origin_quats[j])
        render_img = renderView(model_files[j], [tf_trans.quaternion_multiply(init_q,q)], camera_dist, standard_lighting=True)
        if(len(render_img) == 0):
            continue;
        render_img = transparentOverlay(render_img[0])/255.0

        disp_imgs[j, :, :disp_w//3, :] = cv2.resize(origin_imgs[j], (disp_w//3, disp_h))
        disp_imgs[j, :, disp_w//3:2*(disp_w//3), :] = cv2.resize(query_imgs[j], (disp_w//3, disp_h))
        disp_imgs[j, :, 2*(disp_w//3):, :] = cv2.resize(render_img, (disp_w//3, disp_h))
        
        true_axis, true_angle = quat2AxisAngle(true_quats[j])
        
        display_string_true = 'True Angle: {:.3f}, True Axis: {}'.format(true_angle*180/np.pi, true_axis)
        text_height = disp_h-40
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
        text_height += 20
        
        est_axis, est_angle = quat2AxisAngle(est_q)
        display_string_est  = 'Est Angle:  {:.3f}, Est Axis:  {}, Error: {}'.format(est_angle*180/np.pi, est_axis, diff*180/np.pi)
        cv2.putText(disp_imgs[j], display_string_est, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_est, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
        text_height += 20
    
    disp_imgs = np.flip(np.transpose(disp_imgs, (0,3,1,2)), 1)
    return disp_imgs
