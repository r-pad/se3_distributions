# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:39:11 2017

@author: bokorn
"""

import cv2
import numpy as np
from .data_preprocessing import quat2AxisAngle, index2Quat, transparentOverlay, indexAngularDiff, quatAngularDiff

import scipy
from scipy import ndimage as ndi

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import transformations as tf_trans
from .pose_renderer import renderView

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
    
def makeEulerDisplayImages(origin_imgs, query_imgs, 
                     true_d0, true_d1, true_d2, true_quats,
                     est_d0, est_d1, est_d2, est_quats,
                     bin_height = 10, text_height = 0,
                     norm_mean = np.array([0.485, 0.456, 0.406]),
                     norm_std = np.array([0.229, 0.224, 0.225])):

    origin_imgs = np.transpose(origin_imgs, (0,2,3,1))
    query_imgs = np.transpose(query_imgs, (0,2,3,1))
    origin_imgs = np.minimum(np.maximum(origin_imgs*norm_std + norm_mean, 0.0), 1.0)
    query_imgs = np.minimum(np.maximum(query_imgs*norm_std + norm_mean, 0.0), 1.0)
    
    n, h, w, c = origin_imgs.shape
    disp_w = 2*true_d0.shape[-1]
    disp_h = (h*disp_w)//(2*w)
    
    if(est_d0 is not None and est_d1 is not None and est_d2 is not None):
        disp_imgs = np.zeros((n, disp_h+6*bin_height+2+text_height, disp_w, c), dtype='float32')
    else:
        disp_imgs = np.zeros((n, disp_h+text_height, disp_w, c), dtype='float32')
    
    for j in range(n):
        disp_imgs[j, :disp_h, :disp_w//2, :] = cv2.resize(origin_imgs[j], (disp_w//2, disp_h))
        disp_imgs[j, :disp_h, disp_w//2:, :] = cv2.resize(query_imgs[j], (disp_w//2, disp_h))

        true_axis, true_angle = quat2AxisAngle(true_quats[j])
        display_string_true = 'True Angle: {:.3f}, True Axis: {}'.format(true_angle*180/np.pi, true_axis)
        cv2.putText(disp_imgs[j], display_string_true, (10, disp_h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_true, (10, disp_h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
                    
        if(est_quats is not None):
            est_q = est_quats[j]/np.linalg.norm(est_quats[j])
            est_q *= np.sign(est_q[3])
            est_angle = 2*np.arccos(est_q[3])
            est_axis = est_q[:3]/np.sin(est_angle/2.0)               
            display_string_est  = 'Est Angle:  {:.3f}, Est Axis:  {}'.format(est_angle*180/np.pi, est_axis)        
            cv2.putText(disp_imgs[j], display_string_est, (10, disp_h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
            cv2.putText(disp_imgs[j], display_string_est, (10, disp_h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)

        if(est_d0 is not None and est_d1 is not None and est_d2 is not None):
            disp_col = disp_h + text_height
            for k, v in enumerate([true_d0[j], est_d0[j], true_d1[j], est_d1[j], true_d2[j], est_d2[j]]):
                if(k%2 == 1):
                    sm = np.exp(v)
                    cmap = plt.cm.Blues
    
                else:
                    sm = v
                    cmap = plt.cm.Reds
    
                sm = sm / np.sum(sm)
                sm_img = get_colors(sm, cmap)[:,:3]
                sm_img = np.repeat(np.tile(sm_img, (bin_height, 1, 1)), 2, axis=1)
                disp_imgs[j, disp_col:disp_col+bin_height, :, :] = sm_img            
                disp_col += bin_height
                if(k%2 == 1):
                    disp_col += 1
        
            #quat_img = np.zeros(text_height, disp_w, c)
    disp_imgs = np.transpose(disp_imgs, (0,3,1,2))

    return disp_imgs
    
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
            
    disp_imgs = np.transpose(disp_imgs, (0,3,1,2))

    return disp_imgs

#def imageMosaic(class_values, num_bins, cmap = plt.cm.Blues):
#    class_prob = class_values.reshape(num_bins)
#    sm = class_prob.transpose([0,2,1]).reshape(num_bins[0], num_bins[1]*num_bins[2]) / class_prob.sum()
#    sm_img = get_colors(sm, cmap)[:,:,:3]
#    
#    mosaic_list = np.split(sm_img, num_bins[2], axis=1)
#    
#    return mosaic_list

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
    
#def makeProbabilityMosaic(class_values, num_bins, cmap = plt.cm.Blues, 
#                          max_idx=True, max_color=(0,1,0), max_radius = 2, mosaic_list = None):
#    
#    if(mosaic_list is None):
#        mosaic_list = imageMosaic(class_values, num_bins, cmap = plt.cm.Blues)
#    
#    if(max_idx is not None and max_idx == True):    
#        max_idx = np.unravel_index(np.argmax(class_values), num_bins)
#    
#    if(max_idx):
#        max_slice = mosaic_list[max_idx[2]].copy()
#        cv2.circle(max_slice, (max_idx[1], max_idx[0]), max_radius, max_color, -1) 
#        mosaic_list[max_idx[2]] = max_slice
#        
#    for img_slice in mosaic_list:
#        mosaic_list.append(img_slice)
#        mosaic_list.append(np.zeros((num_bins[0],1,3)))   
    
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

def index2Axis(idx, num_bins):
    R = tf_trans.quaternion_matrix(index2Quat(idx, num_bins))
    return R[0,:3], R[1,:3], R[2,:3]
    
def makeModeImages(class_values, num_bins, filter_sigma = 1, max_window=5):
    disp_imgs = []
    for j, vals in enumerate(class_values):
        disp_imgs.append(renderModes(np.exp(vals), num_bins, filter_sigma=filter_sigma, max_window = max_window))

    disp_imgs = np.transpose(np.array(disp_imgs), (0,3,1,2))
    return disp_imgs

def renderModes(class_prob, num_bins, filter_sigma=0, max_window = 5, num_modes=10):
    class_prob = class_prob.reshape(num_bins)

    filtered_values = ndi.filters.gaussian_filter(class_prob, sigma=filter_sigma, mode='wrap')
    max_values = ndi.maximum_filter(filtered_values, size=max_window, mode='wrap')
    local_max_idxs = np.nonzero(max_values == filtered_values)
    local_max_vals = filtered_values[local_max_idxs]
    local_max_idxs = np.array(local_max_idxs).T

    modes = np.concatenate([np.expand_dims(local_max_vals,axis=1), local_max_idxs], axis=1)
    modes = np.sort(modes, axis=0)[::-1]
    modes = modes[:num_modes]
    fig = plt.figure()#get_figure()
    ax = fig.gca(projection='3d')

    xs = []
    ys = []
    zs = []
    
    for val_idx in modes:
        p = val_idx[0]
        idx = val_idx[1:]
        x_axis, y_axis, z_axis = index2Axis(idx, num_bins)

        xs.append(p*x_axis)
        ys.append(p*y_axis)
        zs.append(p*z_axis)

    xs = np.array(xs).T
    ys = np.array(ys).T
    zs = np.array(zs).T
    
    ax.quiver(0, 0, 0, xs[0], xs[1], xs[2], colors='red', arrow_length_ratio=0)
    ax.quiver(0, 0, 0, ys[0], ys[1], ys[2], colors='green', arrow_length_ratio=0)
    ax.quiver(0, 0, 0, zs[0], zs[1], zs[2], colors='blue', arrow_length_ratio=0)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    return fig2rgb_array(fig)

from numpy import radians as rad
from matplotlib.patches import Arc, RegularPolygon
def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black'):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort

def makeDistributionImages(class_values, num_bins):
    disp_imgs = []
    for j, vals in enumerate(class_values):
        disp_imgs.append(renderDistribution(np.exp(vals), num_bins))

    disp_imgs = np.transpose(np.array(disp_imgs), (0,3,1,2))
    return disp_imgs

def renderDistribution(class_prob, num_bins, percent_disp = None, std_disp = None):
    class_prob = class_prob/class_prob.max()
    fig = get_figure()

    if(percent_disp is not None):
        p_size = class_prob.shape[0]
        p_idxs = np.random.choice(p_size, int(p_size*percent_disp), replace=False)
    elif(std_disp is not None):    
        std_p = class_prob.std()
        mean_p = class_prob.mean()
        p_idxs = class_prob > mean_p + std_disp*std_p
    else:
        p_idxs = np.arange(class_prob.shape[0])
        
    p_vals = class_prob[p_idxs]

    xs = []
    cx = []
    ys = []
    cy = []
    zs = []
    cz = []
    for j, p in zip(p_idxs, p_vals):
        idx = np.unravel_index(j, num_bins)
        x_axis, y_axis, z_axis = index2Axis(idx, num_bins)
        xs.append(x_axis)
        cx.append((1,0,0,p))
        ys.append(y_axis)
        cy.append((0,1,0,p))
        zs.append(z_axis)
        cz.append((0,0,1,p))

    xs = np.array(xs).T
    ys = np.array(ys).T
    zs = np.array(zs).T

    #ax.quiver(0, 0, 0, xs[0], xs[1], xs[2], colors=cx, arrow_length_ratio=0)
    #ax.quiver(0, 0, 0, ys[0], ys[1], ys[2], colors=cy, arrow_length_ratio=0)
    #ax.quiver(0, 0, 0, zs[0], zs[1], zs[2], colors=cz, arrow_length_ratio=0)
    #ax.scatter(xs[0], xs[1], xs[2], s=1, c=cx)
    #ax.scatter(ys[0], ys[1], ys[2], s=1, c=cy)
    #ax.scatter(zs[0], zs[1], zs[2], s=1, c=cz)
    ax = plt.subplot(321, title='X+ Axis')
    ax.scatter(xs[0, xs[2]>=0], xs[1, xs[2]>=0], s=1, c=cx)
    ax = plt.subplot(322, title='Y+ Axis')
    ax.scatter(ys[0, ys[2]>=0], ys[1, ys[2]>=0], s=1, c=cy)
    ax = plt.subplot(323, title='Z+ Axis')
    ax.scatter(zs[0, zs[2]>=0], zs[1, zs[2]>=0], s=1, c=cz)
    ax = plt.subplot(324, title='X+ Axis')
    ax.scatter(xs[0, xs[2]<0], xs[1, xs[2]<0], s=1, c=cx)
    ax = plt.subplot(325, title='Y+ Axis')
    ax.scatter(ys[0, ys[2]<0], ys[1, ys[2]<0], s=1, c=cy)
    ax = plt.subplot(326, title='Z+ Axis')
    ax.scatter(zs[0, zs[2]<0], zs[1, zs[2]<0], s=1, c=cz)
    #ax.auto_scale_xy([-1, 1], [-1, 1])

    return fig2rgb_array(fig)
    #ax.view_init(30, angle)

def renderAxisHistogram(class_prob, num_bins, percent_disp = 0.05, std_disp = None):
    fig = get_figure()

    xs = []
    ys = []
    zs = []
    ws = []
    for j, p in enumerate(class_prob):
        idx = np.unravel_index(j, num_bins)
        x_axis, y_axis, z_axis = index2Axis(idx, num_bins)
        xs.append(x_axis)
        ys.append(y_axis)
        zs.append(z_axis)
        ws.append(p)

    xs = np.array(xs).T
    ys = np.array(ys).T
    zs = np.array(zs).T
    ax = plt.subplot(311, title='X Axis')
    H, xedges, yedges = np.histogram2d(np.arccos(xs[2]), np.arctan(xs[1], xs[0]), bins = [10,20], weights=ws)
    ax.imshow(H, interpolation='bilinear', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax = plt.subplot(312, title='Y Axis')
    H, xedges, yedges = np.histogram2d(np.arccos(ys[2]), np.arctan(ys[1], ys[0]), bins = [10,20], weights=ws)
    ax.imshow(H, interpolation='bilinear', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax = plt.subplot(313, title='Z Axis')
    H, xedges, yedges = np.histogram2d(np.arccos(zs[2]), np.arctan(zs[1], zs[0]), bins = [10,20], weights=ws)
    ax.imshow(H, interpolation='bilinear', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    return fig2rgb_array(fig)
    #ax.view_init(30, angle)


def renderTopRotations(true_values, est_values, num_bins, model_files, 
                       disp_width = 2000, origin_quats=None, num_renders=12, 
                       camera_dist=2):    
   

    rows = int(np.sqrt(num_renders))
    cols = int(np.ceil(num_renders/rows))
    
    render_aspect_ratio = 540./960.    
    render_width = (disp_width-cols+1)//cols
    render_height = int(render_aspect_ratio*render_width)

    mosaic_height = render_width//num_bins[2]
    n = est_values.shape[0]
    disp_imgs = np.zeros((n, (render_height + 2*mosaic_height + 3)*rows-1, 
                          (render_width + 1)*cols-1, 3), dtype='float32')

    for j in range(n):
        class_prob = np.exp(est_values[0])
        class_prob /= class_prob.sum()
        class_cumsum = np.cumsum(class_prob)
    
        if(origin_quats is None):
            origin_quats = [np.array([0,0,0,1])]
            
        quats = []
        indices = []
        probs = []    
        
        true_idx = np.unravel_index(np.argmax(true_values[j]), num_bins)
        true_q = tf_trans.quaternion_multiply(index2Quat(true_idx, num_bins), origin_quats[j])
        
        quats.append(true_q)
        indices.append(true_idx)
        probs.append(class_prob[np.argmax(true_values[j])])
        
        max_idx = np.unravel_index(np.argmax(class_prob), num_bins)
        max_q = tf_trans.quaternion_multiply(index2Quat(max_idx, num_bins), origin_quats[j])

        quats.append(max_q)
        indices.append(max_idx)
        probs.append(np.max(class_prob))
        
        rand_samples = np.random.rand(num_renders-2)
        for v in rand_samples:
            k = np.nonzero(v<class_cumsum)[0][0]
            p = class_prob[k]
            idx = np.unravel_index(k, num_bins)
            q = tf_trans.quaternion_multiply(index2Quat(idx, num_bins), origin_quats[j])
            
            quats.append(q)
            indices.append(idx)
            probs.append(p)

        render_imgs = renderView(model_files[j], quats, camera_dist, standard_lighting=True)
        true_mosaic_img = makeProbabilityMosaic(true_values[0], num_bins, plt.cm.Blues)
        true_mosaic_img = cv2.resize(true_mosaic_img, (render_width, mosaic_height))
#        est_mosaic_img = makeProbabilityMosaic(class_prob, num_bins, plt.cm.Reds, max_idx=False)
#        est_mosaic_img = cv2.resize(est_mosaic_img, (render_width, mosaic_height))

        for k, (img, q, p, idx) in enumerate(zip(render_imgs, quats, probs, indices)):
            r = k//cols
            c = k%cols
            disp_mosaic_img = makeProbabilityMosaic(class_prob, num_bins, plt.cm.Reds, max_idx=idx, max_radius=6)
            disp_mosaic_img = cv2.resize(disp_mosaic_img, (render_width, mosaic_height))
#            target_pixel_x = int((idx[1] + idx[0]*(num_bins[1]+1)) * render_width/((num_bins[1]+1)*num_bins[2]-1))
#            target_pixel_y = int(idx[0]*mosaic_height/num_bins[0])
#            disp_mosaic_img = est_mosaic_img.copy()
#            cv2.circle(disp_mosaic_img, (target_pixel_x, target_pixel_y), 6, (0,1,0), -1)
            
            c0 = c * (render_width + 1)
            r0 = r * (render_height + 2*mosaic_height + 3)

            disp_imgs[j,r0:r0+render_height,c0:c0+render_width,:] = cv2.resize(transparentOverlay(img), (render_width, render_height))/255.0
            if(k == 0):
                display_string = 'GROUND TRUTH'
                cv2.putText(disp_imgs[j], display_string, (c0, r0+20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
                cv2.putText(disp_imgs[j], display_string, (c0, r0+20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
            elif(k == 1):
                display_string = 'MAX BIN'
                cv2.putText(disp_imgs[j], display_string, (c0, r0+20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
                cv2.putText(disp_imgs[j], display_string, (c0, r0+20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
                            
            display_string = '{}: p:{:0.3e} e:{:0.3}'.format(tuple(idx), p, indexAngularDiff(idx, true_idx, num_bins))
            cv2.putText(disp_imgs[j], display_string, (c0, r0+render_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
            cv2.putText(disp_imgs[j], display_string, (c0, r0+render_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
            axis, angle = quat2AxisAngle(index2Quat(idx, num_bins))
            display_string = '{:.3f}, {}'.format(angle*180/np.pi, np.round(axis, 2))
            cv2.putText(disp_imgs[j], display_string, (c0, r0+render_height-5),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
            cv2.putText(disp_imgs[j], display_string, (c0, r0+render_height-5),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
            r0 += render_height + 1
            disp_imgs[j,r0:r0+mosaic_height,c0:c0+render_width,:] = true_mosaic_img
            r0 += mosaic_height + 1
            disp_imgs[j,r0:r0+mosaic_height,c0:c0+render_width,:] = disp_mosaic_img

    disp_imgs = np.transpose(disp_imgs, (0,3,1,2))
    return disp_imgs

def renderQuaternions(origin_imgs, query_imgs, 
                      true_quats, est_quats,
                      model_files, origin_quats=None,
                      display_width = 1250, 
                      norm_mean = np.array([0.485, 0.456, 0.406]),
                      norm_std = np.array([0.229, 0.224, 0.225]), 
                      camera_dist = 2, render_gripper = True):

    origin_imgs = np.transpose(origin_imgs, (0,2,3,1))
    query_imgs = np.transpose(query_imgs, (0,2,3,1))
    origin_imgs = np.minimum(np.maximum(origin_imgs*norm_std + norm_mean, 0.0), 1.0)
    query_imgs = np.minimum(np.maximum(query_imgs*norm_std + norm_mean, 0.0), 1.0)
    
    n, h, w, c = origin_imgs.shape
    disp_w = 3*(display_width//3)
    disp_h = (h*disp_w)//(3*w)
    
    disp_imgs = np.zeros((n, disp_h, disp_w, c), dtype='float32')
    if(render_gripper):
        gripper_q = np.array([ 0.5,  0.5,  0.5,  0.5])
    else:
        gripper_q = np.array([ 0.0,  0.0,  0.0,  1.0])
        
    for j in range(n):
        est_q = est_quats[j]/np.linalg.norm(est_quats[j])
        est_q *= np.sign(est_q[3])
        diff = quatAngularDiff(est_q, true_quats[j])

        if(origin_quats is None):
            origin_quats = [np.array([0,0,0,1])]

        q = tf_trans.quaternion_multiply(origin_quats[j], gripper_q) #tf_trans.quaternion_multiply(est_q, origin_quats[j])
        render_img = transparentOverlay(renderView(model_files[j], [q], camera_dist, standard_lighting=True)[0])/255.0

        disp_imgs[j, :, :disp_w//3, :] = cv2.resize(origin_imgs[j], (disp_w//3, disp_h))
        disp_imgs[j, :, disp_w//3:2*(disp_w//3), :] = cv2.resize(query_imgs[j], (disp_w//3, disp_h))
        disp_imgs[j, :, 2*(disp_w//3):, :] = cv2.resize(render_img, (disp_w//3, disp_h))
        
        true_axis, true_angle = quat2AxisAngle(true_quats[j])
        
        display_string_true = '{}'.format(origin_quats[j])
        text_height = disp_h-60
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)        
        
        display_string_true = 'True Angle: {:.3f}, True Axis: {}'.format(true_angle*180/np.pi, true_axis)
        text_height = disp_h-40
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_true, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
        text_height += 20
        
        est_axis, est_angle = quat2AxisAngle(est_q)
        display_string_est  = 'Est Angle:  {:.3f}, Est Axis:  {}, Error: {}'.format(est_angle*180/np.pi, est_axis, diff)
        cv2.putText(disp_imgs[j], display_string_est, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_est, (10, text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
        text_height += 20
        
    disp_imgs = np.transpose(disp_imgs, (0,3,1,2))
    return disp_imgs