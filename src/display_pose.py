# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:39:11 2017

@author: bokorn
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))
    
def makeDisplayImage(origin_imgs, query_imgs, 
                     true_azims, true_elevs, true_tilts, true_quats,
                     est_azims, est_elevs, est_tilts, est_quats,
                     bin_height = 10, text_height = 0):
    origin_imgs = np.transpose(origin_imgs, (0,2,3,1))/255.0
    query_imgs = np.transpose(query_imgs, (0,2,3,1))/255.0
    n, h, w, c = origin_imgs.shape
    disp_w = 2*true_azims.shape[-1]
    disp_h = (h*disp_w)//(2*w)
    disp_imgs = np.zeros((n, disp_h+6*bin_height+2+text_height, disp_w, c), dtype='float32')
    
    for j in range(n):
        disp_imgs[j, :disp_h, :disp_w//2, :] = cv2.resize(origin_imgs[j], (disp_w//2, disp_h))
        disp_imgs[j, :disp_h, disp_w//2:, :] = cv2.resize(query_imgs[j], (disp_w//2, disp_h))

        true_quats[j] *= np.sign(true_quats[j,3])
        true_angle = 2*np.arccos(true_quats[j,3])
        true_axis = true_quats[j,:3]/np.sin(true_angle/2.0)
        est_q = est_quats[j]/np.linalg.norm(est_quats[j])
        est_q *= np.sign(est_q[3])
        est_angle = 2*np.arccos(est_q[3])
        est_axis = est_q[:3]/np.sin(est_angle/2.0)       
        display_string_true = 'True Angle: {:.3f}, True Axis: {}'.format(true_angle*180/np.pi, true_axis)
        display_string_est  = 'Est Angle:  {:.3f}, Est Axis:  {}'.format(est_angle*180/np.pi, est_axis)

        cv2.putText(disp_imgs[j], display_string_true, (10, disp_h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_est, (10, disp_h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (1.0,1.0,1.0), 2)
        cv2.putText(disp_imgs[j], display_string_true, (10, disp_h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)
        cv2.putText(disp_imgs[j], display_string_est, (10, disp_h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0.0,0.0,0.0), 1)

        disp_col = disp_h + text_height
        for k, v in enumerate([true_azims[j], est_azims[j], true_elevs[j], est_elevs[j], true_tilts[j], est_tilts[j]]):
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