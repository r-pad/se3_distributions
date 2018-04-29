# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:03:29 2018

@author: bokorn
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model_renderer.pose_renderer import renderView

from generic_pose.training.utils import to_var, to_np
from generic_pose.utils.data_preprocessing import (resizeAndPad, 
                                                   cropAndResize, 
                                                   transparentOverlay, 
                                                   quatDiff, 
                                                   quatAngularDiff,
                                                   quat2AxisAngle)

import generic_pose.utils.transformations as tf_trans
from generic_pose.models.pose_networks import gen_pose_net

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

to_tensor = transforms.ToTensor()
                                     
def preprocessImages(image, img_size, normalized_tensor = True, 
                     background = None, crop_percent = None):
    if (len(image.shape) == 2):
        image = np.expand_dims(image, axis=2)
    
    if(image.shape[2] == 4):
        image = transparentOverlay(image, background)
    
    if(crop_percent is not None):
        image = cropAndResize(image, img_size, crop_percent)
    else:
        image = resizeAndPad(image, img_size)
    
    if(normalized_tensor):
        image = normalize(to_tensor(image))

    return image

def evalQuat(quat_est, origin_quat, query_quat, max_angle = np.pi/4.0):
    quat_est /= np.linalg.norm(quat_est)
    axis_est, angle_est = quat2AxisAngle(quat_est)
    clipped_angle = np.clip(angle_est,-max_angle,max_angle)
    clipped_est = tf_trans.quaternion_about_axis(clipped_angle, axis_est)
    
    iter_quat = tf_trans.quaternion_multiply(clipped_est, origin_quat)
    err = quatAngularDiff(iter_quat, query_quat)
    return iter_quat, err

def evaluateIterQuatBatch(model, model_filename,
                          origin_quats = None, 
                          query_quats = None,
                          num_samples = 1,
                          return_images = True,
                          img_size = (224,224),
                          camera_dist = 2.0,
                          max_angle = np.pi/4.0, 
                          num_iters = 10,
                          min_err = np.pi/36.0):
    if(origin_quats is None):
        iter_quats = [tf_trans.random_quaternion() for _ in range(num_samples)]
    else:
        iter_quats = origin_quats
    
    if(query_quats is None):
        query_quats = [tf_trans.random_quaternion() for _ in range(num_samples)]
    
    rendered_images = renderView(model_filename, iter_quats+query_quats,
                                 camera_dist = camera_dist,
                                 standard_lighting = True)
    iter_imgs = to_var(torch.stack([preprocessImages(img, img_size) for img in rendered_images[:num_samples]]))
    query_imgs = to_var(torch.stack([preprocessImages(img, img_size) for img in rendered_images[num_samples:]]))
    
    iter_errors = [[quatAngularDiff(iq, qq) for iq, qq in zip(iter_quats, query_quats)]]
    if(return_images):
        iter_images = [rendered_images[:num_samples]]
    for j in range(num_iters):
        quat_ests = to_np(model.forward(iter_imgs, query_imgs))
        iter_quats, errs = zip(*[evalQuat(qe, oq, qq, max_angle) for qe, oq, qq in zip(quat_ests, iter_quats, query_quats)])
        iter_imgs = renderView(model_filename, iter_quats, 
                               camera_dist = camera_dist,
                               standard_lighting=True)
        iter_errors.append(errs)
        if(return_images):
            iter_images.append(iter_imgs)
        iter_imgs = to_var(torch.stack([preprocessImages(img, img_size) for img in iter_imgs]))
        
    if(return_images):
        query_imgs = rendered_images[num_samples:]
        return iter_errors, iter_images, query_imgs
    else:
        return iter_errors
        
def evaluateIterQuat(model, model_filename,
                     origin_quat = None, query_quat=None,
                     img_size = (224,224),
                     camera_dist = 2.0,
                     max_angle = np.pi/4.0, 
                     num_iters = 10,
                     min_err = np.pi/36.0):
    if(origin_quat is None):
        origin_quat = tf_trans.random_quaternion()
    if(query_quat is None):
        query_quat = tf_trans.random_quaternion()
        
    rendered_images = renderView(model_filename, [origin_quat, query_quat],
                                 camera_dist = camera_dist,
                                 standard_lighting = True)
    iter_img = to_var(preprocessImages(rendered_images[0], img_size)).unsqueeze(0)
    iter_quat = origin_quat
    query_img  = to_var(preprocessImages(rendered_images[1], img_size)).unsqueeze(0)
    
    iter_errors = [quatAngularDiff(iter_quat, query_quat)]
    iter_images = [rendered_images[0]]
    for j in range(num_iters):
        quat_est = to_np(model.forward(iter_img, query_img))[0]
        quat_est /= np.linalg.norm(quat_est)
        axis_est, angle_est = quat2AxisAngle(quat_est)
        clipped_angle = np.clip(angle_est,-max_angle,max_angle)
        clipped_est = tf_trans.quaternion_about_axis(clipped_angle, axis_est)
        
        iter_quat = tf_trans.quaternion_multiply(clipped_est, iter_quat)
        err = quatAngularDiff(iter_quat, query_quat)
        iter_errors.append(err)
        iter_img = renderView(model_filename, [iter_quat], 
                              camera_dist = camera_dist,
                              standard_lighting=True)[0]
        iter_images.append(iter_img)
        iter_img = to_var(preprocessImages(iter_img, img_size)).unsqueeze(0)

        if(abs(err) < min_err):
            break
    
    return iter_errors, iter_images

def evaluateIterQuatModel(weight_filename, model_filename,
                          model_type='alexnet', compare_type='basic',
                          img_size = (224,224),
                          camera_dist = 2.0,
                          max_angle = np.pi/4.0, 
                          num_iters = 10,
                          min_err = np.pi/36.0):

    model = gen_pose_net(model_type, 
                         compare_type, 
                         output_dim = 4)
                         
    model.load_state_dict(torch.load(weight_filename))
    model.eval()
    model.cuda()

    errs, imgs = evaluateIterQuat(model, model_filename,
                                  img_size = img_size,
                                  camera_dist = camera_dist,
                                  max_angle = max_angle, 
                                  num_iters = num_iters,
                                  min_err = min_err)
    
    return errs, imgs

