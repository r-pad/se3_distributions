#!/usr/bin/env python

try:
    from model_renderer.pose_renderer import BpyRenderer
    from generic_pose.datasets.ycb_dataset import ycbRenderTransform 
    BPY_IMPORTED = True 
except ImportError:
    BPY_IMPORTED = False 

import numpy as np

import cv2
import torch
from collections import namedtuple
from functools import partial
from generic_pose.utils.image_preprocessing import preprocessImages, unprocessImages

def matchTemplate(img, template):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_loc, max_val

def createMultiscaleImage(img, scales = None):
    if scales is None:
        scales = [1.5, 1.25, 1, .66, .33]
    h,w = img.shape[:2]
    
    image_pyrimid = []
    for s in scales:
        hs = int(s*h)
        ws = int(s*w)
        if(s != 1):
            image_pyrimid.append(cv2.resize(img, (hs, ws)))
        #if(s > 1):
            #image_pyrimid.append(cv2.pyrUp(img, dstsize=(hs,ws)))
        #elif(s < 1):
            #image_pyrimid.append(cv2.pyrDown(img, dstsize=(hs,ws)))
        else:
            image_pyrimid.append(img)

    return image_pyrimid

def computeT(img, template, f, p, 
             template_dist = 0.2, f_t = 490, pyr_scales = None, 
             bbox_corner = (0,0), resolution_percentage = 50): 


    diag_t = (100.0/resolution_percentage)*np.linalg.norm(template.shape[:2])
    max_scale = np.min(np.array(img.shape[:2]) / np.array(template.shape[:2]))
    scales = np.linspace(0.25, max_scale, 10)
    image_pyrimid = createMultiscaleImage(template, scales = scales)
    max_loc = None
    max_lvl = None
    max_val = -np.inf
    for pyr_lvl, temp_pyr in enumerate(image_pyrimid):
        if(all(np.array(temp_pyr.shape[:2]) < np.array(img.shape[:2]))):
            loc, val = matchTemplate(temp_pyr, img)
            if(val > max_val):
                max_val = val
                max_loc = loc
                max_lvl = pyr_lvl

    ht, wt = image_pyrimid[max_lvl].shape[:2]
    x = max_loc[0] + bbox_corner[0] + ht//2
    y = max_loc[1] + bbox_corner[1] + wt//2
    diag = np.linalg.norm([ht, wt])

    tz = template_dist * diag_t/diag * f/f_t
    ty = tz / f * (y - p[1])
    tx = tz / f * (x - p[0])

    return np.array([tx,ty,tz]), np.array([x,y,wt,ht])


class TemplatePoseEstimator(object):
    def __init__(self, focal_length, image_center, renders_dir = None, object_model = None, 
                 render_distance = 0.2, render_f = 290, render_center = (112,112)):
        self.render_distance = render_distance
        
        self.render_f = render_f
        self.render_p = render_center

        self.img_f = focal_length
        self.img_p = image_center

        if(renders_dir is not None and not BPY_IMPORTED):
            self.renders = torch.load(os.path.join(renders_dir, 'renders.pt'))
            self.render_diag = np.linalg.norm(self.renders.shape[2:])
        else:
            self.renders = None
        
        if(object_model is not None):
            assert BPY_IMPORTED, 'Blender must be installed and a python module (bpy) to render model'
            self.renderer = BpyRenderer(transform_func = ycbRenderTransform)
            self.renderer.loadModel(object_model, emit = 0.5)
            self.render_diag = np.linalg.norm([224, 224])
        else:
            self.renderer = None

    def __call__(self, img, quat = None, top_idx = None, pyr_scales = None, 
                 bbox_corner = (0,0)): 
        if(quat is not None and self.renderer is not None):
            template = self.renderer.renderPose([quat],camera_dist=self.render_distance)[0]
        elif(top_idx is not None and self.renders is not None):
            template = unprocessImages(self.renders[top_idx].unsqueeze(0))[0]
        else:
            raise ValueError('quat or top_idx must be given')

        return computeT(img, template, f = self.img_f, p =self.img_p, 
                        f_t = self.render_r, pyr_scales = pyr_scales, 
                        bbox_corner = bbox_corner)


