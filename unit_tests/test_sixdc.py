# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn
"""

import os
from pysixd.inout import load_ply, load_gt, load_cam_params, load_depth
from pysixd import renderer, misc, visibility
import matplotlib.pyplot as plt

sixdc_folder = '/media/bokorn/ExtraDrive2/benchmark/linemod6DC'

model = load_ply(os.path.join(sixdc_folder, 'models/obj_01.ply'))

gt = load_gt(os.path.join(sixdc_folder, 'test/01/gt.yml'))
R_gt = gt[0][0]['cam_R_m2c']
t_gt = gt[0][0]['cam_t_m2c']

cam = load_cam_params(os.path.join(sixdc_folder, 'camera.yml'))
K = cam['K']

depth_test = load_depth(os.path.join(sixdc_folder, 'test/01/depth/0000.png'))
rgb_test = load_im(os.path.join(sixdc_folder, 'test/01/rgb/0000.png'))
im_size = (depth_test.shape[1], depth_test.shape[0])

depth_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
                           clip_far=10000, mode='depth')

dist_test = misc.depth_im_to_dist_im(depth_test, K)
dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
delta = 15
visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)


imgplot = ax.imshow(visib_gt)

import IPython; IPython.embed()
