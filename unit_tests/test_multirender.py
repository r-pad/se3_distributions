# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
from model_renderer.pose_renderer import BpyRenderer

import os
import cv2
import numpy as np
import time

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
import quat_math

import multiprocessing

def renderMat(renderer, mat, index, t):
    render_img = renderer.renderTrans(mat)
    print('Multi Time {}: {}'.format(index, time.time()-t))
    return
    results_dir = '/home/bokorn/results/test/renders/'
    #cv2.imwrite(results_dir + '{}_multi_render.png'.format(index), render_img)
    print('Rendered', results_dir + '{}_mulit_render.png'.format(index))
        
def main():
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    results_dir = '/home/bokorn/results/test/renders/'
    dataset = YCBDataset(data_dir=benchmark_folder,
                         image_set='valid_split',
                         img_size=(224,224),
                         obj=1)

    dataset.loop_truth = None
    
    renderer = BpyRenderer()                                                            
    renderer.loadModel(dataset.getModelFilename())
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    renderer.setCameraMatrix(fx, fy, px, py, 640, 480)

    trans_mats = []
    for j in range(10):
        trans_mat = quat_math.quaternion_matrix(quat_math.random_quaternion())
        ycb_mat = quat_math.euler_matrix(-np.pi/2,0,0)
        trans_mat = trans_mat.dot(ycb_mat)
        trans_mat[:3,3] = [0, 0, 1] 
        trans_mats.append(trans_mat)

    t = time.time()
    for j, mat in enumerate(trans_mats):    
        render_img = renderer.renderTrans(mat)
    #    cv2.imwrite(results_dir + '{}_seq_render.png'.format(j), render_img)
    print('Seq Time:', time.time()-t)
    workers = []
    t = time.time()
    for j, mat in enumerate(trans_mats):    
        w = multiprocessing.Process(
            target=renderMat,
            args=(renderer, mat, j, t))
        w.daemon = True
        w.start()
        workers.append(w) 
    import IPython; IPython.embed() 
if __name__=='__main__':
    main()

