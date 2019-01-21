# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
from model_renderer.pose_renderer import BpyRenderer

import os
import cv2
import numpy as np
import scipy.io as sio

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
import quat_math

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

        
def main():
    augmentation_prob = 0.0
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    results_dir = '/home/bokorn/results/test/masks/'
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

    index = 10
    image_prefix = os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index])

    img = cv2.imread(image_prefix + '-color.png')
    mask = 255*(cv2.imread(image_prefix + '-label.png')[:,:,:1] == dataset.obj).astype('uint8')
    data = sio.loadmat(os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index] + '-meta.mat'))
    pose_idx = np.where(data['cls_indexes'].flatten()==dataset.obj)[0][0]
    trans_mat = np.vstack([data['poses'][:,:,pose_idx], [0,0,0,1]])
    
    ycb_mat = quat_math.euler_matrix(-np.pi/2,0,0)
    trans_mat = trans_mat.dot(ycb_mat)
    #trans_mat[:3,3] = [0, 0, 1] 
    
    render_img = renderer.renderTrans(trans_mat)
    render_mask = render_img[:,:,3]
    cv2.imwrite(results_dir + '{}_img.png'.format(index), img)
    cv2.imwrite(results_dir + '{}_mask.png'.format(index), mask)
    cv2.imwrite(results_dir + '{}_rendered_img.png'.format(index), render_img)
    cv2.imwrite(results_dir + '{}_rendered_mask.png'.format(index), render_mask)
    combined_mask = np.zeros([480, 640, 3])
    combined_mask[:,:,:1] = mask
    combined_mask[:,:,2] = render_mask
    cv2.imwrite(results_dir + '{}_combined_mask.png'.format(index), combined_mask)


    import IPython; IPython.embed() 
if __name__=='__main__':
    main()

