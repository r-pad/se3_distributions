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

from generic_pose.datasets.ycb_dataset import YCBDataset, setYCBCamera
from generic_pose.utils.image_preprocessing import cropAndPad
from generic_pose.utils.image_preprocessing import unprocessImages
from generic_pose.utils.data_augmentation import rotateAndScale, rotateData
import quat_math

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

ycb_mat = quat_math.euler_matrix(-np.pi/2,0,0)

def renderQuat(renderer, quat):
    render_mat = quat_math.quaternion_matrix(quat).dot(ycb_mat)
    render_mat[:3,3] = [0, 0, 1] 
    return renderer.renderTrans(render_mat)

def main():
    benchmark_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset'
    test_dir = '/home/bokorn/results/test/rotation/'
    dataset = YCBDataset(data_dir=benchmark_folder,                                         
                             image_set='train_split',
                             img_size=(224, 224),                                    
                             obj=1)


    renderer = BpyRenderer()                                                            
    renderer.loadModel(dataset.getModelFilename())
 
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    renderer.setCameraMatrix(fx, fy, px, py, 640, 480)

    if(False):
        index = 1000
        image_prefix = os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index])

        img = cv2.imread(image_prefix + '-color.png')
        mask = 255*(cv2.imread(image_prefix + '-label.png')[:,:,:1] == dataset.obj).astype('uint8')
        img_masked = np.concatenate([img, mask], axis=2)
        cv2.imwrite(test_dir + 'img.png', cropAndPad(img_masked))
        
        data = sio.loadmat(os.path.join(dataset.data_dir, 'data', dataset.data_filenames[index] + '-meta.mat'))
        pose_idx = np.where(data['cls_indexes'].flatten()==dataset.obj)[0][0]
        
        trans_mat = np.vstack([data['poses'][:,:,pose_idx], [0,0,0,1]])
        q = quat_math.quaternion_from_matrix(trans_mat)
        
        render_img = renderQuat(renderer, q)
        cv2.imwrite(test_dir + 'render.png', cropAndPad(render_img))
     
    if(False):
        #render_mat = trans_mat.dot(ycb_mat)
     
        theta = np.deg2rad(-77)
        img_rot = rotateAndScale(img, theta)
        img_masked_rot = rotateAndScale(img_masked, theta)
     
        #theta = -77
        #rows,cols = img.shape[:2]
        #M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        #img_rot = cv2.warpAffine(img,M,(cols,rows))
        #img_masked_rot = cv2.warpAffine(img_masked,M,(cols,rows))
        cv2.imwrite(test_dir + 'rot_img.png', cropAndPad(img_masked_rot))
        cv2.imwrite(test_dir + 'rot_img_full.png', img_rot)
      
        rot_mat = quat_math.quaternion_matrix(quat_math.quaternion_about_axis(theta, [0,0,-1]))
        q_rot = quat_math.quaternion_multiply(quat_math.quaternion_about_axis(theta, [0,0,-1]), q)
        render_mat = quat_math.quaternion_matrix(q_rot).dot(ycb_mat)
        render_mat[:3,3] = [0, 0, 1] 
        render_img_rot = renderer.renderTrans(render_mat)
        cv2.imwrite(test_dir + 'rot_render_img.png', cropAndPad(render_img_rot))
        
        render_mat = rot_mat.dot(trans_mat).dot(ycb_mat)
        render_img_rot = renderer.renderTrans(render_mat)
        cv2.imwrite(test_dir + 'rot_render_full.png', render_img_rot)
    
    if(False):
        img_rot, q_rot = rotateData(img_masked, q)
        render_mat = quat_math.quaternion_matrix(q_rot).dot(ycb_mat)
        render_mat[:3,3] = [0, 0, 1] 
        render_img_rot = renderer.renderTrans(render_mat)
        cv2.imwrite(test_dir + 'rot_img.png', cropAndPad(img_rot))
        cv2.imwrite(test_dir + 'rot_render_img.png', cropAndPad(render_img_rot))
     
    brightness_jitter = 0.5
    contrast_jitter = 0.5
    saturation_jitter = 0.5
    hue_jitter = 0.25
    max_translation = (0.2, 0.2)
    max_scale = (0.8, 1.2)
    rotate_image = True
    max_num_occlusions = 3
    max_occlusion_area = (0.1, 0.3)
    augmentation_prob = 0.5

    train_dataset = YCBDataset(data_dir=benchmark_folder, 
                               image_set='train_split',
                               img_size=(224, 224),
                               obj=1,
                               use_syn_data=True,
                               brightness_jitter = brightness_jitter,
                               contrast_jitter = contrast_jitter,
                               saturation_jitter = saturation_jitter,
                               hue_jitter = hue_jitter,
                               max_translation = max_translation,
                               max_scale = max_scale,
                               rotate_image = rotate_image,
                               max_num_occlusions = max_num_occlusions,
                               max_occlusion_area = max_occlusion_area,
                               augmentation_prob = augmentation_prob)

    train_dataset.loop_truth = None
    for j in range(10):
        idx = np.random.randint(len(train_dataset))
        train_dataset.augmentation_prob = augmentation_prob
        img, q = train_dataset.__getitem__(idx)[:2]
        cv2.imwrite(test_dir + '{}_aug.png'.format(j), 
                unprocessImages(img.unsqueeze(0))[0])
        cv2.imwrite(test_dir + '{}_aug_rend.png'.format(j), 
                cropAndPad(renderQuat(renderer, q)))
        train_dataset.augmentation_prob = 0
        img, q = train_dataset.__getitem__(idx)[:2]
        cv2.imwrite(test_dir + '{}_org.png'.format(j), 
                unprocessImages(img.unsqueeze(0))[0])
        cv2.imwrite(test_dir + '{}_org_rend.png'.format(j), 
                cropAndPad(renderQuat(renderer, q)))
         
       
    import IPython; IPython.embed()

if __name__=='__main__':
    main()
    import IPython; IPython.embed()

