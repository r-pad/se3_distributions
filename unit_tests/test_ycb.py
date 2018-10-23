# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from model_renderer.pose_renderer import BpyRenderer

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import numpy as np
import cv2

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils import to_np, to_var
from generic_pose.utils.image_preprocessing import preprocessImages, unprocessImages
from generic_pose.utils.pose_processing import viewpoint2Pose, pose2Viewpoint
from quat_math.transformations import quaternion_multiply, quaternion_about_axis

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

def savePose(renderer, quat, filename,
             camera_dist= 0.3, 
             img_size=(224,224)):
    render = preprocessImages(renderer.renderPose([quat], camera_dist = camera_dist),
                              img_size = img_size,
                              normalize_tensors = True).float()
    cv2.imwrite(filename, unprocessImages(render)[0])

def visualizeDataset(dataset, renderer, image_prefix,
                     indices = [0], 
                     model_scale = 1.0,
                     camera_dist = 0.3,
                     img_size = (224, 224)):
    renderer.deleteAll()       
    renderer.loadModel(dataset.getModelFilename(),
                       model_scale = model_scale, 
                       emit = 0.5)
    for index in indices:
        quat = dataset.getQuat(index)
        image  = dataset.getImage(index)
        #pose_quat = pose2Viewpoint(quat)
        #render_quat = quaternion_multiply(pose_quat, quaternion_about_axis(np.pi/2, [1,0,0]))
        savePose(renderer, quat, image_prefix + '{}_render.png'.format(index),
                 camera_dist = camera_dist, img_size = img_size)

        #render = preprocessImages(renderer.renderPose([render_quat], camera_dist = camera_dist),
        #                          img_size = img_size,
        #                          normalize_tensors = True).float()

        cv2.imwrite(image_prefix + '{}_real.png'.format(index), unprocessImages(image.unsqueeze(0))[0])
        #cv2.imwrite(image_prefix + 'render_{}.png'.format(index), unprocessImages(render)[0])
        #import IPython; IPython.embed()

def main():
    from argparse import ArgumentParser
    from generic_pose.models.pose_networks import gen_pose_net, load_state_dict
    parser = ArgumentParser()

    parser.add_argument('--benchmark_folder', type=str, default='/media/bokorn/ExtraDrive2/benchmark/ycb/YCB_Video_Dataset')
    parser.add_argument('--image_prefix', type=str, default='/home/bokorn/Downloads/test/')
    parser.add_argument('--target_object', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)

    args = parser.parse_args()
    
    dataset = YCBDataset(data_dir=args.benchmark_folder, 
                         image_set='train',
                         img_size=(args.height, args.width),
                         obj=args.target_object)
    dataset.loop_truth = [1]
    #loader = DataLoader(dataset,
    #                    num_workers=num_workers, 
    #                    batch_size=int(batch_size/2), 
    #                    shuffle=True)
    renderer = BpyRenderer(transform_func = ycbRenderTransform)
    visualizeDataset(dataset, renderer, args.image_prefix)
    import IPython; IPython.embed()

if __name__=='__main__':
    main()
