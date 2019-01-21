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

from torch.utils.data import DataLoader

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.datasets.renderer_dataset import PoseRendererDataset
from generic_pose.utils.image_preprocessing import unprocessImages
import quat_math

       
def main():
    benchmark_folder='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/' 
    results_dir = '/home/bokorn/results/test/renders/'
    ycb_dataset = YCBDataset(data_dir=benchmark_folder,
                         image_set='valid_split',
                         img_size=(224,224),
                         obj=1)
    renderer_dataset = PoseRendererDataset(ycb_dataset.getModelFilename(), img_size=(224,224))
    loader = DataLoader(renderer_dataset,
                        num_workers=3, 
                        batch_size=16, 
                        shuffle=True)

    for j, (img, quat, _, _) in enumerate(loader):
        print(j)
        if(j > 10):
            break
    import IPython; IPython.embed() 
if __name__=='__main__':
    main()

