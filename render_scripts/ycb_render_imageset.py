# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""

from model_renderer.pose_renderer import BpyRenderer


import os
import time
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
       
def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--benchmark_folder', type=str, default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset')
    args = parser.parse_args()
    
    dataset = YCBDataset(data_dir=args.benchmark_folder, 
                         image_set='train_split',
                         img_size=(224, 224),
                         use_syn_data=True,
                         )
    for cls_idx in range(1, len(dataset.classes)):
        dataset.setObject(cls_idx)            
        dataset.generateRenderedImages()
    import IPython; IPython.embed()

if __name__=='__main__':
    main()

