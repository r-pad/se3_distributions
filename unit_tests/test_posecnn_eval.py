# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn 
"""
import os
import cv2
import numpy as np

from generic_pose.datasets.ycb_dataset import YCBDataset 
from generic_pose.eval.posecnn_eval import evaluatePoses, getYCBThresholds
from generic_pose.bbTrans.discretized4dSphere import S3Grid

def quatAngularDiffDot(q1, q2):
        return 2*np.arccos(np.abs(q1.dot(q2.T)))

def main():
    benchmark_folder = '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset'
    grid = S3Grid(2)
    grid.Simplify()
    grid_verts = grid.vertices
    
    thresholds = getYCBThresholds('/home/bokorn/src/posecnn_thecat/PoseCNN/data/LOV/extents.txt') 

    for j in range(1,22):
        ycb_dataset = YCBDataset(data_dir=benchmark_folder, 
                                 image_set='valid_split',
                                 img_size=(224, 224),                                    
                                 obj=j)
        min_idxs = np.argmin(quatAngularDiffDot(grid_verts, ycb_dataset.quats), axis=0)
        quat_preds = grid_verts[min_idxs]
        acc, errors = evaluatePoses(ycb_dataset, quat_preds, thresholds[j])
        import IPython; IPython.embed()


if __name__=='__main__':
    main()
