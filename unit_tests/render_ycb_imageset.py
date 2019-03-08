# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 00:27:44 2018

@author: bokorn
"""
from tqdm import tqdm
import torch
import cv2
import os

from generic_pose.datasets.ycb_dataset import YCBDataset, ycbRenderTransform
from generic_pose.utils.image_preprocessing import unprocessImages
        
def render(obj, results_prefix, image_set, benchmark_dir):
    if(image_set == 'base'):
        classes =['']
        with open(os.path.join(benchmark_dir, 'image_sets', 'classes.txt')) as f:
            classes.extend([x.rstrip('\n') for x in f.readlines()])

        base_render_folder = os.path.join(benchmark_dir,
                                          'base_renders',
                                          classes[obj],
                                          '{}'.format(2))
        base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
        base_vertices = torch.load(os.path.join(base_render_folder, 'vertices.pt'))
        
        num_imgs = base_renders.shape[0]
        digits = len(str(num_imgs))
        pbar = tqdm(range(num_imgs))
        for j in pbar:
            cv2.imwrite(results_prefix + '{1:0{0}d}.png'.format(digits, j), 
                        unprocessImages(base_renders[j].unsqueeze(0))[0])
    else:
        dataset = YCBDataset(data_dir=benchmark_dir,
                             image_set=image_set,
                             img_size=(224,224),
                             use_posecnn_masks = True,
                             obj=obj)

        dataset.loop_truth = None
        dataset.resample_on_none = False
        num_imgs = len(dataset)
        digits = len(str(num_imgs))
        pbar = tqdm(range(num_imgs))
        for j in pbar:
            cv2.imwrite(results_prefix + '{1:0{0}d}.png'.format(digits, j), 
                        dataset.getImage(j, preprocess = False)[0])
        return 

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('obj', type=int) 
    parser.add_argument('--results_prefix', type=str, default=None)
    parser.add_argument('--image_set', type=str, default='valid_split')
    parser.add_argument('--benchmark_dir', type=str, 
        default='/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/')
    args = parser.parse_args()
    
    if(args.results_prefix is None):
        args.results_prefix = '/home/bokorn/results/test/visualizations/model_{}/{}_'.format(args.obj, args.image_set)
    
    render(args.obj, args.results_prefix, args.image_set, args.benchmark_dir)

