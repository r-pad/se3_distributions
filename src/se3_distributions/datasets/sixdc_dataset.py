# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn
"""

import os
import cv2
import numpy as np
import torch

from se3_distributions.datasets.image_dataset import PoseImageDataset
from se3_distributions.utils import SingularArray
import se3_distributions.utils.transformations as tf_trans
from se3_distributions.utils.pose_processing import viewpoint2Pose
from pysixd.inout import load_ply, load_gt, load_cam_params, load_depth, load_im, save_im, load_yaml

def sixdcRenderTransform(q):
    trans_quat = q.copy()
    return viewpoint2Pose(trans_quat)

class SixDCDataset(PoseImageDataset):
    def __init__(self, data_dir, use_mask = True,
                 holdout_ratio = 0.0,
                 random_seed = 0,
                 *args, **kwargs):

        super(SixDCDataset, self).__init__(*args, **kwargs)
        
        self.data_dir = data_dir
        self.use_mask = use_mask
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        self.sequence_names = sorted(next(os.walk(os.path.join(data_dir, 'test')))[1])
        self.occluded_object_ids = [1,2,5,6,8,9,10,11,12]
        self.object_names = self.sequence_names

        self.model_filenames = {}
        self.test_gt = {}
        self.primary_idxs = {}
        self.holdout_idxs = {}
        self.model_info = load_yaml(os.path.join(self.data_dir, 'models/models_info.yml'))
        
        self.model_scales = {}
        for k,v in self.model_info.items():
            self.model_scales[k] = 1/v['diameter']

        for j in range(1, 16):
            self.model_filenames[j] = os.path.join(self.data_dir, 'models/obj_{:02d}.ply'.format(j))

        for seq in self.sequence_names:
            gt = load_gt(os.path.join(self.data_dir, 'test/{}/gt.yml'.format(seq)))
            self.test_gt[seq] = {}
            for k, v in gt.items():
                self.test_gt[seq][k] = {}
                for d in v:
                    self.test_gt[seq][k][d['obj_id']] = d
            seq_size = len(self.test_gt[seq])
            self.holdout_idxs[seq] = np.random.choice(seq_size, 
                                                      int(seq_size*holdout_ratio), 
                                                      replace=False)
            self.primary_idxs[seq] = np.setdiff1d(np.arange(seq_size), self.holdout_idxs[seq])

        self.occluded_img_ids = {1:[],2:[],5:[],6:[],8:[],9:[],10:[],11:[],12:[]}
        for j,gt in self.test_gt['02'].items():
            for k, v in gt.items():
                self.occluded_img_ids[k].append(j)
        
        ####### HACK ########
        self.occluded_img_ids[10].remove(97)
        

        self.cam = load_cam_params(os.path.join(self.data_dir, 'camera.yml'))
        self.K = self.cam['K']
        self.seq = '02'
        self.obj = 1
        self.model = None
        self.model_idx = None
        self.data_models = SingularArray(self.obj)
        self.loop_truth = [1]
        self.holdout = False 
   
    def setSequence(self, seq, obj = None):
        self.seq = '{:02d}'.format(int(seq))
        if(self.seq == '02'):
            if(obj is None):
                self.obj = 2
            elif(obj in self.occluded_object_ids):
                self.obj = obj
            else:
                raise ValueError('Setting obj is only for objects 1, 2, 5, 6, and 8-12')
        elif(obj is None):
            self.obj = int(seq)
        else:
            raise ValueError('Setting obj is only valid of seq 02')

        self.data_models = SingularArray(self.obj)

    def occludedIdx(self, index):
        return self.occluded_img_ids[self.obj][index]

    def getModelFilename(self):
        return self.model_filenames[self.obj]

    def getModelScale(self):
        return self.model_scales[self.obj]

    def getIndex(self, index):
        if(self.seq == '02'):
            return self.occludedIdx(index)
        elif(self.holdout):
            return self.holdout_idxs[self.seq][index]
        return self.primary_idxs[self.seq][index]

    def getQuat(self, index):
        index = self.getIndex(index)
        rot = self.test_gt[self.seq][index][self.obj]['cam_R_m2c']
        mat = np.eye(4)
        mat[:3,:3] = rot
        quat = tf_trans.quaternion_from_matrix(mat)
        return quat

    def renderMask(self, index, delta = 15):
        #index = self.getIndex(index)
        if(self.model_idx != self.obj):
            self.model = load_ply(self.model_filenames[self.obj])
            self.model_idx = self.obj

        gt = self.test_gt[self.seq][index][self.obj]
        R_gt = gt['cam_R_m2c']
        t_gt = gt['cam_t_m2c']
        depth_img = load_depth(os.path.join(self.data_dir, 'test/{}/depth/{:04d}.png'.format(self.seq, index)))
        im_size = (depth_img.shape[1], depth_img.shape[0])
        depth_gt = renderer.render(self.model, im_size, self.K, R_gt, t_gt, clip_near=100,
                                   clip_far=10000, mode='depth')

        dist_img = misc.depth_im_to_dist_im(depth_img, self.K)
        dist_gt = misc.depth_im_to_dist_im(depth_gt, self.K)
        mask = visibility.estimate_visib_mask_gt(dist_img, dist_gt, delta)

        return mask

    def getImage(self, index, boarder_ratio=0.25):
        index = self.getIndex(index)
        gt = self.test_gt[self.seq][index][self.obj]
        assert(self.obj == gt['obj_id']), 'obj != obj_id'
        #img = load_im(os.path.join(self.data_dir, 'test/{}/rgb/{:04d}.png'.format(self.seq, index)))
        img = cv2.imread(os.path.join(self.data_dir, 'test/{}/rgb/{:04d}.png'.format(self.seq, index)))
        bbox = gt['obj_bb']
        ##cv2.
        min_bb = np.array([bbox[0], bbox[1]])
        max_bb = np.array([bbox[0]+bbox[2], bbox[1]+bbox[3]])
        min_bb = np.maximum(min_bb, 0)
        max_bb = np.minimum(max_bb, img.shape[1::-1])
        shape = max_bb - min_bb
        max_dim = max(shape)
        max_bb = max_bb + np.ceil((max_dim - shape)/2) + int(max_dim*boarder_ratio)
        min_bb = min_bb - np.floor((max_dim - shape)/2) - int(max_dim*boarder_ratio)

        margin = min(min(min_bb), min(np.array(img.shape[1::-1]) - max_bb))
        if(margin < 0):
            min_bb -= margin
            max_bb += margin

        x1, y1 = max_bb.astype(int)
        x0, y0 = min_bb.astype(int)
        if(self.use_mask):

            mask = cv2.imread(os.path.join(self.data_dir, 'test/{}/mask/{:04d}_{:02d}.png'.format(self.seq, index, self.obj)))
            if(mask is None):
                mask = self.renderMask(index)
            else:
                mask = mask[:,:,:1] #np.expand_dims(mask, axis=-1)
            img = np.concatenate([img, mask], axis=2)
        #print(margin)
        #print(index, bbox)
        #print(y0,y1, x0,x1)
        #print(img.shape)
        crop_img = self.preprocessImages(img[y0:y1, x0:x1, :], normalize_tensor = True)
        
        return crop_img

    def renderAndSaveMask(self, index):
        index = self.getIndex(index)
        mask = self.renderMask(index)
        save_im(os.path.join(self.data_dir, 
                    'test/{}/mask/{:04d}_{:02d}.png'.format(self.seq, index, self.obj)),
                    mask)
        #return self.model_filenames[self.obj]
        return mask.astype(float)

    def setRenderMasks(self):
        from pysixd import renderer, misc, visibility
        self.__getitem__ = self.renderAndSaveMask
        self.getImage = self.renderAndSaveMask

    def __len__(self):
        if(self.seq == '02'):
            return len(self.occluded_img_ids[self.obj])
        elif(self.holdout):
            return len(self.holdout_idxs[self.seq])
        return len(self.primary_idxs[self.seq])
