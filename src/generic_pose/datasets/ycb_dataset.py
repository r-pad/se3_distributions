# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn with some code pulled from https://github.com/yuxng/PoseCNN/blob/master/lib/datasets/lov.py
"""

import os
import cv2
import torch
import numpy as np
import scipy.io as sio

from generic_pose.datasets.image_dataset import PoseImageDataset
from generic_pose.utils import SingularArray
import generic_pose.utils.transformations as tf_trans
from generic_pose.utils.pose_processing import viewpoint2Pose
from generic_pose.utils.image_preprocessing import cropAndPad

def ycbRenderTransform(q):
    trans_quat = q.copy()
    trans_quat = tf_trans.quaternion_multiply(trans_quat, tf_trans.quaternion_about_axis(-np.pi/2, [1,0,0]))
    return viewpoint2Pose(trans_quat)

class YCBDataset(PoseImageDataset):
    def __init__(self, data_dir, image_set, 
                 obj = None, use_syn_data = False, 
                 *args, **kwargs):

        super(YCBDataset, self).__init__(*args, **kwargs)
        self.use_syn_data = use_syn_data
        self.data_dir = data_dir

	#self.classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
        #                '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
        #                '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self.classes = ['__background__']
        with open(os.path.join(self.data_dir, 'image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])

        self.num_classes = len(self.classes)
        self.model_filenames = {}
        for j in range(1, self.num_classes):
            self.model_filenames[j] = os.path.join(self.data_dir, 'models', self.classes[j], 'textured.obj')

        self.symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        self.image_set = image_set
        self.append_rendered = False
        #self.data_filenames = self.loadImageSet()
        if(obj is not None):
            self.setObject(obj)
        #self.points, self.points_all = self.load_object_points()
	
    def loadObjectPoints(self):
        points = [[] for _ in xrange(len(self.classes))]
        num = np.inf
        
        for i in xrange(1, len(self.classes)):
            point_file = os.path.join(self.data_dir, 'models', self.classes[i], 'points.xyz')
            #print point_file
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self.num_classes, num, 3), dtype=np.float32)
        for i in xrange(1, len(self.classes)):
            points_all[i, :, :] = points[i][:num, :]

        return points, points_all

    def loadImageSet(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self.data_dir, 'image_sets', 
                                      self.obj_name+'_'+self.image_set+'.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        if(self.use_syn_data):
            syn_set_file = os.path.join(self.data_dir, 'image_sets',
                                        self.obj_name+'_syn.txt')
            assert os.path.exists(syn_set_file), \
                'Path does not exist: {}'.format(syn_set_file)
            with open(syn_set_file) as f:
                image_index.extend([x.rstrip('\n') for x in f.readlines()])

        return image_index

    def splitImages(self):
        self.videos = {}
        for fn in self.data_filenames:
            vid = fn.split('/')[0]
            if(vid in self.videos.keys()):
                self.videos[vid].append(fn)
            else:
                self.videos[vid] = [fn]


    def generateObjectImageSet(self):
        obj_image_sets = {}
        for cls in self.classes[1:]:
            obj_image_sets[cls] = []

        image_set_file = os.path.join(self.data_dir, 'image_sets', self.image_set+'.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            filenames = [x.rstrip('\n') for x in f.readlines()]
 
        for fn in filenames:
            with open(os.path.join(self.data_dir, 'data', fn  + '-box.txt')) as f:
                bboxes = [x.rstrip('\n').split(' ') for x in f.readlines()]
                for bb in bboxes:
                    (x0, y0, x1, y1) = np.array(bb[1:], dtype='float')
                    if(x1-x0 > 0 and y1-y0 > 0):
                        obj_image_sets[bb[0]].append(fn)
        
        for k,v in obj_image_sets.items():
            with open(os.path.join(self.data_dir, 'image_sets', k+'_'+self.image_set+'.txt'), 'w') as f:
                f.write('\n'.join(v))

    def generateSyntheticImageSet(self):
        import glob
        syn_data_dir = os.path.join(self.data_dir, 'data_syn')
        #label_filenames = sorted(glob.glob(os.path.join(syn_data_dir,'-label.png')))
        data_filenames =  sorted(glob.glob(os.path.join(syn_data_dir,'*-meta.mat')))
	
        obj_image_sets = {}
        for cls in self.classes[1:]:
            obj_image_sets[cls] = []
 
        for fn in data_filenames:
            data_prefix = '-'.join(fn.split('/')[-1].split('-')[:-1])
            data = sio.loadmat(os.path.join(self.data_dir, 'data_syn', data_prefix + '-meta.mat'))
            cls_idxs = data['cls_indexes'].flatten().astype(int)
            
            img = cv2.imread(os.path.join(self.data_dir, 'data_syn', data_prefix  + '-label.png'))
            data_prefix = os.path.join('..', 'data_syn', data_prefix)
            for idx in cls_idxs: 
                if(np.sum(img == idx) > 0):
                    obj_image_sets[self.classes[idx]].append(data_prefix)
        for k,v in obj_image_sets.items():
            with open(os.path.join(self.data_dir, 'image_sets', k+'_syn.txt'), 'w') as f:
                f.write('\n'.join(v))

    def generateRenderedImages(self):
        from model_renderer.pose_renderer import BpyRenderer
        for cls_idx in range(8, len(self.classes)):
            self.setObject(cls_idx)
            renderer = BpyRenderer(transform_func = ycbRenderTransform)
            renderer.loadModel(self.getModelFilename(), emit = 0.5)
            print("Rendering Object {}: {}".format(cls_idx, self.getObjectName()))
            renderPoses = renderer.renderPose

            render_filenames = []
            render_quats = []
            for fn in self.data_filenames: 
                data = sio.loadmat(os.path.join(self.data_dir, 'data', fn + '-meta.mat'))
                pose_idx = np.where(data['cls_indexes'].flatten()==self.obj)[0][0]
                mat = np.eye(4)
                mat[:3,:3] = data['poses'][:3,:3,pose_idx]
                render_filenames.append(os.path.join(self.data_dir, 'data', fn + '-{}-render.png'.format(cls_idx)))
                render_quats.append(tf_trans.quaternion_from_matrix(mat))

            renderPoses(render_quats, camera_dist = 0.33, image_filenames = render_filenames)

    def getObjectName(self):
        return self.classes[self.obj]

    def setObject(self, obj):
        self.obj = obj
        self.obj_name = self.getObjectName()
        self.data_filenames = self.loadImageSet()
        self.data_models = SingularArray(self.obj)

    def getModelFilename(self):
        return os.path.join(self.data_dir, 'models', self.classes[self.obj], 'textured.obj')

    def getQuat(self, index):
        data = sio.loadmat(os.path.join(self.data_dir, 'data', self.data_filenames[index] + '-meta.mat'))
        pose_idx = np.where(data['cls_indexes'].flatten()==self.obj)[0][0]
        mat = np.eye(4)
        mat[:3,:3] = data['poses'][:3,:3,pose_idx]
        quat = tf_trans.quaternion_from_matrix(mat)
        return quat

    def getImage(self, index, boarder_ratio=0.25):
        image_prefix = os.path.join(self.data_dir, 'data', self.data_filenames[index])
        img = cv2.imread(image_prefix + '-color.png')
        mask = 255*(cv2.imread(image_prefix + '-label.png')[:,:,:1] == self.obj).astype('uint8')
        if(np.sum(mask) == 0):
            #import IPython; IPython.embed()
            print('Index {} invalid for {} ({}:{})'.format(index, self.getObjectName(),
                    self.image_set, image_prefix))
            return None
        img = np.concatenate([img, mask], axis=2)
        crop_img = self.preprocessImages(cropAndPad(img), normalize_tensor = True, augment_img = True)
        if(self.append_rendered):
            #import IPython; IPython.embed();
            rendered_img = self.preprocessImages(cv2.imread(image_prefix + '-color.png'), normalize_tensor = True)
            crop_img = torch.cat((crop_img, rendered_img), 0)
       
        return crop_img

    def __len__(self):
        return len(self.data_filenames)
 
