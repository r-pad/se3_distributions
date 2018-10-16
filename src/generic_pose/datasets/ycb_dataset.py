# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time
@author: bokorn with some code pulled from https://github.com/yuxng/PoseCNN/blob/master/lib/datasets/lov.py
"""

import os
import cv2
import numpy as np
import scipy.io as sio

from generic_pose.datasets.image_dataset import PoseImageDataset
import generic_pose.utils.transformations as tf_trans

class YCBDataset(PoseImageDataset):
    def __init__(self, data_dir, image_set, 
                 obj = None, 
                 *args, **kwargs):

        super(YCBDataset, self).__init__(*args, **kwargs)
        self.data_dir = data_dir

	#self.classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
        #                '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
        #                '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self.classes = ['__background__']
        with open(os.path.join(self.data_dir, 'image_sets', 'classes.txt')) as f:
            self.classes.extend([x.rstrip('\n') for x in f.readlines()])

        self.num_classes = len(self.classes)
        self.symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])

        self.image_set = image_set
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
        return image_index

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
                    obj_image_sets[bb[0]].append(fn)
        
        for k,v in obj_image_sets.items():
            with open(os.path.join(self.data_dir, 'image_sets', k+'_'+self.image_set+'.txt'), 'w') as f:
                f.write('\n'.join(v))

    def getObjectName(self):
        return self.classes[self.obj]

    def setObject(self, obj):
        self.obj = obj
        self.obj_name = self.getObjectName()
        self.data_filenames = self.loadImageSet()

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
        mask = (cv2.imread(image_prefix + '-label.png')[:,:,:1] == self.obj).astype('uint8')
        #bb = cv2.boundingRect(cv2.findContours((lbls[:,:,0]==21).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1][0])
        #x0, y0 = bb[:2]
        #x1, y1 = np.add(bb[:2], bb[2:])

        with open(image_prefix + '-box.txt') as f:
            bboxes = [x.rstrip('\n').split(' ') for x in f.readlines()]
        for bb in bboxes:
            if(bb[0] == self.classes[self.obj]):
                (x0, y0, x1, y1) = np.array(bb[1:], dtype='float')
                break
        
        h = y1 - y0
        w = x1 - x0
        img_h, img_w = img.shape[:2]
        if(y0 > 0):
            y0 = max(0, y0 - boarder_ratio*h)
        if(x0 > 0):
            x0 = max(0, x0 - boarder_ratio*w)
        if(y1 < img_h):
            y1 = min(img_h, y1 + boarder_ratio*h)
        if(x1 < img_w):
            x1 = min(img_w, x1 + boarder_ratio*w)
        
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)

        img = np.concatenate([img, mask], axis=2)
        crop_img = self.preprocessImages(img[y0:y1, x0:x1, :], normalize_tensor = True)
        return crop_img

    def __len__(self):
        return len(self.data_filenames)
 
