# -*- coding: utf-8 -*-
"""
Created on Tues at some point in time

@author: bokorn
"""
import cv2
import numpy as np
import glob

import se3_distributions.utils.transformations as tf_trans
from se3_distributions.datasets.image_dataset import PoseImageDataset                                     

fx=572.41140; px=325.26110; fy=573.57043; py=242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

def loadMatFile(filename, check_size=False, header=True):
    with open(filename, 'r') as f:
        data=f.read().splitlines()
    #if(header):
    #    size = [int(d) for d in data[0].split(' ')]
    ret = np.array([list(filter(None, d.split(' '))) for d in data[header:]], dtype=float)
    #if(check_size and header):
    #    assert ret.shape == tuple(size), 'Array size {} does not match first line {}'.format(ret.shape, tuple(size))
    return ret

class LinemodDataset(PoseImageDataset):
    def __init__(self, data_folders, use_mask = False,
                 *args, **kwargs):

        super(LinemodDataset, self).__init__(*args, **kwargs)
        
        self.use_mask = use_mask

        if(type(data_folders) is str and data_folders[-4:] == '.txt'):
            with open(data_folders, 'r') as f:    
                data_folders = f.read().split()

        if(type(data_folders) is list):
            files = []
            for folder in data_folders:
                files.extend(glob.glob(folder + '/**/*.jpg', recursive=True))
        elif(type(data_folders) is str):
            files = glob.glob(data_folders + '/**/*.jpg', recursive=True)
        else:
            raise AssertionError('Invalid data_folders type {}'.format(type(data_folders)))

        self.data_filenames = []
        self.data_models = []

        self.model_filenames = {}
        self.model_idxs = {}
        self.model_bbox = {}
        self.model_pts = {}

        for j, path in enumerate(files):
            path_split = path.split('/')
            model = path_split[-3]
            filename = path_split[-1].replace('color','{}').replace('.jpg','.{}')

            if(model in self.model_idxs):
                self.model_idxs[model].append(j)
            else:
                self.model_idxs[model] = [j]

                self.model_filenames[model] = '/'.join(path_split[:-2] + ['mesh.ply'])

                pts_filename = '/'.join(path_split[:-2] + ['object.xyz'])
                self.model_pts[model] = loadMatFile(pts_filename)[:,:3].transpose()
                minmax = np.array([np.min(self.model_pts[model], axis=1),
                                   np.max(self.model_pts[model], axis=1)])
                bbox = np.reshape(np.stack(np.meshgrid(minmax[:,0], 
                                                       minmax[:,1], 
                                                       minmax[:,2])),(3,-1))

                self.model_bbox[model] = bbox

            self.data_filenames.append('/'.join(path_split[:-1] + [filename]))
            self.data_models.append(model)

    def getQuat(self, index):
        rot = loadMatFile(self.data_filenames[index].format('rot','rot')) 
        mat = np.eye(4)
        mat[:3,:3] = rot
        quat = tf_trans.quaternion_from_matrix(mat)
        return quat

    def getMask(self, model, rot, trans, image_size):
        pt_img = np.zeros((image_size[0]+2, image_size[1]+2, 1), dtype=np.uint8)
        mask = np.ones(image_size[:2] + (1,), dtype=np.uint8)*255
        pts = self.model_pts[model]
        pts = np.matmul(rot,pts) + trans
        pxls = np.matmul(intrinsic_matrix, pts)
        pxls = (pxls[:2]/pxls[2,:]).astype(int)
        for px in pxls.transpose():
            cv2.circle(pt_img, tuple(px+1), 0, 255, thickness=0)
        cv2.floodFill(mask, pt_img, (0,0), 0)
        return mask

    def getImage(self, index, boarder_ratio=0.25):
        model = self.data_models[index]
        img = cv2.imread(self.data_filenames[index].format('color','jpg'), cv2.IMREAD_UNCHANGED)
        rot = loadMatFile(self.data_filenames[index].format('rot','rot')) 
        trans= loadMatFile(self.data_filenames[index].format('tra','tra'))
        pts = np.matmul(rot, self.model_bbox[model]) + trans
        pxls = np.matmul(intrinsic_matrix, pts)
        pxls = pxls[:2]/pxls[2,:]
        min_bb = np.min(pxls,axis=1).astype(int)       
        max_bb = np.max(pxls,axis=1).astype(int)       
        shape = max_bb - min_bb
        max_dim = max(shape)
        max_bb = max_bb + np.ceil((max_dim - shape)/2) + int(max_dim*boarder_ratio)
        min_bb = min_bb - np.floor((max_dim - shape)/2) - int(max_dim*boarder_ratio)
        og_min_bb = min_bb.copy()
        og_max_bb = max_bb.copy()
        margin = min(min(min_bb), min(np.array(img.shape[1::-1]) - max_bb))
        if(margin < 0):
            min_bb -= margin
            max_bb += margin

        x1, y1 = max_bb.astype(int)
        x0, y0 = min_bb.astype(int)
        if(self.use_mask):
            mask = cv2.imread(self.data_filenames[index].format('mask','png'), cv2.IMREAD_UNCHANGED)
            if(mask is None):
                mask = self.getMask(model, rot, trans, img.shape)
            else:
                mask = np.expand_dims(mask, axis=-1)
            img = np.concatenate([img, mask], axis=2)
        crop_img = self.preprocessImages(img[y0:y1, x0:x1, :], normalize_tensor = True)
        return crop_img

    def renderMasks(self, index):
        model = self.data_models[index]
        img = cv2.imread(self.data_filenames[index].format('color','jpg'), cv2.IMREAD_UNCHANGED)
        rot = loadMatFile(self.data_filenames[index].format('rot','rot')) 
        trans= loadMatFile(self.data_filenames[index].format('tra','tra'))
        mask = self.getMask(model, rot, trans, img.shape)
        cv2.imwrite(self.data_filenames[index].format('mask','png'), mask)
        return model

    def setRenderMasks(self):
        self.loop_truth = [1]
        self.__getitem__ = self.renderMasks

    def __len__(self):
        return len(self.data_filenames)
