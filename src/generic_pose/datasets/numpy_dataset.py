# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:38:19 2018

@author: bokorn
"""
import cv2
import numpy as np
import glob

from generic_pose.datasets.image_dataset import PoseImageDataset
import resource
import pickle

class NumpyImageDataset(PoseImageDataset):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    def __init__(self, data_folders,
                 model_filenames = None,
                 num_model_imgs = float('inf'),
                 pkl_save_filename = None,
                 *args, **kwargs):

        super(NumpyImageDataset, self).__init__(*args, **kwargs)
        
        self.model_filenames = model_filenames

        if(type(data_folders) is str and data_folders[-4:] == '.txt'):
            print('Loading Filenames')
            with open(data_folders, 'r') as f:    
                data_folders = f.read().split()

        if(type(data_folders) is str and data_folders[-4:] == '.pkl'):
            with open(data_folders, 'rb') as f:
                data = pickle.load(f)
            self.data_filenames = data['data_filenames']
            self.data_models = data['data_models']
            self.data_classes = data['data_classes']
            self.quats = data['quats']
            self.model_idxs = data['model_idxs']
            self.model_class = data['model_class']
            self.class_idxs = data['class_idxs']
        else:
            print('Searching for pngs')
            if(type(data_folders) is list):
                files = []
                for folder in data_folders:
                    files.extend(glob.glob(folder+'/**/*.png', recursive=True))
            elif(type(data_folders) is str):
                files = glob.glob(data_folders+'/**/*.png', recursive=True)
            else:
                raise AssertionError('Invalid data_folders type {}'.format(type(data_folders)))

            self.data_filenames = []
            self.data_models = []
            self.data_classes = []
            self.quats = []

            self.model_idxs = {}
            self.model_class = {}
            self.class_idxs = {}

            print('Processing Files')
            for j, path in enumerate(files):
                if(j >= num_model_imgs):
                    break
                
                [model_class, model, filename] = '.'.join(path.split('.')[:-1]).split('/')[-3:]

                if(model_class in self.class_idxs):
                    self.class_idxs[model_class].append(model)
                else:
                    self.class_idxs[model_class] = [model]
                    self.model_class[model] = model_class

                if(model in self.model_idxs):
                    self.model_idxs[model].append(j)
                else:
                    self.model_idxs[model] = [j]
                file_prefix = '.'.join(path.split('.')[:-1])
                self.data_filenames.append(file_prefix)
                self.data_models.append(model)
                self.data_classes.append(model_class)
                self.quats.append(np.load(file_prefix + '.npy'))
            if(pkl_save_filename is not None):
                print('Pickling Data')
                with open(pkl_save_filename, "wb" ) as f:
                    pickle.dump({'data_filenames':self.data_filenames,
                                 'data_models':self.data_models,
                                 'data_classes':self.data_classes,
                                 'quats':self.quats,
                                 'model_idxs':self.model_idxs,
                                 'model_class':self.model_class,
                                 'class_idxs':self.class_idxs}, f)


    def getQuat(self, index):
        return self.quats[index]
    '''
        q = np.load(self.data_filenames[index] + '.npy')
        return q

        try:
            with np.load(self.data_filenames[index] + '.npy') as q:
                return 
        except AttributeError as err:
            raise AttributeError('Error on file index {} ({}): {}'.format(index, self.data_filenames[index] + '.npy', err))
    '''
    def getImage(self, index):
        return cv2.imread(self.data_filenames[index] + '.png', cv2.IMREAD_UNCHANGED), None
       
    def __len__(self):
        return len(self.data_filenames)
