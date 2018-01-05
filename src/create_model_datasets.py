# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:34:37 2018

@author: bokorn
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:13:09 2017

@author: bokorn
"""

from model_dataset_generator import ModelDataSetGenerator
import numpy as np

data_dir = '/media/bokorn/ExtraDrive1/shapenetcore/'
dataset_gen = ModelDataSetGenerator(data_dir)


#chair_model_name = '1bcd9c3fe6c9087e593ebeeedbff73b'
#
#chairs_1_train, chairs_1_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80)
#
#print('chairs_1_train:', len(chairs_1_train))
#print('chairs_1_valid:', len(chairs_1_valid))
#with open('../training_sets/model_sets/chairs_1_train.txt', 'w') as f:
#    for fn in chairs_1_train:
#        f.write("%s\n" % fn)
#with open('../training_sets/model_sets/chairs_1_valid.txt', 'w') as f:
#    for fn in chairs_1_valid:
#        f.write("%s\n" % fn)
#        
#chairs_1_deg_5_train, chairs_1_deg_5_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 5.0*np.pi/180.0)
#chairs_1_deg_10_train, chairs_1_deg_10_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 10.0*np.pi/180.0)
#chairs_1_deg_20_train, chairs_1_deg_20_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 20.0*np.pi/180.0)
#chairs_1_deg_45_train, chairs_1_deg_45_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 45.0*np.pi/180.0)
#chairs_1_deg_90_train, chairs_1_deg_90_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 90.0*np.pi/180.0)
#chairs_1_deg_135_train, chairs_1_deg_135_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 135.0*np.pi/180.0)
#chairs_1_deg_180_train, chairs_1_deg_180_valid = dataset_gen.modelTrainValidSplit(chair_model_name, train_ratio=.80, max_orientation_offset = 180.0*np.pi/180.0)
#
##print('chairs_1_deg_5_train:', len(chairs_1_deg_5_train))
##print('chairs_1_deg_5_valid:', len(chairs_1_deg_5_valid))
##with open('../training_sets/model_sets/chairs_1_deg_5_train.txt', 'w') as f:
##    for fn in chairs_1_deg_5_train:
##        f.write("%s\n" % fn)
##with open('../training_sets/model_sets/chairs_1_deg_5_valid.txt', 'w') as f:
##    for fn in chairs_1_deg_5_valid:
##        f.write("%s\n" % fn)
##        
##print('chairs_1_deg_10_train:', len(chairs_1_deg_10_train))
##print('chairs_1_deg_10_valid:', len(chairs_1_deg_10_valid))
##with open('../training_sets/model_sets/chairs_1_deg_10_train.txt', 'w') as f:
##    for fn in chairs_1_deg_10_train:
##        f.write("%s\n" % fn)
##with open('../training_sets/model_sets/chairs_1_deg_10_valid.txt', 'w') as f:
##    for fn in chairs_1_deg_10_valid:
##        f.write("%s\n" % fn)
#
#print('chairs_1_deg_20_train:', len(chairs_1_deg_20_train))
#print('chairs_1_deg_20_valid:', len(chairs_1_deg_20_valid))
#with open('../training_sets/model_sets/chairs_1_deg_20_train.txt', 'w') as f:
#    for fn in chairs_1_deg_20_train:
#        f.write("%s\n" % fn)
#with open('../training_sets/model_sets/chairs_1_deg_20_valid.txt', 'w') as f:
#    for fn in chairs_1_deg_20_valid:
#        f.write("%s\n" % fn)
#        
#print('chairs_1_deg_45_train:', len(chairs_1_deg_45_train))
#print('chairs_1_deg_45_valid:', len(chairs_1_deg_45_valid))
#with open('../training_sets/model_sets/chairs_1_deg_45_train.txt', 'w') as f:
#    for fn in chairs_1_deg_45_train:
#        f.write("%s\n" % fn)
#with open('../training_sets/model_sets/chairs_1_deg_45_valid.txt', 'w') as f:
#    for fn in chairs_1_deg_45_valid:
#        f.write("%s\n" % fn)
#        
#print('chairs_1_deg_90_train:', len(chairs_1_deg_90_train))
#print('chairs_1_deg_90_valid:', len(chairs_1_deg_90_valid))
#with open('../training_sets/model_sets/chairs_1_deg_90_train.txt', 'w') as f:
#    for fn in chairs_1_deg_90_train:
#        f.write("%s\n" % fn)
#with open('../training_sets/model_sets/chairs_1_deg_90_valid.txt', 'w') as f:
#    for fn in chairs_1_deg_90_valid:
#        f.write("%s\n" % fn)
#        
#print('chairs_1_deg_135_train:', len(chairs_1_deg_135_train))
#print('chairs_1_deg_135_valid:', len(chairs_1_deg_135_valid))
#with open('../training_sets/model_sets/chairs_1_deg_135_train.txt', 'w') as f:
#    for fn in chairs_1_deg_135_train:
#        f.write("%s\n" % fn)
#with open('../training_sets/model_sets/chairs_1_deg_135_valid.txt', 'w') as f:
#    for fn in chairs_1_deg_135_valid:
#        f.write("%s\n" % fn)
#        
#print('chairs_1_deg_180_train:', len(chairs_1_deg_180_train))
#print('chairs_1_deg_180_valid:', len(chairs_1_deg_180_valid))
#with open('../training_sets/model_sets/chairs_1_deg_180_train.txt', 'w') as f:
#    for fn in chairs_1_deg_180_train:
#        f.write("%s\n" % fn)
#with open('../training_sets/model_sets/chairs_1_deg_180_valid.txt', 'w') as f:
#    for fn in chairs_1_deg_180_valid:
#        f.write("%s\n" % fn)

car_class = '02958343'

cars_3_train, cars_3_valid = dataset_gen.classTrainValidSplit(car_class, num_models=6, train_ratio=.5)
cars_10_train, cars_10_valid = dataset_gen.classTrainValidSplit(car_class, num_models=20, train_ratio=.5)
cars_100_train, cars_100_valid = dataset_gen.classTrainValidSplit(car_class, num_models=200, train_ratio=.5)

print('cars_3_train:', len(cars_3_train))
print('cars_3_valid:', len(cars_3_valid))
with open('../training_sets/model_sets/cars_3_train.txt', 'w') as f:
    for fn in cars_3_train:
        f.write("%s\n" % fn)
with open('../training_sets/model_sets/cars_3_valid.txt', 'w') as f:
    for fn in cars_3_valid:
        f.write("%s\n" % fn)
        
print('cars_10_train:', len(cars_10_train))
print('cars_10_valid:', len(cars_10_valid))
with open('../training_sets/model_sets/cars_10_train.txt', 'w') as f:
    for fn in cars_10_train:
        f.write("%s\n" % fn)
with open('../training_sets/model_sets/cars_10_valid.txt', 'w') as f:
    for fn in cars_10_valid:
        f.write("%s\n" % fn)
        
print('cars_100_train:', len(cars_100_train))
print('cars_100_valid:', len(cars_100_valid))
with open('../training_sets/model_sets/cars_100_train.txt', 'w') as f:
    for fn in cars_100_train:
        f.write("%s\n" % fn)
with open('../training_sets/model_sets/cars_100_valid.txt', 'w') as f:
    for fn in cars_100_valid:
        f.write("%s\n" % fn)
        
import IPython; IPython.embed()