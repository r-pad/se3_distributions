# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:23:44 2018

@author: bokorn
"""
import tensorflow as tf
import numpy as np
import cv2

log_path = '/home/bokorn/results/alexnet_cls_dot_drill/2018-03-16_10-05-38/logs/train/events.out.tfevents.1521194833.compute-0-9.local'
log_path = '/home/bokorn/results/alexnet_reg_loop_car3/2018-03-16_11-51-19/logs/train/events.out.tfevents.1521201148.compute-0-11.local'

def convertSummary(val):
    if(val.HasField('simple_value')):
        return val.simple_value
    elif(val.HasField('obsolete_old_style_histogram')):
        raise NotImplementedError()
    elif(val.HasField('image')):
        return cv2.imdecode(np.frombuffer(val.image.encoded_image_string, np.uint8), cv2.IMREAD_COLOR)
    elif(val.HasField('histo')):
        return {'bins':val.histo.bucket, 'lims':val.histo.bucket_limit}                
    elif(val.HasField('audio')):
        raise NotImplementedError()
    elif(val.HasField('tensor')):
        raise NotImplementedError()
        return val.tensor.string_val
    else:
        raise ValueError('Invalid summary type %'.format(val))
        
    

def getSummaryData(path, tags):
    if(type(tags) is str):
        tags = [tags]
    data = {}
    for t in tags:
        data[t] = []
    try:
        for e in tf.train.summary_iterator(path):
            for v in e.summary.value:
                if v.tag in tags:
                    data[v.tag].append(convertSummary(v))
    except:
        print('Error')
        pass
    return data

car3_loop_train_path = '/home/bokorn/results/alexnet_reg_loop_car3/2018-03-16_11-51-19/logs/train/events.out.tfevents.1521201148.compute-0-11.local'
car3_loop_valid_path = '/home/bokorn/results/alexnet_reg_loop_car3/2018-03-16_11-51-19/logs/valid/events.out.tfevents.1521201148.compute-0-11.local'

car3_nolo_train_path = '/home/bokorn/results/alexnet_reg_car3/2018-03-16_11-51-49/logs/train/events.out.tfevents.1521201171.compute-0-11.local'
car3_nolo_valid_path = '/home/bokorn/results/alexnet_reg_car3/2018-03-16_11-51-49/logs/valid/events.out.tfevents.1521201171.compute-0-11.local'

loop_train_data = getSummaryData(car3_loop_train_path, ['mean_err', 'cross_mean_err'])
loop_valid_data = getSummaryData(car3_loop_valid_path, ['mean_err', 'cross_mean_err'])
nolo_train_data = getSummaryData(car3_nolo_train_path, ['mean_err', 'cross_mean_err'])
nolo_valid_data = getSummaryData(car3_nolo_valid_path, ['mean_err', 'cross_mean_err'])

n, bins, patches = plt.hist(np.array(axis_angles)*180/np.pi, 1000, density=True, facecolor='g', alpha=0.75)
plt.savefig("/home/bokorn/results/test/imgs/angle.png")

import IPython; IPython.embed()