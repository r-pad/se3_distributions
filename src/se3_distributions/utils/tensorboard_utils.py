# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 18:47:55 2018

@author: bokorn
"""
import cv2
import numpy as np
import tensorflow as tf

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
                    data[v.tag].append([e.step, convertSummary(v)])
    except Exception as e:
        print(e)
        pass
    return data

def getWallTime(path):
    data = []
    try:
        for e in tf.train.summary_iterator(path):
            data.append([e.step, e.wall_time])
    except Exception as e:
        print(e)
        pass
    return data
    
