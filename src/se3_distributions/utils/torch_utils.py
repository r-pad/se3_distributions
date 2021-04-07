# -*- coding: utf-8 -*-
"""
@author: bokorn
"""

import torch
import gc
import inspect
import subprocess

def get_gpu_memory_map():   
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                      '--format=csv,nounits,noheader'])
    return float(result)

def find_names(obj):
    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names

def getTensors():
    info = []
    objs = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or torch.is_tensor(getattr(obj, 'data', None)):
                #print(type(obj), obj.size())
                info.append([type(obj), obj.size()])
                objs.append(obj)
        except:
            pass
    return info, objs
