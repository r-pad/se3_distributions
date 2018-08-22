# -*- coding: utf-8 -*-
"""
@author: bokorn
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from generic_pose.loss.distance_loss import robustRegressionLoss
from generic_pose.training.utils import to_var
data = np.random.randn(10000)*10
labels = test_data**2 + 10

data = to_var(data)

network = nn.Sequential(
                        nn.Linear(1, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 1)
                        )
num_batches = 100
for j in range(num_batches):
    optimizer.zero_grad()
    pred = network.forward(data)
    loss = robustRegressionLoss(pred, labels)
    print(loss)

