# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:52:16 2018

@author: bokorn
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.models.alexnet import alexnet

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

model = alexnet()
model.eval()
model.cuda()
optimizer = Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

img = to_var(torch.from_numpy(np.random.rand(10,3,224,224)).float())
val = to_var(torch.from_numpy(np.zeros(10)).long())

optimizer.zero_grad()
res = model(img)
loss = criterion(res, val)
loss.backward()
optimizer.step()

print(loss)