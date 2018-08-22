import numpy as np
import generic_pose.losses.distance_loss as dl    
import generic_pose.losses.distance_loss as dl    
import generic_pose.losses.distance_loss as dl    
import torch
import generic_pose.losses.distance_loss as dl    

lbls = torch.randn(10,4)
lbls = lbls.div(torch.norm(lbls,p=2,dim=1).unsqueeze(1).expand_as(lbls))
lbls = lbls * torch.sign(lbls[:,3:]) 

lbl_th = dl.tensor2Angle(lbls)
th = 20*np.pi/180
lbl_o = dl.expDecayTheta(lbls, th)

import IPython; IPython.embed()


