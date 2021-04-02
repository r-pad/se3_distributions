import numpy as np
import torch
import torch.nn.functional as F

class IsoBingham(torch.nn.Module):
    def __init__(self, feature_size, num_obj):
        super(IsoBingham, self).__init__()
        
        self.conv1_s = torch.nn.Conv1d(feature_size, 640, 1)
        
        self.conv2_s = torch.nn.Conv1d(640, 256, 1)
        
        self.conv3_s = torch.nn.Conv1d(256, 128, 1)
        
        self.conv4_s = torch.nn.Conv1d(128, num_obj*1, 1) #sigma

        self.num_obj = num_obj
        self.feature_size = feature_size
        
    def forward(self, feature, obj):
        bs = feature.shape[0]
        feature = feature.view(bs, self.feature_size, -1)

        sx = F.relu(self.conv1_s(feature))      
        sx = F.relu(self.conv2_s(sx))
        sx = F.relu(self.conv3_s(sx))
        sx = torch.abs(self.conv4_s(sx)).view(bs, self.num_obj, 1, -1)
        
        out_sx = torch.cat([torch.index_select(sx[b], 0, obj[b]) for b in range(bs)])
        out_sx = out_sx.contiguous().transpose(2, 1).contiguous()
        
        return out_sx
    
class DuelBingham(torch.nn.Module):
    def __init__(self, feature_size, num_obj):
        super(DuelBingham, self).__init__()
        
        self.conv1_bq = torch.nn.Conv1d(feature_size, 640, 1)
        self.conv1_bz = torch.nn.Conv1d(feature_size, 640, 1)

        self.conv2_bq = torch.nn.Conv1d(640, 256, 1)
        self.conv2_bz = torch.nn.Conv1d(640, 256, 1)
        
        self.conv3_bq = torch.nn.Conv1d(256, 128, 1)
        self.conv3_bz = torch.nn.Conv1d(256, 128, 1)

        self.conv4_bq = torch.nn.Conv1d(128, num_obj*4, 1) #duel quat
        self.conv4_bz = torch.nn.Conv1d(128, num_obj*3, 1) #concentration

        self.num_obj = num_obj
        self.feature_size = feature_size
        
    def forward(self, feature, obj):
        bs = feature.shape[0]
        feature = feature.view(bs, self.feature_size, -1)

        bqx = F.relu(self.conv1_bq(feature))      
        bzx = F.relu(self.conv1_bz(feature))      

        bqx = F.relu(self.conv2_bq(bqx))
        bzx = F.relu(self.conv2_bz(bzx))

        bqx = F.relu(self.conv3_bq(bqx))
        bzx = F.relu(self.conv3_bz(bzx))

        bqx = self.conv4_bq(bqx).view(bs, self.num_obj, 4, -1)
        bzx = torch.abs(self.conv4_bz(bzx)).view(bs, self.num_obj, 3, -1)
        
        out_bqx = torch.cat([torch.index_select(bqx[b], 0, obj[b]) for b in range(bs)])
        out_bzx = torch.cat([torch.index_select(bzx[b], 0, obj[b]) for b in range(bs)])
        
        out_bqx = out_bqx.contiguous().transpose(2, 1).contiguous()
        out_bzx = out_bzx.contiguous().transpose(2, 1).contiguous()
        
        return out_bqx, out_bzx
