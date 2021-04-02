import numpy as np
import torch
from object_pose_utils.utils import to_np


from quat_math import quaternion_matrix, quaternion_from_matrix

def evaluateDenseFusion(model, img, points, choose, obj, 
        refine_model = None, return_all = False, use_global_feat=True):
    df_obj_idx = obj - 1
    img = img.unsqueeze(0).cuda()
    points = points.unsqueeze(0).cuda()
    choose = choose.unsqueeze(0).cuda()
    df_obj_idx = df_obj_idx.unsqueeze(0).cuda()
    
    pred_r, pred_t, pred_c, emb, feat_local, feat_global = model.allFeatures(img, points, choose, df_obj_idx)
    
    if(use_global_feat):
        feat = feat_global
    else:
        feat = feat_local

    #feat = model.globalFeature(img, points, choose, df_obj_idx)
    #pred_r, pred_t, pred_c, emb = model(img, points, choose, df_obj_idx)

    how_max, which_max = torch.max(pred_c, 1)
    if(pred_t.shape[1] == points.shape[1]):
        pred_t = points[0] + pred_t[0,:]
    pred_q = pred_r[0,:,[1,2,3,0]]
    pred_q /= torch.norm(pred_q, dim=1).view(-1,1)

    if(return_all):
        return pred_q, pred_t, pred_c, feat

    max_c = to_np(how_max)
    max_q = to_np(pred_q[which_max.item()])
    max_t = to_np(pred_t[which_max.item()])
    
    if(not use_global_feat):
        feat = feat[:, :, which_max.item()]
        
    if(refine_model is not None):
        num_points = points.shape[1]
        for _ in range(0, 2):
            T = torch.Tensor(max_t).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            mat = quaternion_matrix(max_q)
            R = torch.Tensor(mat[:3, :3]).cuda().view(1, 3, 3)
            mat[0:3, 3] = max_t

            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refine_model(new_points, emb, df_obj_idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            pred_q = pred_r[0,0,[1,2,3,0]]

            pred_q = to_np(pred_q)
            pred_t = to_np(pred_t)
            pred_mat = quaternion_matrix(pred_q)
            pred_mat[0:3, 3] = pred_t

            final_mat = np.dot(mat, pred_mat)
            max_q = quaternion_from_matrix(final_mat)
            max_t = final_mat[0:3, 3]
            
    return max_q, max_t, feat


def fullEvaluateDenseFusion(model, img, points, choose, obj,
                            refine_model = None, return_all = False):
    df_obj_idx = obj - 1
    img = img.unsqueeze(0).cuda()
    points = points.unsqueeze(0).cuda()
    choose = choose.unsqueeze(0).cuda()
    df_obj_idx = df_obj_idx.unsqueeze(0).cuda()

    if(hasattr(model, 'allFeatures')):
        res = model.allFeatures(img, points, choose, df_obj_idx)
        pred_r, pred_t, pred_c, emb, feat_all, feat = res

    else:
        feat = model.globalFeature(img, points, choose, df_obj_idx)
        pred_r, pred_t, pred_c, emb = model(img, points, choose, df_obj_idx)
        feat_all = None

    how_max, which_max = torch.max(pred_c, 1)
    if(len(pred_t) == len(points)):
        pred_t = points[0] + pred_t[0,:]
    pred_q = pred_r[0,:,[1,2,3,0]]
    pred_q /= torch.norm(pred_q, dim=1).view(-1,1)

    max_c = to_np(how_max)
    max_q = to_np(pred_q[which_max.item()])
    max_t = to_np(pred_t[which_max.item()])
    
    if(feat_all is not None):
        max_feat = to_np(feat_all[0,:,which_max.item()])
    else:
        max_feat = None
        
    if(refine_model is not None):
        refine_q = max_q.copy()
        refine_t = max_t.copy()
        num_points = points.shape[1]
        for _ in range(0, 2):
            T = torch.Tensor(refine_t).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            mat = quaternion_matrix(refine_q)
            R = torch.Tensor(mat[:3, :3]).cuda().view(1, 3, 3)
            mat[0:3, 3] = refine_t

            new_points = torch.bmm((points - T), R).contiguous()
            pred_dr, pred_dt = refine_model(new_points, emb, df_obj_idx)
            pred_dr = pred_dr.view(1, 1, -1)
            pred_dr = pred_dr / (torch.norm(pred_dr, dim=2).view(1, 1, 1))
            pred_dq = pred_dr[0,0,[1,2,3,0]]

            pred_dq = to_np(pred_dq)
            pred_dt = to_np(pred_dt)
            pred_mat = quaternion_matrix(pred_dq)
            pred_mat[0:3, 3] = pred_dt

            final_mat = np.dot(mat, pred_mat)
            refine_q = quaternion_from_matrix(final_mat)
            refine_t = final_mat[0:3, 3]
    else:
        refine_q = None
        refine_t = None
        
    return max_q, max_t, max_c, max_feat, pred_q, pred_t, pred_c, refine_q, refine_t, feat

