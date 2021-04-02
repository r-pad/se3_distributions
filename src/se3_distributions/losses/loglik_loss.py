import torch
import numpy as np

from object_pose_utils.utils import to_np, to_var
from object_pose_utils.utils.pose_processing import tensorAngularDiff

def logLikelihood(preds, dists, p = 1, 
                  k = None, r = None, eps = 1e-9):

    mask = torch.ones_like(dists)

    if(r is not None):
        radius_mask = (dists <= r).type_as(dists)
        mask *= radius_mask
    
    if(k is not None):
        knn_mask = torch.zeros_like(dists)
        knn_mask.scatter_(-1, torch.argsort(dists, dim = -1)[:,:k], 1)
        mask *= knn_mask

    inv_dists = mask * 1./(torch.pow(dists, p) + eps)

    grid_size = dists.shape[1]

    log_lik = torch.log(torch.sum(preds*inv_dists, dim=-1)) \
        - torch.log(torch.sum(inv_dists, dim=-1)) \
        + np.log(grid_size/np.pi**2) \
        - torch.log(torch.sum(preds, dim=-1))

    return log_lik

def evaluateFeature(model, objs, features,
                    grid_features = None,
                    grid_size = 3885):    

    num_features = features.shape[0]
    feature_size = features.shape[1]

    rep_indices = np.repeat(np.arange(num_features), grid_size)
    if(grid_features is not None):
        g_features = []
        for idx in objs:
            g_features.append(grid_features[idx.item()])
        g_features = torch.stack(g_features)

        lik_est = model(to_var(g_features.view(-1, feature_size)).cuda(),
                        to_var(features).cuda()[rep_indices])
        lik_est = lik_est.view(num_features, grid_size, -1).transpose(1,2)
    else:
        lik_est = model(to_var(features).cuda())
        lik_est = lik_est.view(num_features, -1, grid_size)
        
    if(lik_est.shape[1] > 1):
        obj_idxs = (objs.cuda()-1).view(-1,1,1).repeat(1, 1, grid_size)
        lik_est = lik_est.gather(1, obj_idxs)[:,0,:]
    else:
        lik_est = lik_est.view(num_features, grid_size)
    
    return lik_est

def evaluateLikelihood(model, objs, 
                       quats, 
                       features, 
                       grid_vertices,
                       grid_features = None,
                       optimizer = None,
                       calc_metrics = True,
                       retain_graph = False,
                       p = 1,
                       k = None,
                       r = None,
                       ):
    
    g_vertices = []
    for idx in objs:
        g_vertices.append(grid_vertices[idx.item()])
    g_vertices = torch.stack(g_vertices)

    grid_size = g_vertices.shape[1]
    num_features = features.shape[0]
    feature_size = features.shape[1]
    
    rep_indices = np.repeat(np.arange(num_features), grid_size)
    dist_true = tensorAngularDiff(to_var(g_vertices.view(-1, 4)).cuda(),
                                  to_var(quats).cuda()[rep_indices]).view(-1,grid_size)
    if(grid_features is not None):
        g_features = []
        for idx in objs:
            g_features.append(grid_features[idx.item()])
        g_features = torch.stack(g_features)

        lik_est = model(to_var(g_features.view(-1, feature_size)).cuda(),
                        to_var(features).cuda()[rep_indices])
        lik_est = lik_est.view(num_features, grid_size, -1).transpose(1,2)
    else:
        lik_est = model(to_var(features).cuda())
        lik_est = lik_est.view(num_features, -1, grid_size)

    if(lik_est.shape[1] > 1):
        obj_idxs = (objs.cuda()-1).view(-1,1,1).repeat(1, 1, grid_size)
        lik_est = lik_est.gather(1, obj_idxs)[:,0,:]
    else:
        lik_est = lik_est.view(num_features, grid_size)


    loss = -logLikelihood(lik_est, dist_true, p = p, k = k, r = r).mean()

    if(not torch.isnan(loss) and optimizer is not None):
        model.train()
        loss.backward(retain_graph=retain_graph)

        for p in model.parameters():
            if(torch.any(torch.isnan(p.grad.data))):
                p.grad.data = 0.

        optimizer.step()

    metrics = {}
    metrics['loss'] = float(to_np(loss))
    
    if(calc_metrics):
        for obj_id in torch.unique(objs):
            obj_idxs = (objs == obj_id).nonzero()[:,0]
            lik_est_obj = lik_est[obj_idxs]
            dist_true_obj = dist_true[obj_idxs]
            top_idx = torch.argmax(lik_est_obj, dim=1).detach()
            metrics['{}_top_idx_vec'.format(obj_id)] = to_np(top_idx)
            true_idx = torch.argmin(dist_true_obj, dim=1).detach()
            metrics['{}_true_idx_vec'.format(obj_id)] = to_np(true_idx)

            metrics['{}_rank_gt'.format(obj_id)] = to_np((torch.sort(lik_est_obj, descending=True, dim=1)[1] \
                    == true_idx.unsqueeze(1)).nonzero()[:,1]).mean()

            metrics['{}_rank_top'.format(obj_id)] = to_np((torch.sort(dist_true_obj, dim=1)[1] \
                    == top_idx.unsqueeze(1)).nonzero()[:,1]).mean()
             
            dist_shape = lik_est_obj.shape
            true_idx_flat = np.ravel_multi_index(np.stack([np.arange(dist_shape[0]), to_np(true_idx)]), dist_shape)
            top_idx_flat = np.ravel_multi_index(np.stack([np.arange(dist_shape[0]), to_np(top_idx)]), dist_shape)

            metrics['{}_output_gt'.format(obj_id)] = to_np(lik_est_obj.view(-1)[true_idx_flat]).mean()
            metrics['{}_dist_gt'.format(obj_id)] = to_np(dist_true_obj.view(-1)[true_idx_flat]).mean()*180./np.pi
            metrics['{}_dist_top'.format(obj_id)] = to_np(dist_true_obj.view(-1)[top_idx_flat]).mean()*180./np.pi

    return metrics  

