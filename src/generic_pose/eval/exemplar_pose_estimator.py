# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
import torch

from generic_pose.utils.image_preprocessing import preprocessImages
from generic_pose.utils import to_var, to_np
from generic_pose.bbTrans.discretized4dSphere import S3Grid
from quat_math import projectedAverageQuaternion

import os
root_folder = os.path.dirname(os.path.abspath(__file__))
#vert600 = np.load(os.path.join(root_folder, 'ordered_600_cell.npy'))

def insideTetra1(tetra, p, return_all = False):
    v0 = np.zeros(4)
    v1 = tetra.vertices[0]
    v2 = tetra.vertices[1]
    v3 = tetra.vertices[2]
    v4 = tetra.vertices[3]
    
    #D0 = np.stack([v0,v1,v2,v3,v4])
    D1 = np.linalg.det(np.hstack([np.stack([v0,v1,v2,v3, p]), np.ones([5,1])]))
    D2 = np.linalg.det(np.hstack([np.stack([v0,v1,v2, p,v4]), np.ones([5,1])]))
    D3 = np.linalg.det(np.hstack([np.stack([v0,v1, p,v3,v4]), np.ones([5,1])]))
    D4 = np.linalg.det(np.hstack([np.stack([v0, p,v2,v3,v4]), np.ones([5,1])]))

    signs = np.array([np.sign(D1),
                      np.sign(D2),
                      np.sign(D3),
                      np.sign(D4)])
    inside = np.all(signs == signs[0])
    if(return_all):
        return inside, signs
    else:
        return inside

def insideTetra2(tetra, p, return_all = False):
    # From this post
    # http://steve.hollasch.net/cgindex/geometry/ptintet.html
    v0 = np.zeros(4)
    v1 = tetra.vertices[0]
    v2 = tetra.vertices[1]
    v3 = tetra.vertices[2]
    v4 = tetra.vertices[3]
    
    D1 = np.linalg.det(np.hstack([np.stack([v0,v2,v3,v4, v1]), np.ones([5,1])]))
    D2 = np.linalg.det(np.hstack([np.stack([v0,v1,v3,v4, v2]), np.ones([5,1])]))
    D3 = np.linalg.det(np.hstack([np.stack([v0,v1,v2,v4, v3]), np.ones([5,1])]))
    D4 = np.linalg.det(np.hstack([np.stack([v0,v1,v2,v3, v4]), np.ones([5,1])]))
    
    P1 = np.linalg.det(np.hstack([np.stack([v0,v2,v3,v4, p]), np.ones([5,1])]))
    P2 = np.linalg.det(np.hstack([np.stack([v0,v1,v3,v4, p]), np.ones([5,1])]))
    P3 = np.linalg.det(np.hstack([np.stack([v0,v1,v2,v4, p]), np.ones([5,1])]))
    P4 = np.linalg.det(np.hstack([np.stack([v0,v1,v2,v3, p]), np.ones([5,1])]))

    D_signs = np.array([np.sign(D1),
                        np.sign(D2),
                        np.sign(D3),
                        np.sign(D4)])

    P_signs = np.array([np.sign(P1),
                        np.sign(P2),
                        np.sign(P3),
                        np.sign(P4)])

    inside = np.all(D_signs == P_signs)
    if(return_all):
        return inside, D_signs, P_signs
    else:
        return inside


def metricGreaterThan(sorted_vals, max_metrics, metric_func):
    for k in range(len(sorted_vals)):
        v = metric_func(sorted_vals[k:])
        if(v > max_metrics[k]):
            return True
        elif(v < max_metrics[k]):
            return False
    return False


def topTetrahedron(dists, tetras, metric_func=np.mean):
    max_vert_idx = np.argwhere(dists == np.amax(dists)).flatten()
    tetra_mask = np.bitwise_or.reduce(np.isin(tetras, max_vert_idx), axis=1)
    tetra_idxs = np.nonzero(tetra_mask)[0]
    max_tetra_idx = -1
    max_val = -float('inf')
    max_metrics = [-float('inf') for _ in range(4)]
    for j, indices in zip(tetra_idxs, tetras[tetra_mask]):
        vals = []
        for idx in indices:
            vals.append(dists[idx])
    #for j in tetra_idxs:
    #    vals = []
    #    for idx in tetras[j]:
    #        vals.append(dists[idx])
        vals = sorted(vals, reverse=True)
    #    if(np.average(vals) > max_val):
    #        max_val = np.average(vals)
        if(metricGreaterThan(vals, max_metrics, metric_func)):
            max_metrics = [metric_func(vals[k:]) for k in range(4)]
            max_tetra_idx = j
            
    return max_tetra_idx

def refineTetrahedron(q, tetrahedron, dist_func, metric_func, levels=2):
    print(levels, insideTetra2(tetrahedron, q) or insideTetra2(tetrahedron, -q))
    if(levels == 0):
        #dists = dist_func(tetrahedron.vertices)
        return projectedAverageQuaternion(tetrahedron.vertices)#, weights = 1/np.array(dists))

    tetras = tetrahedron.Subdivide()

    max_idx = -1
    max_val = -float('inf')

    for j, tra in enumerate(tetras):
        dists = dist_func(tra.vertices)
        val = metric_func(dists)
        print(j, val, insideTetra2(tra, q) or insideTetra2(tra, -q))
        print(dists)
        if(val > max_val):
            max_val = val
            max_idx = j
    print(max_idx, max_val)
    return refineTetrahedron(q, tetras[max_idx], dist_func, metric_func, levels=levels-1)

class ExemplarDistPoseEstimator(object):
    def __init__(self, model_filename, dist_network,
                 img_size = (224,224),
                 use_bpy_renderer=False,
                 base_level = 0,
                 model_scale = 1.0,
                 camera_dist = 0.33):
                 #inverse_dist = True):
        #if(inverse_dist):
        #    self.inverse_dist = -1
        #else:
        #    self.inverse_dist = 1

        self.img_size = img_size
        self.camera_dist = camera_dist
        self.dist_network = dist_network
        self.dist_network.cuda()
        self.dist_network.eval()
        
        self.base_vertices = np.unique(self.grid.vertices, axis = 0)
        self.base_size = self.base_vertices.shape[0]
        self.base_renders = to_var(preprocessImages(self.renderPoses(self.base_vertices), 
                                                    img_size = self.img_size,
                                                    normalize_tensors = True).float(), 
                                   requires_grad = True)
        
        
        self.grid = S3Grid(base_level)
        self.grid.Simplify()
        self.renderer = BpyRenderer(transform_func = ycbRenderTransform)
        self.renderer.loadModel(self.train_dataset.getModelFilename(),
                                emit = 0.5)
        self.renderPoses = self.renderer.renderPose
        base_render_folder = os.path.join(benchmark_folder,
                                          'base_renders',
                                          self.train_dataset.getObjectName(),
                                          '{}'.format(base_level))
        
        self.base_vertices, self.base_vert_indices = np.unique(self.grid.vertices, 
                                                               return_inverse=True,
                                                               axis = 0)
        if(os.path.exists(os.path.join(base_render_folder, 'renders.pt'))):
            self.base_renders = torch.load(os.path.join(base_render_folder, 'renders.pt'))
        else:
            self.base_renders = preprocessImages(self.renderPoses(self.base_vertices, camera_dist = camera_dist),
                                                 img_size = self.img_size,
                                                 normalize_tensors = True).float()
            import pathlib
            pathlib.Path(base_render_folder).mkdir(parents=True, exist_ok=True)
            torch.save(self.base_renders, os.path.join(base_render_folder, 'renders.pt'))
            torch.save(self.base_vertices, os.path.join(base_render_folder, 'vertices.pt'))
        
        self.base_size = self.base_vertices.shape[0]

        if(self.base_size > 1500):
            self.base_features = []
            for j in range(self.base_size//1500):
                j_srt = j*1500
                j_end = (j+1)*1500
                self.base_features.append(to_np(
                    self.dist_network.originFeatures(self.base_renders[j_srt:j_end]).detach()))
            self.base_features.append(to_np(
                self.dist_network.originFeatures(self.base_renders[j_end:]).detach()))
            self.base_features = to_var(torch.from_numpy(np.vstack(self.base_features)), 
                                        requires_grad = False)
            torch.cuda.empty_cache()
        else:
            self.base_features = self.dist_network.originFeatures(self.base_renders)

    def baseDistance(self, img, preprocess=True):
        if(preprocess):
            img = preprocessImages([img],
                                   img_size = self.img_size,
                                   normalize_tensors = True)
        num_imgs = img.shape[0]
        #img = to_var(img.repeat(60,1,1,1).float())
        #dists = to_np(self.dist_network(self.base_renders, img))
        query_features = self.dist_network.queryFeatures(to_var(img.float(), requires_grad=False))
        dists = self.dist_network.compare_network(self.base_features,
                                                  query_features.repeat(self.base_size,1))
        return dists.flatten()
            
    def getMaxTetraIndex(self, dists, metric='max'):
        max_idx = torch.argmax(dists)
        max_base_indices = np.nonzero(self.base_vert_indices==max_idx)[0]
        tetras = []
        
        max_tetra_idx = -1
        max_val = -float('inf')
        if(metric == 'max'):
            metric_func = np.max
        elif(metric == 'avg'):
            metric_func = np.mean
        for idx in min_base_indices:
            neighborhood, tetra_idxs = self.grid.GetNeighborhood(idx)
            for tetra, t_idx in zip(neighborhood, tetra_idxs):
                dist_vals = []
                for v_idx in tetra:
                    if(v_idx != idx):
                        dist_vals.append(dists[v_idx])
                val = metric_func(dist_vals)
                if(m_val > max_val):
                    max_val = val
                    max_tetra_idx = t_idx
            
        #max_tetrahedron = self.grid.GetTetrahedron(max_tetra)
        #return max_tetrahedron
        return max_tetra_idx


   # def topTetrahedron(self, dists, metric='max'):
   #     return self.grid.GetTetrahedron(topTetraIndex(dists, metric))

    def refine(self, query_features, tetrahedron):
        #tetrahedron = self.grid.GetTetrahedron(index)
        sub_tetrahedra = tetrahedron.Subdivide()
        tetra_verts = []
        for tetra in sub_tetrahedra:
            tetra_verts.extend(tetra.vertices)
        verts, indices = np.unique(tetra_verts, return_inverse=True, axis=0)
        verts_size = verts.shape[0]
        renders = preprocessImages(self.renderPoses(verts, camera_dist = self.camera_dist),
                                                    img_size = self.img_size,
                                                    normalize_tensors = True).float()
        refine_features = dist_network.originFeatures(renders)
        query_features.repeat(self.base_size,1)
        dists = self.dist_network.compare_network(refine_features,
                                                  query_features.repeat(verts_size,1)).flatten()
        dist_expand = dists[indices]
        max_idx = torch.argmax(dists)

        max_base_indices = np.nonzero(indices==max_idx)[0]

