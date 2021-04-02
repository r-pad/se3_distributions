# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""
from model_renderer.pose_renderer import BpyRenderer

import numpy as np
import torch
import time

from se3_distributions.utils.tetra_utils import *
from se3_distributions.utils.image_preprocessing import preprocessImages
from se3_distributions.utils import to_var, to_np
#from se3_distributions.bbTrans.discretized4dSphere import S3Grid
from se3_distributions.eval.multiscale_grid import MultiResGrid

import os
root_folder = os.path.dirname(os.path.abspath(__file__))

class ExemplarDistPoseEstimator(object):
    def __init__(self, dist_network,
                 renderer_transform_func,
                 model_filename,
                 base_render_folder = None,
                 img_size = (224,224),
                 base_level = 2,
                 model_scale = 1.0,
                 camera_dist = 0.33):
        self.img_size = img_size
        self.camera_dist = camera_dist
        self.dist_network = dist_network
        self.dist_network.cuda()
        self.dist_network.eval()

        self.renderer = BpyRenderer(transform_func = renderer_transform_func)
        self.renderer.loadModel(model_filename, emit = 0.5)
        self.renderPoses = self.renderer.renderPose
        self.base_level = base_level
        self.base_render_folder = base_render_folder
        self.reset()

    def reset(self):
         
        self.grid = MultiResGrid(self.base_level)
        self.base_vertices = self.grid.vertices
      
        if(os.path.exists(os.path.join(self.base_render_folder, 'renders.pt'))):
            print('Loading Base Level')
            self.renders = torch.load(os.path.join(self.base_render_folder, 'renders.pt'))
            base_vertices = torch.load(os.path.join(self.base_render_folder, 'vertices.pt'))
            assert np.all(base_vertices == base_vertices), 'Saved vertices do not match grid'
        else:
            print('Rendering Base Level')
            self.renders = preprocessImages(self.renderPoses(self.base_vertices, camera_dist = self.camera_dist),
                                                 img_size = self.img_size,
                                                 normalize_tensors = True).float()
            import pathlib
            pathlib.Path(self.base_render_folder).mkdir(parents=True, exist_ok=True)
            torch.save(self.renders, os.path.join(self.base_render_folder, 'renders.pt'))
            torch.save(self.base_vertices, os.path.join(self.base_render_folder, 'vertices.pt'))
        
        self.base_size = self.base_vertices.shape[0]
        if(self.base_size > 1500):
            self.features = []
            for j in range(self.base_size//1500):
                j_srt = j*1500
                j_end = (j+1)*1500
                self.features.append(to_np(
                    self.dist_network.originFeatures(to_var(self.renders[j_srt:j_end])).detach()))
            self.features.append(to_np(
                self.dist_network.originFeatures(to_var(self.renders[j_end:])).detach()))
            self.features = to_var(torch.from_numpy(np.vstack(self.features)), 
                                        requires_grad = False)
            torch.cuda.empty_cache()
        else:
            self.features = self.dist_network.originFeatures(self.renders)
        
        self.tetra_dists = []
        self.query_features = None
        self.vert_dists = []

    def estimate(self, img, preprocess=True):
        if(preprocess):
            img = preprocessImages([img],
                                   img_size = self.img_size,
                                   normalize_tensors = True)
        num_imgs = img.shape[0]
        query_features = self.dist_network.queryFeatures(to_var(img.float(), requires_grad=False)).repeat(self.base_size,1)
        dists = self.dist_network.compare_network(self.features,
                                                  query_features)
        return dists.flatten() 
            
    def baseDistance(self, img, preprocess=True):
        self.vert_dists = []
        if(preprocess):
            img = preprocessImages([img],
                                   img_size = self.img_size,
                                   normalize_tensors = True)
        num_imgs = img.shape[0]
        #img = to_var(img.repeat(60,1,1,1).float())
        #dists = to_np(self.dist_network(self.renders, img))
        self.query_features = self.dist_network.queryFeatures(to_var(img.float(), requires_grad=False))
        self.updateDistance()
        return self.vert_dists

    def updateFeatures(self):
        idx_new = len(self.vert_dists)
        new_renders = preprocessImages(self.renderPoses(self.grid.vertices[idx_new:], 
                                                        camera_dist = self.camera_dist),
                                       img_size = self.img_size,
                                       normalize_tensors = True).float()
        self.features = torch.cat([self.features, 
                                   self.dist_network.originFeatures(to_var(new_renders))])
        self.renders = torch.cat([self.renders, new_renders])

    def updateDistance(self):
        idx_new = len(self.vert_dists)
        size_new = len(self.grid.vertices)

        new_dists = to_np(self.dist_network.compare_network(self.features[idx_new:],
                              self.query_features.repeat(size_new-idx_new,1)))
        
        if(idx_new > 0): 
            self.vert_dists = np.hstack([self.vert_dists, new_dists.flatten()])
            self.vert_mask = np.hstack([self.vert_mask, np.zeros_like(new_dists.flatten(), dtype=bool)]) 
        else:
            self.vert_dists = new_dists.flatten()
            self.vert_mask = np.zeros_like(new_dists.flatten(), dtype=bool) 

        #new_dists = self.dist_network.compare_network(self.features[idx_new:],
        #                                              self.query_features.repeat(size_new-idx_new,1))
        #
        #if(idx_new > 0): 
        #    self.vert_dists = torch.cat([self.vert_dists, new_dists.detach().flatten()])
        #else:
        #    self.vert_dists = new_dists.detach().flatten()

    def refine(self, num_indices = 1, expansion_range = 0):
        max_neighborhood = np.ma.masked_array(self.vert_dists, 
            self.vert_mask).argsort(fill_value=self.vert_dists.min())[-num_indices:]
        print('Expanding neighborhood around {}'.format(max_neighborhood))
        for j in range(expansion_range + 1):
            verts = [] 
            tetra = []
            for v in max_neighborhood:
                vs, ts = self.grid.GetNeighborhood(v);
                verts.extend(vs.tolist())
                tetra.extend(ts.tolist())
            max_neighborhood = np.unique(verts)
        for t_idx in tetra:
            self.grid.SubdivideTetrahedra(t_idx)
        self.updateFeatures()
        self.updateDistance()
        self.vert_mask[max_neighborhood] = True

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

    def refineTetrahedron(self, query_features, tetrahedron):
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

