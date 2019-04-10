# -*- coding: utf-8 -*-
"""
Created on Some night, Way to late

@author: bokorn
"""
import torch
import numpy as np
from sklearn.neighbors import KDTree
import scipy

from generic_pose.utils.pose_processing import getGaussianKernal
from generic_pose.eval.multiscale_grid import MultiResGrid

def insideTetra(tetra, q):
    v0 = np.zeros(4)
    v1 = tetra.vertices[0]
    v2 = tetra.vertices[1]
    v3 = tetra.vertices[2]
    v4 = tetra.vertices[3]
    
    n1 = scipy.linalg.null_space(np.vstack([v2,v3,v4]))
    n2 = scipy.linalg.null_space(np.vstack([v1,v3,v4]))
    n3 = scipy.linalg.null_space(np.vstack([v1,v2,v4]))
    n4 = scipy.linalg.null_space(np.vstack([v1,v2,v3]))

    if (np.sign(q.dot(n1)) == np.sign(v1.T.dot(n1)) and \
        np.sign(q.dot(n2)) == np.sign(v2.T.dot(n2)) and \
        np.sign(q.dot(n3)) == np.sign(v3.T.dot(n3)) and \
        np.sign(q.dot(n4)) == np.sign(v4.T.dot(n4))) or \
       (np.sign(-q.dot(n1)) == np.sign(v1.T.dot(n1)) and \
        np.sign(-q.dot(n2)) == np.sign(v2.T.dot(n2)) and \
        np.sign(-q.dot(n3)) == np.sign(v3.T.dot(n3)) and \
        np.sign(-q.dot(n4)) == np.sign(v4.T.dot(n4))):
        return True
    
    return False

def quatL2Dist(q1, q2):
    return min(np.linalg.norm(q1-q2),np.linalg.norm(q1+q2))

class TetraInterpolation(object):
    def __init__(self, grid_level, values):
        #self.vertices = vertices
        self.values = values
        self.level = grid_level
        self.grid = MultiResGrid(self.level) 
        self.num_verts = self.grid.vertices.shape[0]

        self.kd_tree = KDTree(np.vstack([self.grid.vertices, -self.grid.vertices]))
        
        self.max_edge = -np.inf
        for j, tet in enumerate(self.grid.GetTetrahedra(self.level)):
            v1 = tet.vertices[0]
            v2 = tet.vertices[1]
            v3 = tet.vertices[2]
            v4 = tet.vertices[3]
            d12 = quatL2Dist(v1,v2)
            d13 = quatL2Dist(v1,v3)
            d14 = quatL2Dist(v1,v4)
            d23 = quatL2Dist(v2,v3)
            d24 = quatL2Dist(v2,v4)
            d34 = quatL2Dist(v3,v4)
            self.max_edge = max(self.max_edge, d12, d13, d14, d23, d24, d34)

    def containingTetra(self, q):
        vert_ids = self.kd_tree.query_radius([q], r = self.max_edge)[0]
        vert_ids = np.where(vert_ids < self.num_verts, vert_ids, vert_ids - self.num_verts)
        tetra_ids = set()

        for v_id in vert_ids:
            _, t_ids = self.grid.GetNeighborhood(v_id, level=2)
            tetra_ids.update(t_ids)

        for t_id in tetra_ids:
            if(insideTetra(self.grid.GetTetrahedron(t_id, level=2), q)):
                return t_id
        
        return None

    def baryTetraTransform(self, t_id):
        return np.linalg.inv(np.vstack(self.grid.GetTetrahedron(t_id, self.level).vertices).T)

    def __call__(self, q):
        t_id = self.containingTetra(q)
        A = self.baryTetraTransform(t_id)
        v_ids = self.grid.GetTetras(self.level)[t_id]

        q_bar = np.matmul(A, q)
        val_verts = self.values[v_ids]
        return q_bar.dot(val_verts)

class GaussianInterpolation(object):
    def __init__(self, vertices, values, sigma=np.pi/9):
        self.vertices = vertices
        self.values = values
        self.sigma = sigma

    def __call__(self, q):
        K = getGaussianKernal(q, self.vertices, sigma=self.sigma)
        return torch.mm(self.values.t(), K)
        

