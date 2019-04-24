# -*- coding: utf-8 -*-
"""
Created on Some night, Way to late

@author: bokorn
"""
import os
import torch
import numpy as np
from sklearn.neighbors import KDTree
import scipy
import scipy.io as sio
from functools import partial

from generic_pose.utils.pose_processing import getGaussianKernal
from generic_pose.eval.multiscale_grid import MultiResGrid

eps = 1e-12

root_folder = os.path.dirname(os.path.abspath(__file__))

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
    qs1 = q.dot(n1)
    vs1 = v1.T.dot(n1)
    qs2 = q.dot(n2) 
    vs2 = v2.T.dot(n2) 
    qs3 = q.dot(n3) 
    vs3 = v3.T.dot(n3)
    qs4 = q.dot(n4)
    vs4 = v4.T.dot(n4)
    
    if (vs1*qs1 >= -eps and vs2*qs2 >= -eps and vs3*qs3 >= -eps and vs4*qs4 >= -eps) or \
       (vs1*qs1 <=  eps and vs2*qs2 <=  eps and vs3*qs3 <=  eps and vs4*qs4 <=  eps):
        return True
    #if (np.sign(q.dot(n1)) == np.sign(v1.T.dot(n1)) and \
    #    np.sign(q.dot(n2)) == np.sign(v2.T.dot(n2)) and \
    #    np.sign(q.dot(n3)) == np.sign(v3.T.dot(n3)) and \
    #    np.sign(q.dot(n4)) == np.sign(v4.T.dot(n4))) or \
    #   (np.sign(-q.dot(n1)) == np.sign(v1.T.dot(n1)) and \
    #    np.sign(-q.dot(n2)) == np.sign(v2.T.dot(n2)) and \
    #    np.sign(-q.dot(n3)) == np.sign(v3.T.dot(n3)) and \
    #    np.sign(-q.dot(n4)) == np.sign(v4.T.dot(n4))):
    #    return True
    return False

def quatL2Dist(q1, q2):
    return min(np.linalg.norm(q1-q2),np.linalg.norm(q1+q2))

class TetraInterpolation(object):
    def __init__(self, grid_level, values):
        #self.vertices = vertices
        eta = np.abs(values).sum()*(np.pi**2)/values.shape[0]
        self.values = 1./eta * values
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
            _, t_ids = self.grid.GetNeighborhood(v_id, level=self.level)
            tetra_ids.update(t_ids)

        for t_id in tetra_ids:
            if(insideTetra(self.grid.GetTetrahedron(t_id, level=self.level), q)):
                return t_id
        
        raise ValueError('Did not find containing tetrahedra for {}'.format(q)) 
        return None

    def baryTetraTransform(self, t_id):
        return np.linalg.inv(np.vstack(self.grid.GetTetrahedron(t_id, self.level).vertices).T)

    def __call__(self, q):
        t_id = self.containingTetra(q)
        A = self.baryTetraTransform(t_id)
        v_ids = self.grid.GetTetras(self.level)[t_id]

        q_bar = np.matmul(A, q)
        q_bar /= q_bar.sum()
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

bingham_normalization_data = sio.loadmat(os.path.join(root_folder, 'bingham_normalization.mat'))
bingham_normalization_eta = bingham_normalization_data['eta'][0]
bingham_normalization_sigma = bingham_normalization_data['sigma'][0]
binghamNormC = partial(np.interp, 
                       xp=bingham_normalization_sigma, 
                       fp=bingham_normalization_eta)

def proj(u, v):
    return v.dot(u)/u.dot(u)*u

def gramSchmidt(v):
    n = v.size   
    B = np.zeros([n,n])
    for j in range(n):
        u = v
        for k in range(j):
            u = u - proj(B[:,k], v)
        B[:,j] = u/np.linalg.norm(u)
        v = np.random.randn(n)
    return B

def makeBingham(q_center, sigma):
    M = gramSchmidt(q_center)
    Z = np.diag([0,-sigma, -sigma, -sigma])
    eta = binghamNormC(sigma)
    def p(x):
        return 1./eta * np.exp((x.dot(M).dot(Z).dot(M.T)*x).sum(1))
    return p

class BinghamInterpolation(object):
    def __init__(self, vertices, values, sigma=np.pi/9):
        self.vertices = vertices
        self.values = values/values.sum()
        Ms = []
        for v in self.vertices:
            Ms.append(torch.Tensor(gramSchmidt(v)))
        M = torch.stack(Ms)
        Z = torch.diag(torch.Tensor([0,-sigma, -sigma, -sigma]))
        self.MZMt = torch.bmm(torch.bmm(M, Z.repeat([len(Ms),1,1])), torch.transpose(M,2,1))
        self.eta = binghamNormC(sigma)

    def __call__(self, q):
        bingham_p = 1./self.eta*torch.exp(torch.mul(q.transpose(1,0).unsqueeze(2), 
            torch.matmul(q,self.MZMt.transpose(2,0))).sum([0]))
        return (self.values * bingham_p).sum(1)

