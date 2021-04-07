# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

import numpy as np
from quat_math import projectedAverageQuaternion

def vec_in_list(target, list_vecs):
    return next((True for elem in list_vecs if elem.size == target.size and np.array_equal(elem, target)), False)

def vec_close_in_list(target, list_vecs):
    return next((True for elem in list_vecs if elem.size == target.size and np.allclose(elem, target)), False)

def insideTetra(tetra, p, return_all = False):
    v0 = np.zeros(4)
    v1 = tetra.vertices[0]
    v2 = tetra.vertices[1]
    v3 = tetra.vertices[2]
    v4 = tetra.vertices[3]
    
    D1 = np.linalg.det(np.stack([v2,v3,v4, v1]))
    D2 = np.linalg.det(np.stack([v1,v3,v4, v2]))
    D3 = np.linalg.det(np.stack([v1,v2,v4, v3]))
    D4 = np.linalg.det(np.stack([v1,v2,v3, v4]))

    P1 = np.linalg.det(np.stack([v2,v3,v4, p]))
    P2 = np.linalg.det(np.stack([v1,v3,v4, p]))
    P3 = np.linalg.det(np.stack([v1,v2,v4, p]))
    P4 = np.linalg.det(np.stack([v1,v2,v3, p]))

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
    #print(levels, insideTetra2(tetrahedron, q) or insideTetra2(tetrahedron, -q))
    if(levels == 0):
        dists = dist_func(tetrahedron.vertices)
        idx = np.argmax(dists)
        return tetrahedron.vertices[idx]
        return projectedAverageQuaternion(tetrahedron.vertices)#, weights = 1/np.array(dists))

    tetras = tetrahedron.Subdivide()

    max_idx = -1
    max_val = -float('inf')

    for j, tra in enumerate(tetras):
        dists = dist_func(tra.vertices)
        val = metric_func(dists)
        #print(j, val, insideTetra2(tra, q) or insideTetra2(tra, -q))
        #print(dists)
        if(val > max_val):
            max_val = val
            max_idx = j
    #print(max_idx, max_val)
    return refineTetrahedron(q, tetras[max_idx], dist_func, metric_func, levels=levels-1)


