# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:35:53 2017

@author: bokorn
"""

from generic_pose.bbTrans.discretized4dSphere import S3Grid
from generic_pose.utils.tetra_utils import vec_in_list, vec_close_in_list

import numpy as np

class MultiResGrid(S3Grid):
    def __init__(self, depth):
        super().__init__(depth)
        #self.tetra = self.tetra[self.tetra_levels[-2]:]
        #self.tetra_levels = [0, len(self.tetra)]
        #self.depth = 0
        self.Simplify()

    def SubdivideTetrahedra(self, idx):
        tetra = self.GetTetras(self.depth)[idx]
        hedron = self.GetTetrahedron(idx)
        sub_hedrons = hedron.Subdivide()
        vert_size = self.vertices.shape[0]
        new_verts = []
        new_tetra = []
        for sh in sub_hedrons:
            indices = []
            for v in sh.vertices:
                t_idx = np.where((hedron.vertices == v).all(axis=1))[0]
                if(len(t_idx) > 0):
                    indices.append(tetra[int(t_idx)])
                else:
                    if(len(new_verts) > 0):
                        nv_idx = np.where((new_verts == v).all(axis=1))[0]
                    else:
                        nv_idx = []
                    if(len(nv_idx) > 0):
                        indices.append(vert_size + int(nv_idx))
                    else:
                        new_verts.append(v)
                        indices.append(vert_size + len(new_verts))
            new_tetra.append(np.array(indices).copy())
        self.vertices = np.concatenate([self.vertices, new_verts])
        self.tetra = np.concatenate([np.delete(self.tetra, idx+self.tetra_levels[-2], axis = 0), new_tetra])
        self.tetra_levels[-1] += 7
        return new_verts, new_tetra


