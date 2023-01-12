#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 00:32:35 2022

@author: Fenqiang Zhao
For any bugs, please contact: zhaofenqiang0221@gmail.com
"""

import numpy as np
from .utils import get_neighs_order
from .vtk import read_vtk
from .spherical_mapping import computeFaceWiseArea
import os

abspath = os.path.abspath(os.path.dirname(__file__))


def computeVertexArea(vertices, neigh_sorted_orders=None, NUM_NEIGHBORS=6, faces=None):
    """
    assume all vertices are on the standard icosahedron discretized spheres

    Parameters
    ----------
    vertices : TYPE
        DESCRIPTION.
    neigh_sorted_orders : TYPE
        DESCRIPTION.
    NUM_NEIGHBORS : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    n_vertex = vertices.shape[0]
    
    if neigh_sorted_orders is None and faces is None:
        neigh_orders = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_0.mat')
        neigh_sorted_orders = neigh_orders.reshape((n_vertex, 7))
        neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_sorted_orders[:, 0:6]), axis=1)

    elif neigh_sorted_orders is None and faces is not None:
        assert faces.shape[1] == 3
        num_faces = faces.shape[0]
        vertex_has_faces = []
        for j in range(n_vertex):
            vertex_has_faces.append([])
        for j in range(num_faces):
            face = faces[j]
            vertex_has_faces[face[0]].append(j)
            vertex_has_faces[face[1]].append(j)
            vertex_has_faces[face[2]].append(j)
            
        face_wise_area = computeFaceWiseArea(vertices, faces)
        vert_wise_area = np.zeros(n_vertex)
        for j in range(n_vertex):
            vert_wise_area[j] = face_wise_area[vertex_has_faces[j]].sum()
        vert_wise_area = vert_wise_area/3.0
        return vert_wise_area
    else:
         pass

    # triangle area, NxNUM_NEIGHBORS
    area = np.zeros((n_vertex, NUM_NEIGHBORS))
    for i in range(NUM_NEIGHBORS):
        if i < NUM_NEIGHBORS-1:
            a = vertices[neigh_sorted_orders][:, i+1, :]
            b = vertices[neigh_sorted_orders][:, i+2, :]
        else:
            a = vertices[neigh_sorted_orders][:, -1, :]
            b = vertices[neigh_sorted_orders][:, 1, :]
        c = vertices
        cros_vec = np.cross(a-c, b-c)
        area[:, i] = 1/2 * np.linalg.norm(cros_vec, axis=1)
    return (area.sum(axis=1))/3.0

