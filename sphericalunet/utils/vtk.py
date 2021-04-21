#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import numpy as np
import pyvista
import copy, os
import scipy.io as sio 

from sklearn.neighbors import KDTree

abspath = os.path.abspath(os.path.dirname(__file__))

def read_vtk(in_file):
    """
    Read .vtk POLYDATA file
    
    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'curv', 'sulc', ...
    """

    polydata = pyvista.read(in_file)
 
    n_faces = polydata.n_faces
    vertices = np.array(polydata.points)  # get vertices coordinate
    
    # only for triangles polygons data
    faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
    assert len(faces)/4 == n_faces, "faces number is not consistent!"
    faces = np.reshape(faces, (n_faces,4))
    
    data = {'vertices': vertices,
            'faces': faces
            }
    
    point_arrays = polydata.point_arrays
    for key, value in point_arrays.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif  value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data
    

def write_vtk(in_dic, file):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    assert 'vertices' in in_dic, "output vtk data does not have vertices!"
    assert 'faces' in in_dic, "output vtk data does not have faces!"
    
    data = copy.deepcopy(in_dic)
    
    vertices = data['vertices']
    faces = data['faces']
    surf = pyvista.PolyData(vertices, faces)
    
    del data['vertices']
    del data['faces']
    for key, value in data.items():
        surf.point_arrays[key] = value

    surf.save(file, binary=False)  
    
    
def write_vertices(in_ver, file):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    
    with open(file,'a') as f:
        f.write("# vtk DataFile Version 4.2 \n")
        f.write("vtk output \n")
        f.write("ASCII \n")
        f.write("DATASET POLYDATA \n")
        f.write("POINTS " + str(len(in_ver)) + " float \n")
        np.savetxt(f, in_ver)


def remove_field(data, *fields):
    """
    remove the field attribute in data
    
    fileds: list, strings to remove
    data: dic, vtk dictionary
    """
    for field in fields:
        if field in data.keys():
            del data[field]
    
    return data


def resample_label(vertices_fix, vertices_inter, label):
    """
    Resample label using nearest neighbor on sphere
    
    vertices_fix: N*3 numpy array, original sphere
    vertices_inter: M*3 numpy array, the sphere to be interpolated
    label: [N, ?], numpy array, the label on orignal sphere
    """
    assert len(vertices_fix) == len(label), "length of label should be consistent with the vertices on orginal sphere."
    if len(label.shape) == 1:
        label = label[:,np.newaxis]
    
    tree = KDTree(vertices_fix, leaf_size=10)  # build kdtree
    label_inter = np.zeros((len(vertices_inter), label.shape[1])).astype(np.int32)
    for i in range(len(vertices_inter)):
        _, nearest_vertex = tree.query(vertices_inter[i,:][np.newaxis,:], k=1)
        label_inter[i] = label[nearest_vertex]
          
    return label_inter


def smooth_surface_map(vertices, feat, num_iter, neigh_orders=None):
    """
    smooth surface maps
    
    vertices: N*3 numpy array, surface vertices
    neigh_orders: M*3 numpy array, the sphere to be interpolated
    feat: [N, 1], numpy array, the surface map to be smoothed
    num_iter: numbers of smooth operation
    """
    assert vertices.shape[0] == feat.shape[0], "vertices number is different from feature number"
    assert vertices.shape[0] in [42,162,642,2562,10242,40962,163842], "only support icosahedron discretized spheres"
    
    if neigh_orders is None:
        neigh_orders = get_neighs_order(abspath+'/neigh_indices/adj_mat_order_'+ str(int(feat.shape[0])) +'_rotated_0.mat')
    assert neigh_orders.shape[0] == vertices.shape[0] * 7, "neighbor_orders is not right"      
        
    smooth_kernel = np.array([1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/2])
    for i in range(num_iter):
        feat = np.matmul(feat[neigh_orders].reshape((feat.shape[0], 7)), smooth_kernel)
    
    return feat

   
def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders


def faces_to_neigh_orders(faces):
    
    return 0
    

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
## Make data
#u = np.linspace(0, 2 * np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#x = 100 * np.outer(np.cos(u), np.sin(v))
#y = 100 * np.outer(np.sin(u), np.sin(v))
#z = 100 * np.outer(np.ones(np.size(u)), np.cos(v))
#
## Plot the surface
#ax.plot_wireframe(x, y, z, color='b')
#
#ax.set_xlim3d(-200, 200)
#ax.set_ylim3d(-200, 200)
#ax.set_zlim3d(-200, 200)
##        a = np.array([1,0,0])
##        b = np.array([0,2,0])
##        c = np.array([0,0,3])
#ax.scatter(a[0], a[1], a[2], s=50, c='r', marker='o')
#ax.scatter(b[0], b[1], b[2], s=50, c='r', marker='o')
#ax.scatter(c[0], c[1], c[2], s=50, c='r', marker='o')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#ab = b - a
#ac = c - a
#normal = np.cross(ab,ac)
#x = np.linspace(-200,200,100)
#y = np.linspace(-200,200,100)
#X,Y = np.meshgrid(x,y)
#Z = (np.dot(normal, c) - normal[0]*X - normal[1]*Y)/normal[2]
#ax.plot_wireframe(X, Y, Z)
#
#a_1 = a + (a-b) * 100
#b_1 = b + (b-a) * 100
#c_1 = c + (c-b) * 100
#ax.scatter(a_1[0], a_1[1], a_1[2], s=50, c='r', marker='o')
#ax.scatter(b_1[0], b_1[1], b_1[2], s=50, c='r', marker='o')
#ax.scatter(c_1[0], c_1[1], c_1[2], s=50, c='r', marker='o')
#
#
#plt.show()
        
