#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

2022.9.24, update, modify smooth_surface_map to support multiple channel features.

Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import numpy as np
import pyvista
import copy, os
import scipy.io as sio 
import math, multiprocessing


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
    assert len(faces)/4 == n_faces, "faces number is wrong!"
    faces = np.reshape(faces, (n_faces,4))
    
    data = {'vertices': vertices,
            'faces': faces
            }
    
    point_data = polydata.point_data
    for key, value in point_data.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif  value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data
    

def write_vtk(in_dic, file, binary=True):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    surf = pyvista.PolyData(in_dic['vertices'], in_dic['faces'])
    for key, value in in_dic.items():
        if key == 'vertices' or key == 'faces':
            continue
        surf.point_data[key] = value

    surf.save(file, binary=binary)
     
    
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


def multiVertexLabel(vertexs, vertices, tree, label):
    label_inter = np.zeros((vertexs.shape[0], label.shape[1]))
    for i in range(vertexs.shape[0]):
        _, nearest_vertex = tree.query(vertexs[i,:][np.newaxis,:], k=1)
        label_inter[i] = label[nearest_vertex]
    return label_inter

def resample_label(vertices_fix, vertices_inter, label, multiprocess=True):
    """
    
    Resample label using nearest neighbor on sphere
    
    Parameters
    ----------
    vertices_fix : N*3 numpy array,
          original sphere.
    vertices_inter : M*3 numpy array
        the sphere to be interpolated.
    label : [N, ?], numpy array,
         the label on orignal sphere.

    Returns
    -------
    label_inter : TYPE
        DESCRIPTION.

    """
    assert len(vertices_fix) == len(label), "length of label should be "+\
        "consistent with the length of vertices on orginal sphere."
    if len(label.shape) == 1:
        label = label[:,np.newaxis]
    
    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1
    
    tree = KDTree(vertices_fix, leaf_size=10)  # build kdtree
    label_inter = np.zeros((len(vertices_inter), label.shape[1])).astype(np.int32)
    
    """ multiple processes method: 163842:  s, 40962:  s, 10242: s, 2562: s """
    if  multiprocess:
        pool = multiprocessing.Pool()
        cpus = multiprocessing.cpu_count()
        vertexs_num_per_cpu = math.ceil(vertices_inter.shape[0]/cpus)
        results = []
        for i in range(cpus):
            results.append(pool.apply_async(multiVertexLabel, 
                                            args=(vertices_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:],
                                                  vertices_fix, tree, label,)))
        pool.close()
        pool.join()
        for i in range(cpus):
            label_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:] = results[i].get()
            
    else:
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
    
    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    
    if neigh_orders is None:
        neigh_orders = get_neighs_order(abspath+'/neigh_indices/adj_mat_order_'+ str(int(feat.shape[0])) +'_rotated_0.mat')
    assert neigh_orders.shape[0] == vertices.shape[0] * 7, "neighbor_orders is not right"      
        
    smooth_kernel = np.array([1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/2])
    smooth_kernel = np.repeat(smooth_kernel[:, np.newaxis], feat.shape[1], axis=1)
    for i in range(num_iter):
        tmp = np.multiply(feat[neigh_orders].reshape((feat.shape[0], 7, feat.shape[1])), smooth_kernel)
        feat = np.sum(tmp, axis=1)
    
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
