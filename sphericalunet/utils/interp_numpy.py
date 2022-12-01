#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:36:21 2020

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com


Last update on 2022.08.25
...



"""
import numpy as np
import itertools
from sklearn.neighbors import KDTree
from .utils import get_neighs_order
import math, multiprocessing, os

abspath = os.path.abspath(os.path.dirname(__file__))



def get_bi_inter(n_vertex):
    inter_indices_0 = np.load(abspath+'/neigh_indices/img_indices_'+ str(n_vertex) +'_0.npy')
    inter_weights_0 = np.load(abspath+'/neigh_indices/img_weights_'+ str(n_vertex) +'_0.npy')
    
    inter_indices_1 = np.load(abspath+'/neigh_indices/img_indices_'+ str(n_vertex) +'_1.npy')
    inter_weights_1 = np.load(abspath+'/neigh_indices/img_weights_'+ str(n_vertex) +'_1.npy')
    
    inter_indices_2 = np.load(abspath+'/neigh_indices/img_indices_'+ str(n_vertex) +'_2.npy')
    inter_weights_2 = np.load(abspath+'/neigh_indices/img_weights_'+ str(n_vertex) +'_2.npy')
    
    return (inter_indices_0, inter_weights_0), (inter_indices_1, inter_weights_1), (inter_indices_2, inter_weights_2)


def get_latlon_img(bi_inter, feat):
    inter_indices, inter_weights = bi_inter
    width = int(np.sqrt(len(inter_indices)))
    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]
        
    img = np.sum(np.multiply((feat[inter_indices.flatten()]).reshape((inter_indices.shape[0],
                                                                      inter_indices.shape[1], 
                                                                      feat.shape[1])), 
                             np.repeat(inter_weights[:,:, np.newaxis], 
                                       feat.shape[1], axis=-1)), 
                 axis=1)
    
    img = img.reshape((width, width, feat.shape[1]))
    
    return img

 
def isATriangle(neigh_orders, face):
    """
    neigh_orders: int, N x 7
    face: int, 3 x 1
    """
    neighs = neigh_orders[face[0]]
    if face[1] not in neighs or face[2] not in neighs:
        return False
    neighs = neigh_orders[face[1]]
    if face[2] not in neighs:
        return False
    return True


def projectVertex(vertex, v0, v1, v2):
    normal = np.cross(v0 - v2, v1 - v2)
    if np.linalg.norm(normal) == 0:
        normal = v0
    ratio = v0.dot(normal)/vertex.dot(normal)
    return ratio * vertex


def isOnSameSide(P, v0 , v1, v2):
    """
    Check if P and v0 is on the same side
    """
    edge_12 = v2 - v1
    tmp0 = P - v1
    tmp1 = v0 - v1
    
    edge_12 = edge_12 / np.linalg.norm(edge_12)
    tmp0 = tmp0 / np.linalg.norm(tmp0)
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    
    vec1 = np.cross(edge_12, tmp0)
    vec2 = np.cross(edge_12, tmp1)
    
    return vec1.dot(vec2) >= 0


def isInTriangle(vertex, v0, v1, v2):
    """
    Check if the vertices is in the triangle composed by v0 v1 v2
    vertex: N*3, check N vertices at the same time
    v0: (3,)
    v1: (3,)
    v2: (3,)
    """
    # Project point onto the triangle plane
    P = projectVertex(vertex, v0, v1, v2)
          
    return isOnSameSide(P, v0, v1, v2) and isOnSameSide(P, v1, v2, v0) and isOnSameSide(P, v2, v0, v1)




def singleVertexInterpo_ring(vertex, vertices, tree, neigh_orders, ring_iter=1, ring_threshold=3, threshold=1e-8, debug=False):
    
    if ring_iter > ring_threshold:
        print("ring_iter > ring_threshold, use neaerest 3 neighbor")
        _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
        return np.squeeze(top3_near_vertex_index)

    _, top1_near_vertex_index = tree.query(vertex[np.newaxis,:], k=1)
    ring = []
    
    if type(neigh_orders) == list:
        ring.append({np.squeeze(top1_near_vertex_index).tolist()})  # 0-ring index
        ring.append(neigh_orders[list(ring[0])[0]])         # 1-ring neighs
        for i in range(ring_iter-1):
            tmp = set()
            for j in ring[i+1]:
                tmp = set.union(tmp, neigh_orders[j])
            ring.append(tmp-ring[i]-ring[i+1])
        candi_vertex = set.union(ring[-1], ring[-2])
    else:
        ring.append(np.squeeze(top1_near_vertex_index))  # 0-ring index
        ring.append(np.setdiff1d(np.unique(neigh_orders[ring[0]]), ring[0]))    # 1-ring neighs
        for i in range(ring_iter-1):
            tmp = np.setdiff1d(np.unique(neigh_orders[ring[i+1]].flatten()), ring[i+1])
            ring.append(np.setdiff1d(tmp, ring[i]))
        candi_vertex = np.append(ring[-1], ring[-2])

    candi_faces = []
    for t in itertools.combinations(candi_vertex, 3):
        tmp = np.asarray(t)  # get the indices of the potential candidate triangles
        if isATriangle(neigh_orders, tmp):
              candi_faces.append(tmp)
    candi_faces = np.asarray(candi_faces)

    orig_vertex_1 = vertices[candi_faces[:,0]]
    orig_vertex_2 = vertices[candi_faces[:,1]]
    orig_vertex_3 = vertices[candi_faces[:,2]]
    edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
    edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
    faces_normal = np.cross(edge_12, edge_13)    # normals of all the faces
    tmp = (np.linalg.norm(faces_normal, axis=1) == 0).nonzero()[0]
    faces_normal[tmp] = orig_vertex_1[tmp]
    faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:,np.newaxis]

    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
    tmp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
    P = tmp[:, np.newaxis] * vertex  # intersection points

    # find the triangle face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole one
    area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
    area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
    area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
    area_ABC = np.linalg.norm(faces_normal, axis=1)/2.0
    
    tmp = area_BCP + area_ACP + area_ABP - area_ABC
    index = np.argmin(tmp)
    
    if tmp[index] > threshold:
        return singleVertexInterpo_ring(vertex, vertices, tree, neigh_orders, 
                                        ring_iter=ring_iter+1, ring_threshold=ring_threshold, 
                                        threshold=threshold, debug=False)

    return candi_faces[index]
    

# deprecated
# def singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=7, threshold=1e-6, k_threshold=15, debug=False):
    
#     if k > k_threshold:
#         # print("use neaerest neighbor, k=", k)
#         _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
#         top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
#         return top3_near_vertex_index

#     _, top7_near_vertex_index = tree.query(vertex[np.newaxis,:], k=k)
#     candi_faces = []
#     for t in itertools.combinations(np.squeeze(top7_near_vertex_index), 3):
#         tmp = np.asarray(t)  # get the indices of the potential candidate triangles
#         if isATriangle(neigh_orders, tmp):
#              candi_faces.append(tmp)
   
#     if candi_faces:
#         candi_faces = np.asarray(candi_faces)
#     else:
#         if k > k_threshold-5 and debug==True:
#             print("cannot find candidate faces, top k shoulb be larger, function recursion, current k =", k)
#         return singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=k+5, 
#                                      threshold=threshold, k_threshold=k_threshold)

#     orig_vertex_1 = vertices[candi_faces[:,0]]
#     orig_vertex_2 = vertices[candi_faces[:,1]]
#     orig_vertex_3 = vertices[candi_faces[:,2]]
#     edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
#     edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
#     faces_normal = np.cross(edge_12, edge_13)    # normals of all the faces
#     tmp = (np.linalg.norm(faces_normal, axis=1) == 0).nonzero()[0]
#     faces_normal[tmp] = orig_vertex_1[tmp]
#     faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:,np.newaxis]

#     # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
#     tmp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
#     P = tmp[:, np.newaxis] * vertex  # intersection points

#     # find the triangle face that the inersection is in, if the intersection
#     # is in, the area of 3 small triangles is equal to the whole one
#     area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
#     area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABC = np.linalg.norm(faces_normal, axis=1)/2.0
    
#     tmp = area_BCP + area_ACP + area_ABP - area_ABC
#     index = np.argmin(tmp)
    
#     if tmp[index] > threshold:
#         if k > 30 and debug==True:
#             print("candidate faces don't contain the correct one, top k shoulb be larger, function recursion, current k =", k)
#         return singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, k=k+5, 
#                                      threshold=threshold, k_threshold=k_threshold)

#     w = np.array([area_BCP[index], area_ACP[index], area_ABP[index]])
#     if w.sum() == 0:
#         _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
#         top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
#         return top3_near_vertex_index
#     else:
#         return candi_faces[index]
        


def singleVertexInterpo(vertex, vertices, tree, neigh_orders, feat, fast, threshold=1e-8, ring_threshold=3):
    """
    Compute the three indices for sphere interpolation at given position.
    
    """
    _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3) 
    top3_near_vertex_index = np.squeeze(top3_near_vertex_index)
    
    if fast:
        return top3_near_vertex_index
    
    
    if isATriangle(neigh_orders, top3_near_vertex_index):
        v0 = vertices[top3_near_vertex_index[0]]
        v1 = vertices[top3_near_vertex_index[1]]
        v2 = vertices[top3_near_vertex_index[2]]
        
        normal = np.cross(v1-v2, v0-v2)
        vertex_proj = v0.dot(normal)/vertex.dot(normal) * vertex
        area_BCP = np.linalg.norm(np.cross(v2-vertex_proj, v1-vertex_proj))/2.0
        area_ACP = np.linalg.norm(np.cross(v2-vertex_proj, v0-vertex_proj))/2.0
        area_ABP = np.linalg.norm(np.cross(v1-vertex_proj, v0-vertex_proj))/2.0
        area_ABC = np.linalg.norm(normal)/2.0
        
        if area_BCP + area_ACP + area_ABP - area_ABC > threshold:
              # inter_indices = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders, 
              #                                       threshold=threshold, k_threshold=k_threshold)
              inter_indices = singleVertexInterpo_ring(vertex, vertices, tree, 
                                                        neigh_orders, ring_iter=1,
                                                        ring_threshold=ring_threshold, 
                                                        threshold=threshold, debug=False)     
        else:
            inter_indices = top3_near_vertex_index
       
    else:
        # inter_indices = singleVertexInterpo_7(vertex, vertices, tree, neigh_orders,
        #                                       threshold=threshold, k_threshold=k_threshold)
        inter_indices = singleVertexInterpo_ring(vertex, vertices, tree, 
                                                neigh_orders, ring_iter=1,
                                                ring_threshold=ring_threshold, 
                                                threshold=threshold, debug=False)     
    return inter_indices


def multiVertexInterpo(vertexs, vertices, tree, neigh_orders, feat, fast, threshold, ring_threshold):
    inter_indices = np.zeros((vertexs.shape[0], 3), dtype=np.int32)
    
    for i in range(vertexs.shape[0]):
        inter_indices[i, :] = singleVertexInterpo(vertexs[i,:], 
                                                    vertices, 
                                                    tree, 
                                                    neigh_orders, 
                                                    feat, 
                                                    fast,
                                                    threshold,
                                                    ring_threshold)
    return inter_indices

     

def resampleStdSphereSurf(n_curr, n_next, feat, upsample_neighbors):
    assert len(feat) == n_curr, "feat length not cosistent!"
    assert n_next == n_curr*4-6, "This function can only upsample one level higher"+ \
        " If you want to upsample with two levels higher, you need to call this function twice."
    
    feat_inter = np.zeros((n_next, feat.shape[1]))
    feat_inter[0:n_curr, :] = feat
    feat_inter[n_curr:, :] = feat[upsample_neighbors].reshape(n_next-n_curr, 2, feat.shape[1]).mean(1)
    
    return feat_inter


def resampleSphereSurf(vertices_fix, vertices_inter, feat, faces=None, 
                       std=False, upsample_neighbors=None, neigh_orders=None, 
                       fast=False, threshold=1e-6, ring_threshold=3):
    """
    resample sphere surface

    Parameters
    ----------
    vertices_fix :  N*3, numpy array, 
        the original fixed vertices with features.
    vertices_inter : unknown*3, numpy array, 
        points to be interpolated.
    feat :  N*D, 
        features to be interpolated.
    faces :  N*4, numpy array, the first column shoud be all 3
        is the original faces directly read using read_vtk,. The default is None.
    std : bool
        standard sphere interpolation, e.g., interpolate 10242 from 2562.. The default is False.
    upsample_neighbors : TYPE, optional
        DESCRIPTION. The default is None.
    neigh_orders : TYPE, optional
        DESCRIPTION. The default is None.
    fast : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 1e-6.

    Returns
    -------
    resampled feature.
    
    """
    
    
    assert vertices_fix.shape[0] == feat.shape[0], "vertices.shape[0] is not equal to feat.shape[0]"
    assert vertices_fix.shape[1] == 3, "vertices size not right"
    
    vertices_fix = vertices_fix.astype(np.float64)
    vertices_inter = vertices_inter.astype(np.float64)
    feat = feat.astype(np.float64)
    
    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1
    
    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]
        
    if std:
        assert upsample_neighbors is not None, " upsample_neighbors is None"
        return resampleStdSphereSurf(len(vertices_fix), len(vertices_inter), feat, upsample_neighbors)
    
    if not fast:
        if neigh_orders is None:
            if faces is not None:
                assert faces.shape[1] == 4, "faces shape is wrong, should be N*4"
                assert (faces[:,0] == 3).sum() == faces.shape[0], "the first column of faces should be all 3"
                faces = faces[:, 1:]
                
                num_vers = vertices_fix.shape[0]
                neigh_unsorted_orders = []
                for i in range(num_vers):
                    neigh_unsorted_orders.append(set())
                for i in range(faces.shape[0]):
                    face = faces[i]
                    neigh_unsorted_orders[face[0]].add(face[1])
                    neigh_unsorted_orders[face[0]].add(face[2])
                    neigh_unsorted_orders[face[1]].add(face[0])
                    neigh_unsorted_orders[face[1]].add(face[2])
                    neigh_unsorted_orders[face[2]].add(face[0])
                    neigh_unsorted_orders[face[2]].add(face[1])
                
                neigh_orders = neigh_unsorted_orders
                        
                # deprecated, too slow, use above set() method
                # for i in range(faces.shape[0]):
                #     if faces[i,1] not in neigh_orders[faces[i,0]]:
                #         neigh_orders[faces[i,0], np.where(neigh_orders[faces[i,0]] == -1)[0][0]] = faces[i,1]
                #     if faces[i,2] not in neigh_orders[faces[i,0]]:
                #         neigh_orders[faces[i,0], np.where(neigh_orders[faces[i,0]] == -1)[0][0]] = faces[i,2]
                #     if faces[i,0] not in neigh_orders[faces[i,1]]:
                #         neigh_orders[faces[i,1], np.where(neigh_orders[faces[i,1]] == -1)[0][0]] = faces[i,0]
                #     if faces[i,2] not in neigh_orders[faces[i,1]]:
                #         neigh_orders[faces[i,1], np.where(neigh_orders[faces[i,1]] == -1)[0][0]] = faces[i,2]
                #     if faces[i,1] not in neigh_orders[faces[i,2]]:
                #         neigh_orders[faces[i,2], np.where(neigh_orders[faces[i,2]] == -1)[0][0]] = faces[i,1]
                #     if faces[i,0] not in neigh_orders[faces[i,2]]:
                #         neigh_orders[faces[i,2], np.where(neigh_orders[faces[i,2]] == -1)[0][0]] = faces[i,0]
                       
            else:
                neigh_orders = get_neighs_order(abspath+'/neigh_indices/adj_mat_order_'+ str(vertices_fix.shape[0]) +'_rotated_0.mat')
                neigh_orders = neigh_orders.reshape(vertices_fix.shape[0], 7)
        else:
            neigh_orders = neigh_orders.reshape(vertices_fix.shape[0], 7)
        
   
    
    inter_indices = np.zeros((vertices_inter.shape[0], 3), dtype=np.int32)
    tree = KDTree(vertices_fix, leaf_size=10)  # build kdtree
    
    """ Single process, single thread: 163842: 54.5s, 40962: 12.7s, 10242: 3.2s, 2562: 0.8s """
#    for i in range(vertices_inter.shape[0]):
#        print(i)
#        inter_indices[i,:] = singleVertexInterpo(vertices_inter[i,:], vertices_fix, tree, neigh_orders, feat)
       

    """ multiple processes method: 163842:  s, 40962: 2.8s, 10242: 1.0s, 2562: 0.28s """
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    vertexs_num_per_cpu = math.ceil(vertices_inter.shape[0]/cpus)
    results = []
    
    for i in range(cpus):
        results.append(pool.apply_async(multiVertexInterpo, 
                                        args=(vertices_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:],
                                              vertices_fix, tree, neigh_orders, feat, fast, threshold, ring_threshold)))

    pool.close()
    pool.join()

    for i in range(cpus):
        inter_indices[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:] = results[i].get()
        
    v0 = vertices_fix[inter_indices[:, 0]]
    v1 = vertices_fix[inter_indices[:, 1]]    
    v2 = vertices_fix[inter_indices[:, 2]]
    normal = np.cross(v1-v0, v2-v0, axis=1)
    vertex_proj = np.sum(v0*normal, axis=1, keepdims=True) / np.sum(vertices_inter*normal, axis=1, keepdims=True) * vertices_inter
    
    tmp_index = np.argwhere(np.isnan(vertex_proj))[:, 0]  # in case that normal is [0,0,0]
    
    area_12P = np.linalg.norm(np.cross(v2-vertex_proj, v1-vertex_proj, axis=1), axis=1, keepdims=True)/2.0
    area_02P = np.linalg.norm(np.cross(v2-vertex_proj, v0-vertex_proj, axis=1), axis=1, keepdims=True)/2.0
    area_01P = np.linalg.norm(np.cross(v1-vertex_proj, v0-vertex_proj, axis=1), axis=1, keepdims=True)/2.0
    inter_weights = np.concatenate(([area_12P, area_02P, area_01P]), axis=1)
    
    inter_weights[tmp_index] = np.array([1,0,0])    # in case that normal is [0,0,0]
    
    inter_weights = inter_weights / np.sum(inter_weights, axis=1, keepdims=True)

    feat_inter = np.sum(np.multiply(feat[inter_indices], np.repeat(inter_weights[:,:,np.newaxis], feat.shape[1], axis=2)), axis=1)

    return feat_inter

        

def bilinear_interpolate(im, x, y):

    x = np.clip(x, 0.0001, im.shape[1]-1.0001)
    y = np.clip(y, 0.0001, im.shape[1]-1.0001)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa[:,np.newaxis]*Ia + wb[:,np.newaxis]*Ib + wc[:,np.newaxis]*Ic + wd[:,np.newaxis]*Id

        
def bilinearResampleSphereSurf(vertices_inter, feat, bi_inter, radius=1.0):
    """
    ONLY!! assume vertices_fix are on the standard icosahedron discretized spheres!!
    
    """
    img = get_latlon_img(bi_inter, feat)
    
    return bilinearResampleSphereSurfImg(vertices_inter, img, radius=radius)


def bilinearResampleSphereSurfImg(vertices_inter_raw, img, radius=1.0):
    """
    ONLY!! assume vertices_fix are on the standard icosahedron discretized spheres!!
    
    """
    vertices_inter = np.copy(vertices_inter_raw)
    vertices_inter = vertices_inter / radius
    width = img.shape[0]
    
    vertices_inter[:,2] = np.clip(vertices_inter[:,2], -0.999999999, 0.999999999)
    beta = np.arccos(vertices_inter[:,2]/1.0)
    row = beta/(np.pi/(width-1))
    
    tmp = (vertices_inter[:,0] == 0).nonzero()[0]
    vertices_inter[:,0][tmp] = 1e-15
    
    alpha = np.arctan(vertices_inter[:,1]/vertices_inter[:,0])
    tmp = (vertices_inter[:,0] < 0).nonzero()[0]
    alpha[tmp] = np.pi + alpha[tmp]
    
    alpha = 2*np.pi + alpha
    alpha = np.remainder(alpha, 2*np.pi)
    
    col = alpha/(2*np.pi/(width-1))
    
    feat_inter = bilinear_interpolate(img, col, row)
    
    return feat_inter
