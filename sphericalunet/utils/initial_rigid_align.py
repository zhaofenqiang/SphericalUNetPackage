#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import numpy as np

from interp_numpy import resampleSphereSurf, bilinearResampleSphereSurfImg
# from utils import get_neighs_order


def get_rot_mat_zyz(z1, y2, z3):
    """
    first z3, then y2, lastly z1
    """
    return np.array([[np.cos(z1) * np.cos(y2) * np.cos(z3) - np.sin(z1) * np.sin(z3), -np.cos(z1) * np.cos(y2) * np.sin(z3) - np.sin(z1) * np.cos(z3), np.cos(z1) * np.sin(y2)],
                     [np.cos(z1) * np.sin(z3) + np.sin(z1) * np.cos(y2) * np.cos(z3), -np.sin(z1) * np.cos(y2) * np.sin(z3) + np.cos(z1) * np.cos(z3), np.sin(z1) * np.sin(y2)],
                     [-np.sin(y2) * np.cos(z3), np.sin(y2) * np.sin(z3), np.cos(y2)]])

def get_rot_mat_zyx(z1, y2, x3):
    """
    first x3, then y2, lastly z1
    """
    return np.array([[np.cos(z1) * np.cos(y2),      np.cos(z1) * np.sin(y2) * np.sin(x3) - np.sin(z1) * np.cos(x3),        np.sin(z1) * np.sin(x3) + np.cos(z1) * np.cos(x3) * np.sin(y2)],
                     [np.cos(y2) * np.sin(z1),      np.cos(z1) * np.cos(x3) + np.sin(z1) * np.sin(y2) * np.sin(x3),        np.cos(x3) * np.sin(z1) * np.sin(y2) - np.cos(z1) * np.sin(x3)],
                     [-np.sin(y2),                  np.cos(y2) * np.sin(x3),                                               np.cos(y2) * np.cos(x3)]])

    
def initialRigidAlign(moving, fixed, SearchWidth=64/180*(np.pi), numIntervals=16, minSearchWidth=8/180*(np.pi), moving_xyz=None, bi=True, fixed_img=None, verbose=False):
    assert len(moving) == len(moving_xyz), "moving feature's size is not correct"
    radius = np.amax(moving_xyz[:,0])
    if bi == False:
        neigh_orders = None
        fixed_xyz = None
        raise NotImplementedError('Not implemented.')
    
    Center1 = 0.
    bestCenter1 = 0.
    Center2 = 0.
    bestCenter2 = 0.
    Center3 = 0.
    bestCenter3 = 0.
    
    numIntervals = numIntervals+1
    curr_energy = float('inf')
    while SearchWidth >= minSearchWidth:
        for alpha in np.linspace(Center1-SearchWidth, Center1+SearchWidth, num=numIntervals):
            for beta in np.linspace(Center2-SearchWidth, Center2+SearchWidth, num=numIntervals):
                for gamma in np.linspace(Center3-SearchWidth, Center3+SearchWidth, num=numIntervals):
    
                    curr_rot = get_rot_mat_zyx(alpha, beta, gamma)
                    curr_vertices = curr_rot.dot(np.transpose(moving_xyz))
                    curr_vertices = np.transpose(curr_vertices)
    
                    if bi:
                        feat_inter = bilinearResampleSphereSurfImg(curr_vertices, fixed_img, radius=radius)
                    else:
                        feat_inter = resampleSphereSurf(fixed_xyz, curr_vertices, fixed, neigh_orders=neigh_orders)
                    
                    
                    feat_inter = np.squeeze(feat_inter)
                    tmp_energy = np.mean((feat_inter - moving)**2)
                    # tmp_energy = 1-(((feat_inter - feat_inter.mean()) * (moving - moving.mean())).mean() / feat_inter.std() / moving.std())
                       
                    if alpha == 0. and beta == 0. and gamma == 0. :
                        prev_energy = tmp_energy
                    
                    if tmp_energy < curr_energy:
                        if verbose:
                            print('Rotate by', alpha, ',', beta, ', ', gamma)
                            print('current energy: ', curr_energy)
                        curr_energy = tmp_energy
                        bestCenter1 = alpha
                        bestCenter2 = beta
                        bestCenter3 = gamma

        Center1 = bestCenter1
        Center2 = bestCenter2
        Center3 = bestCenter3
    
        SearchWidth = SearchWidth/2.
    
    return np.array([bestCenter1, bestCenter2, bestCenter3]), prev_energy, curr_energy
