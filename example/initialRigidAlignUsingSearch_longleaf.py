#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:23:30 2020

@author: fenqiang
"""

import time
import numpy as np
import argparse

from utils.utils_vtk import read_vtk, write_vtk
from utils.utils import normalize
from utils.utils_interpolation import get_latlon_img
from utils.utils_initial_rigid_align import initialRigidAlign

def get_bi_inter(n_vertex):
    inter_indices = np.load('neigh_indices/img_indices_'+ str(n_vertex) +'_0.npy')
    inter_weights = np.load('neigh_indices/img_weights_'+ str(n_vertex) +'_0.npy')
    return inter_indices, inter_weights

def get_rot_mat_zyx(z1, y2, x3):
    """
    first x3, then y2, lastly z1
    """
    return np.array([[np.cos(z1) * np.cos(y2),      np.cos(z1) * np.sin(y2) * np.sin(x3) - np.sin(z1) * np.cos(x3),        np.sin(z1) * np.sin(x3) + np.cos(z1) * np.cos(x3) * np.sin(y2)],
                     [np.cos(y2) * np.sin(z1),      np.cos(z1) * np.cos(x3) + np.sin(z1) * np.sin(y2) * np.sin(x3),        np.cos(x3) * np.sin(z1) * np.sin(y2) - np.cos(z1) * np.sin(x3)],
                     [-np.sin(y2),                  np.cos(y2) * np.sin(x3),                                               np.cos(y2) * np.cos(x3)]])

def rigidAlign(file, fixed_sulc, fixed_img):
    print(file)
    t1 = time.time()
    data = read_vtk(file)
    sulc = normalize(data['sulc'])
    
    rot_angles, prev_energy, curr_energy = initialRigidAlign(sulc, fixed_sulc,
                                           SearchWidth=50/180*(np.pi), 
                                           numIntervals=8, minSearchWidth=8/180*(np.pi),
                                           moving_xyz=data['vertices'], 
                                           bi=True, fixed_img=fixed_img, 
                                           verbose=True)
    print("prev_energy: ", prev_energy, " curr_energy: ", curr_energy)
    
    rot_mat = get_rot_mat_zyx(*rot_angles)
    rot_vertices = rot_mat.dot(np.transpose(data['vertices']))
    rot_vertices = np.transpose(rot_vertices)
    
    data['vertices'] = rot_vertices
    write_vtk(data, file.replace('.vtk', '.RigidAlignedUsingSearch.vtk'))
    
    t2 = time.time()
    print(t2-t1)
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='rigid align surface to unc atlas',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', '-f', default='None', help="filename")

    atlas = read_vtk('/nas/longleaf/home/fenqiang/Template/UNC-Infant-Cortical-Surface-Atlas/24/24_lh.SphereSurf.vtk')
    fixed_sulc = normalize(atlas['Convexity'])[0:40962]

    bi_inter= get_bi_inter(len(fixed_sulc))
    fixed_img = get_latlon_img(bi_inter, fixed_sulc)

    args =  parser.parse_args()
    file = args.file
    print(file)
    rigidAlign(file, fixed_sulc, fixed_img)