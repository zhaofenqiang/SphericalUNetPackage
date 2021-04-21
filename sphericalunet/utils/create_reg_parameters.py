#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""
import torch, os
import numpy as np

from .utils import get_neighs_order, get_upsample_order
from .vtk import read_vtk
from .interp_torch import getEn, get_bi_inter, getOverlapIndex
from .groupwise_reg import DeformPool

abspath = os.path.abspath(os.path.dirname(__file__))

def readRegConfig(file="./regConfig.txt"):
    """
    

    Returns registration configuration parameters
    -------
    None.

    """
    config = {}
    keys = []
    with open(file) as f:    
        lines = f.readlines() 
        for line in lines: 
            if line != '\n':
                tmp = line.strip().split(" ")
                config[tmp[0]] = tmp[1]
                keys.append(tmp[0])
                
    config_new = {}

    for key in keys:
        if key in ['levels']:
            config_new[key] = [ int(x) for x in config[key].split(',') ]
            n_levels = len(config_new[key])
        elif key in ['weight_l2', 'weight_smooth', \
                     'weight_phi_consis','weight_corr','weight_intra',\
                     'weight_inter','weight_centra','weight_level',\
                     'weight_long', 'weight_sub_smooth']:
            config_new[key] = [ float(x) for x in config[key].split(',') ]
            assert len(config_new[key]) == n_levels
        elif key in ['features']:
            config_new[key] = config[key].split(',')
            assert len(config_new[key]) == n_levels
        elif key in ['diffe', 'bi','long', 'centra']:
            config_new[key] = config[key] == 'True'
            # assert len(config_new[key]) == n_levels
        elif key in ['num_composition']:
            config_new[key] = int(config[key])
        elif key in ['learning_rate']:
            config_new[key] = float(config[key])
        elif key in ['truncated']:
            config_new[key] = [ x == 'True' for x in config[key].split(',') ]
            assert len(config_new[key]) == n_levels
        else:
            raise NotImplementedError("Unrecognized keys")

    return config_new


# load fixed/atlas surface, smooth filter, global parameter pre-defined
def get_fixed_xyz(n_vertex, device):
    fixed_0 = read_vtk(abspath+'/neigh_indices/sphere_' + str(n_vertex) +'_rotated_0.vtk')
    
    fixed_xyz_0 = fixed_0['vertices']/100.0  # fixed spherical coordinate
    fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).cuda(device)
    
    # fixed_1 = read_vtk(abspath+'/neigh_indices/sphere_' + str(n_vertex) +'_rotated_1.vtk')
    # fixed_xyz_1 = fixed_1['vertices']/100.0  # fixed spherical coordinate
    # fixed_xyz_1 = torch.from_numpy(fixed_xyz_1.astype(np.float32)).cuda(device)
    
    # fixed_2 = read_vtk(abspath+'/neigh_indices/sphere_' + str(n_vertex) +'_rotated_2.vtk')
    # fixed_xyz_2 = fixed_2['vertices']/100.0  # fixed spherical coordinate
    # fixed_xyz_2 = torch.from_numpy(fixed_xyz_2.astype(np.float32)).cuda(device)
    
    return fixed_xyz_0


def createRegConfig(config):
    n_vertexs = config['n_vertexs']
    device = config['device']
    
    fixed_xyz_0 = get_fixed_xyz(n_vertexs[-1], device)
    config['fixed_xyz_0'] = fixed_xyz_0
    
    neigh_orders = []
    for n_vertex in n_vertexs:
        tmp = get_neighs_order(abspath+'/neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_' + str(0) + '.mat')
        neigh_orders.append(torch.from_numpy(tmp).to(device))
    config['neigh_orders'] = neigh_orders
    
    Ens = []
    for n_vertex in n_vertexs:
        Ens.append(getEn(int(n_vertex), device))
    config['Ens'] = Ens
    
    merge_indexs = []
    for n_vertex in n_vertexs:
        merge_indexs.append(getOverlapIndex(n_vertex, device))
    config['merge_indexs'] = merge_indexs
    
    bi_inter_0s = []
    for n_vertex in n_vertexs:
        bi_inter_0s.append(get_bi_inter(n_vertex, device)[0])
    config['bi_inter_0s'] = bi_inter_0s
  
    upsample_neighborss = []
    for i_level in range(config['n_levels']):
        if i_level >=1:
            upsample_neighborss.append(get_upsample_order(n_vertexs[i_level]))
    config['upsample_neighborss'] = upsample_neighborss
    
    
    grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
    grad_filter[6] = -6    
    config['grad_filter'] = grad_filter
    
    config['atlas'] = read_vtk(abspath+'/neigh_indices/sphere_' + \
                     str(config['n_vertexs'][-1]) + '_rotated_0.vtk') 
        
    if 'pool_size' in config:
        deform_pools = []
        for n_vertex in n_vertexs:
            deform_pools.append(DeformPool(config['pool_size'], config['running_weight'], n_vertex, device))
        config['deform_pools'] = deform_pools
      
    # sulc_std = 4.7282835 
    # sulc_mean = 0.31142648
    config['norm_method'] = 'PrioiMaxMin'
    config['prior_sulc_min'] =  -12.
    config['prior_sulc_max'] = 14.
    config['prior_curv_min'] = -1.3
    config['prior_curv_max'] = 1.0
      
    return config







