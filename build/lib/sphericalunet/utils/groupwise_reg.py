#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:42:18 2020

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import numpy as np
import torch
from itertools import combinations  

from .vtk import write_vtk
from .utils import normalize


class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, files, subs, max_age, prior_sulc_min, 
                 prior_sulc_max, prior_curv_min, prior_curv_max, n_vertex):
        self.files = files
        self.subs = subs
        self.max_age = max_age
        self.prior_sulc_min = prior_sulc_min
        self.prior_sulc_max = prior_sulc_max
        self.prior_curv_min = prior_curv_min
        self.prior_curv_max = prior_curv_max
        self.n_vertex = n_vertex

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        sulc = normalize(data[0:self.n_vertex, 0], norm_method='PriorMinMax', mi=self.prior_sulc_min, ma=self.prior_sulc_max)
        curv = normalize(data[0:self.n_vertex, 1], norm_method='PriorMinMax', mi=self.prior_curv_min, ma=self.prior_curv_max)
        data = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
        
        age = float(file.split('/')[-3].split('_')[1])
        age = age/self.max_age
        
        sub = file.split('/')[-3].split('_')[0]
        sub_id = self.subs.index(sub)
        tmp = np.zeros(len(self.subs))
        tmp[sub_id] = 1.0        
        
        return data.astype(np.float32), np.asarray(age).astype(np.float32), tmp.astype(np.float32)

    def __len__(self):
        return len(self.files)

###############################################################################

class IntraSubjectSpheres(torch.utils.data.Dataset):
    """ return single subject all time points data, maximum=8
            
    """  
    def __init__(self, files, subs, max_age, prior_sulc_min, 
                 prior_sulc_max, prior_curv_min, prior_curv_max, n_vertex):
        self.subs = subs
        self.max_age = max_age
        self.prior_sulc_min = prior_sulc_min
        self.prior_sulc_max = prior_sulc_max
        self.prior_curv_min = prior_curv_min
        self.prior_curv_max = prior_curv_max
        self.n_vertex = n_vertex
        
        self.sub_file_dic = {}
        for sub in subs:
            self.sub_file_dic[sub] = [ x for x in files if sub in x ]
        
    def __getitem__(self, index):
        sub = self.subs[index]
        files = self.sub_file_dic[sub]
        
        data = np.zeros((len(files),self.n_vertex,2))
        age = np.zeros((len(files),))
        for file in files:
            tmp = np.load(file)
            sulc = normalize(tmp[0:self.n_vertex, 0], norm_method='PriorMinMax', mi=self.prior_sulc_min, ma=self.prior_sulc_max)
            curv = normalize(tmp[0:self.n_vertex, 1], norm_method='PriorMinMax', mi=self.prior_curv_min, ma=self.prior_curv_max)
            data[files.index(file)] = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
         
            tmp = float(file.split('/')[-3].split('_')[1])
            age[files.index(file)] = tmp/self.max_age
        
        sub_id = np.zeros(len(self.subs))
        sub_id[index] = 1.0        
        
        return data.astype(np.float32), np.asarray(age).astype(np.float32), sub_id.astype(np.float32)

    def __len__(self):
        return len(self.subs)

###############################################################################

class PairwiseSpheres(torch.utils.data.Dataset):
    """ return any two surfaces in the files
            
    """  
    def __init__(self, files, prior_sulc_min, prior_sulc_max, prior_curv_min, 
                 prior_curv_max, n_vertex, val=False):
        self.prior_sulc_min = prior_sulc_min
        self.prior_sulc_max = prior_sulc_max
        self.prior_curv_min = prior_curv_min
        self.prior_curv_max = prior_curv_max
        self.n_vertex = n_vertex
        self.val = val
        
        self.files = files
        self.comb = list(combinations(list(range(len(files))), 2))
        
        
    def __getitem__(self, index):
        ind = self.comb[index]
        files = [self.files[ind[0]], self.files[ind[1]]]
        
        data = np.zeros((len(files),self.n_vertex,2))
        for file in files:
            tmp = np.load(file)
            sulc = normalize(tmp[0:self.n_vertex, 0], norm_method='PriorMinMax', mi=self.prior_sulc_min, ma=self.prior_sulc_max)
            curv = normalize(tmp[0:self.n_vertex, 1], norm_method='PriorMinMax', mi=self.prior_curv_min, ma=self.prior_curv_max)
            data[files.index(file)] = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
         
        if self.val:
            return data.astype(np.float32), files
        else:
            return data.astype(np.float32)

    def __len__(self):
        return len(self.comb)


###############################################################################

class DeformPool():
    """This class implements a buffer that stores previously generated
    deformation fields.

    This buffer enables us to update generation network using a history of generated 
    deformation fileds.
    """

    def __init__(self, pool_size, running_weight, n_vertex, device):
        """Initialize the DeformPool class

        Parameters:
            n_vertex (int) -- the number of vertices of this deformation field
            device
        """
        # assert pool_size > 0, "error"
        # assert type(pool_size) is int, "type error"
        
        self.pool_size = pool_size
        self.num_phis = 0
        self.mean = torch.zeros((n_vertex, 3), dtype=torch.float32, device=device)
        self.running_weight = running_weight

    def add(self, phi_3d):
        """Add the generated deformation filed to this buffer.

        Parameters
        ----------
        phi_3d : torch.tensor, shape [n_vertex, 3]
            the deformation field to be added.

        Return None

        """
        self.mean = self.mean.detach()
        
        # decay the running weight from 1 to stable running_weight (e.g., 0.1)
        if self.num_phis < self.pool_size:
            new_weight = -(1.-self.running_weight)/self.pool_size * self.num_phis + 1.
            self.mean = self.mean * (1.-new_weight) + phi_3d * new_weight
            self.num_phis = self.num_phis + 1.
        else:
            self.mean = self.mean * (1-self.running_weight) + phi_3d * self.running_weight
            self.num_phis = self.pool_size
        
    def get_mean(self):
        """Return the mean of deformation fileds from the pool.
        
        """
        return self.mean
    