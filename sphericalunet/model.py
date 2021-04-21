#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import torch
import torch.nn as nn
from .utils.utils import Get_neighs_order, Get_upconv_index
from .layers import onering_conv_layer, pool_layer, upconv_layer
#from unet_parts import *

class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()


#        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica unet
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x
    
    
class Unet(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch, level=7, n_res=5, rotated=0):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            level (int) - - input surface's icosahedron level. default: 7, for 40962 vertices
                            2:42, 3:162, 4:642, 5:2562, 6:10242
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or 
                               90 degrees along x axis (1)
        """
        super(Unet, self).__init__()
        
        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than iput level"
        assert n_res >=2, "number of resolution levels should be larger than 2"     
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*8)
        
        conv_layer = onering_conv_layer
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
      
        self.up = nn.ModuleList([])
        for i in range(n_res-1):
            self.up.append(up_block(conv_layer, chs[n_res-i], chs[n_res-1-i],
                                    neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
            
        self.outc = nn.Linear(chs[1], out_ch)
                
        self.n_res = n_res
        
    def forward(self, x):
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        x = xs[-1]
        for i in range(self.n_res-1):
            x = self.up[i](x, xs[self.n_res-1-i])

        x = self.outc(x) # N * 2
        return x
        



class GenAgeNet(nn.Module):
    """Generation model for atlas construction

    """    
    def __init__(self, level=6, gender=False, out_ch=2):
        """ Initialize the model.

        Parameters:
            n_sub (int) - -  number of the subjects in the group
            level (int) - -  The generated atlas level, default 6 with 10242 vertices
            age (bool) - -   add variable age?
            gender (bool) - -add variable gender? 
        """
        super(GenAgeNet, self).__init__()
        
        self.gender = gender
        self.level = level
        # self.n_sub = n_sub
        
        neigh_orders = Get_neighs_order(rotated=0)
        neigh_orders = neigh_orders[8-level:]
        upconv_index = Get_upconv_index(rotated=0)[(8-level)*2:4]
        
        n_vertex = int(len(neigh_orders[0])/7)
        assert n_vertex in [42,642,2562,10242,40962,163842]
        self.n_vertex = n_vertex

        self.fc_age = nn.Linear(1, 256)
        
        if gender is False:
            chs_0 = 256
        elif gender is True:
            chs_0 = 258  # add variable gender here
        else:
            raise NotImplementedError('Not implemented.')
        
        chs = [3, 8, 8, out_ch]
        if level <= 6:
            self.fc = nn.Linear(chs_0, chs[0]*n_vertex)
        else:
            self.fc = nn.Linear(chs_0, chs[0]*10242)
        
        if level > 6 :
            upblock_list = []
            for i in range(level-6):
                upblock_list.append(nn.BatchNorm1d(chs[0], momentum=0.15, affine=True, track_running_stats=False))
                upblock_list.append(nn.LeakyReLU(0.2))
                upblock_list.append(upconv_layer(chs[0], chs[0], upconv_index[-i*2-2], upconv_index[-i*2-1]))
            self.upconv = torch.nn.Sequential(*upblock_list)
    
        conv_list = []
        for i in range(len(chs)-1):
            conv_list.append(nn.BatchNorm1d(chs[i], momentum=0.15, affine=True, track_running_stats=False))
            conv_list.append(nn.LeakyReLU(0.2))
            conv_list.append(onering_conv_layer(chs[i], chs[i+1], neigh_orders[0]))
        self.conv_block = torch.nn.Sequential(*conv_list)
        
    def forward(self, age=0, gender=0):
        # assert sub_id.shape == torch.Size([1, self.n_sub])
        # x_sub = self.fc_sub(sub_id)      # 1*1024
        assert age.shape == torch.Size([1, 1])
        x_age = self.fc_age(age)     # 1*256
        if self.gender:
            assert gender.shape == torch.Size([1, 2])
            x = torch.cat((x_age, gender),1)   # 1*2050
        else:
            x = x_age
            
        x = self.fc(x) # 1* (10242*3)
        if self.n_vertex <= 10242:
            x = torch.reshape(x, (self.n_vertex,-1)) # 10242 * 3
        else:
            x = torch.reshape(x, (10242,-1))  # 10242 * 3
            x = self.upconv(x)
            
        x = self.conv_block(x)
        
        return x
    

class GenPhiUsingSubId(nn.Module):
    """Generating deformation field from atlas to within-subject-mean

    """    
    def __init__(self, level, n_sub):
        """ Initialize the model.

        Parameters:
            n_sub (int) - -  number of the subjects in the group
            level (int) - -  The generated atlas level, default 6 with 10242 vertices
            age (bool) - -   add variable age?
            gender (bool) - -add variable gender? 
        """
        super(GenPhiUsingSubId, self).__init__()
        
        self.level = level
        self.n_sub = n_sub
        
        neigh_orders = Get_neighs_order(rotated=0)
        neigh_orders = neigh_orders[8-level:]
        upconv_index = Get_upconv_index(rotated=0)[(8-level)*2:4]
        
        n_vertex = int(len(neigh_orders[0])/7)
        assert n_vertex in [42,642,2562,10242,40962,163842]
        self.n_vertex = n_vertex

        self.fc_sub = nn.Linear(n_sub, 256)
        
        chs_0 = 256
        
        chs = [3, 8, 8, 2]
        if level <= 6:
            self.fc = nn.Linear(chs_0, chs[0]*n_vertex)
        else:
            self.fc = nn.Linear(chs_0, chs[0]*10242)
        
        if level > 6 :
            upblock_list = []
            for i in range(level-6):
                upblock_list.append(nn.BatchNorm1d(chs[0], momentum=0.15, affine=True, track_running_stats=False))
                upblock_list.append(nn.LeakyReLU(0.2))
                upblock_list.append(upconv_layer(chs[0], chs[0], upconv_index[-i*2-2], upconv_index[-i*2-1]))
            self.upconv = torch.nn.Sequential(*upblock_list)
    
        conv_list = []
        for i in range(len(chs)-1):
            conv_list.append(nn.BatchNorm1d(chs[i], momentum=0.15, affine=True, track_running_stats=False))
            conv_list.append(nn.LeakyReLU(0.2))
            conv_list.append(onering_conv_layer(chs[i], chs[i+1], neigh_orders[0]))
        self.conv_block = torch.nn.Sequential(*conv_list)
        
    def forward(self, sub_id):
        assert sub_id.shape == torch.Size([1, self.n_sub])
        x = self.fc_sub(sub_id)      # 1*1024
        x = self.fc(x) # 1* (10242*3)
        if self.n_vertex <= 10242:
            x = torch.reshape(x, (self.n_vertex,-1)) # 10242 * 3
        else:
            x = torch.reshape(x, (10242,-1))  # 10242 * 3
            x = self.upconv(x)
            
        x = self.conv_block(x)
        
        return x
        
