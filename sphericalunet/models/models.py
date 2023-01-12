#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import torch
import torch.nn as nn


from sphericalunet.utils.utils import Get_neighs_order, Get_upconv_index, \
    Get_swin_matrices_2order, get_neighs_order
from sphericalunet.models.layers import onering_conv_layer, pool_layer, upconv_layer, \
     pool_layer_batch, upconv_layer_batch, self_attention_layer_swin,\
        upconv_layer_batch_average



class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()

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
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index, joint_ch=0):
        super(up_block, self).__init__()
        
        if joint_ch != 0:
            self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
            self.double_conv = nn.Sequential(
                 conv_layer(joint_ch, out_ch, neigh_orders),
                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                 nn.LeakyReLU(0.2, inplace=True),
                 conv_layer(out_ch, out_ch, neigh_orders),
                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                 nn.LeakyReLU(0.2, inplace=True)
                 )
        else:
            # suppose in_ch = out_ch * 2
            self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
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
    
    
    
class simple_up_block(nn.Module):
    """ simplified up_block without skip connections and joint features
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(simple_up_block, self).__init__()
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        self.double_conv = nn.Sequential(
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
             )

    def forward(self, x):
        x = self.up(x)
        x = self.double_conv(x)
        return x

    
    
class down_block_batch(nn.Module):
    """
    Batch norm version of
    downsampling block in spherical transformer unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False, drop_rate=None, num_heads=1, neigh_orders_2=None):
        super(down_block_batch, self).__init__()

        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders, neigh_orders_2, drop_rate=drop_rate),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.LayerNorm([out_ch, int(neigh_orders_2.shape[0]//7)]),
                # nn.InstanceNorm1d(out_ch, momentum=0.15),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
                conv_layer(out_ch, out_ch, neigh_orders, neigh_orders_2, drop_rate=drop_rate),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.LayerNorm([out_ch, int(neigh_orders_2.shape[0]//7)]),
                # nn.InstanceNorm1d(out_ch, momentum=0.15),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False)
        )

            
        else:
            self.block = nn.Sequential(
                pool_layer_batch(pool_neigh_orders),
                conv_layer(in_ch, out_ch, neigh_orders, drop_rate=drop_rate),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
                conv_layer(out_ch, out_ch, neigh_orders, drop_rate=drop_rate),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        return x
    

class up_block_batch(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block_batch, self).__init__()
        
        self.up = upconv_layer_batch(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
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
        # x = torch.cat((x1, x2), 2) 
        # x = self.double_conv(x.permute(0,2,1)).permute(0,2,1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class patch_merging_layer(nn.Module):
    def __init__(self, neigh_orders, in_ch, out_ch):
        super(patch_merging_layer, self).__init__()

        self.neigh_orders = neigh_orders
        self.proj = nn.Conv1d(int(in_ch*7), out_ch, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False)
        # self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=False, track_running_stats=True)
        # self.norm = nn.InstanceNorm1d(out_ch, momentum=0.15)
    
    def forward(self, x):
        batch_num, feat_num, num_nodes = x.shape
        num_nodes = int((x.size()[2]+6)/4)
        feat_num = x.size()[1]
        x = x[:, :, self.neigh_orders[0:num_nodes*7]].view(batch_num, feat_num, num_nodes, 7).permute((0, 3, 1, 2)).reshape((batch_num, -1, num_nodes))
        x = self.norm(self.proj(x))
        return x


class patch_merging_layer_average(nn.Module):
    def __init__(self, neigh_orders, in_ch, out_ch):
        super(patch_merging_layer_average, self).__init__()

        self.neigh_orders = neigh_orders
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        # self.norm = nn.LayerNorm([out_ch, int((neigh_orders.shape[0]//7+6)/4)])
        self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False)
        # self.norm = nn.InstanceNorm1d(out_ch, momentum=0.15, affine=True)
    
    def forward(self, x):
        batch_num, feat_num, num_nodes = x.shape
        num_nodes = int((x.size()[2]+6)/4)
        feat_num = x.size()[1]
        x = x[:, :, self.neigh_orders[0:num_nodes*7]].view(batch_num, feat_num, num_nodes, 7).permute((0, 3, 1, 2))
        x = torch.Tensor.mean(x, dim=1)
        x = self.norm(self.proj(x))
        return x


class SUnet(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch, level=7, n_res=5, rotated=0, complex_chs=16):
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
        super(SUnet, self).__init__()
        
        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than iput level"
        assert n_res >=2, "number of resolution levels should be larger than 2"     
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
        
        neigh_orders = Get_neighs_order(rotated)
        # import pdb
        # pdb.set_trace()
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*complex_chs)
        
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
        # x's size should be [N (number of vertices) x C (channel)]
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        x = xs[-1]
        for i in range(self.n_res-1):
            x = self.up[i](x, xs[self.n_res-1-i])

        x = self.outc(x) # N * 2
        return x
        
    

class UNet18_40k_SWIN_pred_bigtail_average(nn.Module):
    """Spherical Transformer network architecture

    """    
    def __init__(self, in_ch, out_ch, drop_rate=0.8):
        """ Initialize the Spherical transformer.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        Input: 
            B (batch size) x N (number of vertices) x in_ch tensor
        Return:
            B x N x out_ch tensor
        """
        super(UNet18_40k_SWIN_pred_bigtail_average, self).__init__()
        
        neigh_orders = Get_neighs_order(0)
        matrices = Get_swin_matrices_2order()
        upconv_index = Get_upconv_index(0)
        # upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
   
        chs = [8, 16, 32, 64, 64] # default

        conv_layer = self_attention_layer_swin

        self.init = nn.Conv1d(in_ch, chs[0], kernel_size=1)
        self.down1 = down_block_batch(conv_layer, chs[0], chs[0], matrices[0], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[1])
        self.patch_merging_2 = patch_merging_layer_average(neigh_orders[1], chs[0], chs[1])

        self.down2 = down_block_batch(conv_layer, chs[1], chs[1], matrices[1], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[2])
        self.patch_merging_3 = patch_merging_layer_average(neigh_orders[2], chs[1], chs[2])

        self.down3 = down_block_batch(conv_layer, chs[2], chs[2], matrices[2], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[3])
        self.patch_merging_4 = patch_merging_layer_average(neigh_orders[3], chs[2], chs[3])

        self.down4 = down_block_batch(conv_layer, chs[3], chs[3], matrices[3], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[4])
        self.patch_merging_5 = patch_merging_layer_average(neigh_orders[4], chs[3], chs[4])

        self.down5 = down_block_batch(conv_layer, chs[4], chs[4], matrices[4], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[5])

        self.upsample_1 = upconv_layer_batch_average(chs[4], chs[4], upconv_index[-4], upconv_index[-3])
        self.up1 = down_block_batch(conv_layer, int(chs[4]*2), chs[4], matrices[3], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[4])
        self.trans1 = down_block_batch(conv_layer, chs[3], chs[4], matrices[3], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[4])

        self.upsample_2 = upconv_layer_batch_average(chs[4], chs[4], upconv_index[-6], upconv_index[-5])
        self.up2 = down_block_batch(conv_layer, int(chs[4]*2), chs[4], matrices[2], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[3])
        self.trans2 = down_block_batch(conv_layer, chs[2], chs[4], matrices[2], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[3])

        self.upsample_3 = upconv_layer_batch_average(chs[4], chs[4], upconv_index[-8], upconv_index[-7])
        self.up3 = down_block_batch(conv_layer, int(chs[4]*2), chs[4], matrices[1], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[2])
        self.trans3 = down_block_batch(conv_layer, chs[1], chs[4], matrices[1], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[2])

        self.upsample_4 = upconv_layer_batch_average(chs[4], chs[4], upconv_index[-10], upconv_index[-9])
        self.up4 = down_block_batch(conv_layer, int(chs[4]*2), chs[4], matrices[0], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[1])
        self.trans4 = down_block_batch(conv_layer, chs[0], chs[4], matrices[0], None, True, drop_rate=drop_rate, neigh_orders_2=neigh_orders[1])

        self.outc = nn.Sequential(
                nn.Conv1d(chs[4], out_ch, kernel_size=1)
                )
        self.apply(weight_init)
        self.grads = {}
    
        
    def forward(self, x):
        B, N, C = x.shape
        x1 = torch.permute(x, (0, 2, 1))

        x1 = self.init(x1)
        x1_1 = self.down1(x1)
        x2 = self.patch_merging_2(x1+x1_1)
        x2_1 = self.down2(x2)
        x3 = self.patch_merging_3(x2+x2_1)
        x3_1 = self.down3(x3)
        x4 = self.patch_merging_4(x3+x3_1)
        x4_1 = self.down4(x4)
        x5 = self.patch_merging_5(x4+x4_1)
        x5_1 = self.down5(x5)

        x = self.up1(torch.cat((self.upsample_1(x5_1), self.trans1(x4_1)), dim=1))
        x = self.up2(torch.cat((self.upsample_2(x), self.trans2(x3_1)), dim=1))
        x = self.up3(torch.cat((self.upsample_3(x), self.trans3(x2_1)), dim=1))
        x = self.up4(torch.cat((self.upsample_4(x), self.trans4(x1_1)), dim=1))

        x = self.outc(x)
        x = x.permute(0, 2, 1)
        return x


    
class JointRegAndParc(nn.Module):
    """Joint registration and parcellation network

    """ 
    def __init__(self, in_ch, out_parc_ch, level=7, n_res=5, rotated=0, complex_chs=8):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            level (int) - - input surface's icosahedron level. default: 7 with 40962 vertices
                            2:42, 3:162, 4:642, 5:2562, 6:10242
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or 
                               90 degrees along x axis (1)
            complex_chs (int) - - intermidiate channels for controlling the total paprameters/complexity of the network
        """
        super(JointRegAndParc, self).__init__()
        
        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than input level"
        assert n_res >=2, "number of resolution levels should be larger than 2"     
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
        
        self.n_res = n_res
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*complex_chs)
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(onering_conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(onering_conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
       
        self.up_parc = nn.ModuleList([])
        for i in range(n_res-1):
            self.up_parc.append(up_block(onering_conv_layer, chs[n_res-i], chs[n_res-1-i],
                                    neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
        self.outc_parc = nn.Linear(chs[1], out_parc_ch)
                
        
        self.up_reg = nn.ModuleList([])
        self.up_reg.append(up_block(onering_conv_layer, chs[n_res]*2, chs[n_res-1],
                                    neigh_orders[n_res-2], upconv_indices[(n_res-2)*2], upconv_indices[(n_res-2)*2+1],
                                    joint_ch=chs[n_res-1]*3))
        for i in range(1, n_res-1):
            self.up_reg.append(up_block(onering_conv_layer, chs[n_res-i], chs[n_res-1-i],
                                        neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1],
                                        joint_ch=chs[n_res-1-i]*3))
    
        self.outc_reg = nn.Linear(chs[1], 2)
        
        
        
    def forward(self, x1, x2):
        
        # Surface 1
        x1s = [x1]
        for i in range(self.n_res):
            x1s.append(self.down[i](x1s[i]))
            
        # parc
        x = x1s[-1]
        for i in range(self.n_res-1):
            x = self.up_parc[i](x, x1s[self.n_res-1-i])
        x1_parc = self.outc_parc(x) # N * 36
                
        # Surface 2
        x2s = [x2]
        for i in range(self.n_res):
            x2s.append(self.down[i](x2s[i]))
            
        # parc
        x = x2s[-1]
        for i in range(self.n_res-1):
            x = self.up_parc[i](x, x2s[self.n_res-1-i])
        x2_parc = self.outc_parc(x) # N * 36
        
        # reg
        ys = []
        for i in range(len(x1s)):
            ys.append(torch.cat((x1s[i], x2s[i]), dim=1))
            
        x = ys[-1]
        for i in range(self.n_res-1):
            x = self.up_reg[i](x, ys[self.n_res-1-i])

        x_reg = self.outc_reg(x) # N * 2
    
        return x1_parc, x2_parc, x_reg
    

class LongJointRegAndParc(nn.Module):
    """Joint registration and parcellation network

    """ 
    def __init__(self, in_ch, out_parc_ch, level=7, n_res=5, rotated=0, complex_chs=8, num_long_scans=8):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            level (int) - - input surface's icosahedron level. default: 7 with 40962 vertices
                            2:42, 3:162, 4:642, 5:2562, 6:10242
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or 
                               90 degrees along x axis (1)
            complex_chs (int) - - intermidiate channels for controlling the total paprameters/complexity of the network
        """
        super(LongJointRegAndParc, self).__init__()
        
        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than input level"
        assert n_res >=2, "number of resolution levels should be larger than 2"     
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
        
        self.n_res = n_res
        self.num_long_scans = num_long_scans
        self.in_ch = in_ch
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*complex_chs)
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(onering_conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(onering_conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
       
        self.up_parc = nn.ModuleList([])
        for i in range(n_res-1):
            self.up_parc.append(up_block(onering_conv_layer, chs[n_res-i], chs[n_res-1-i],
                                    neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
        self.outc_parc = nn.Linear(chs[1], out_parc_ch)
                
        
        self.up_reg = nn.ModuleList([])
        self.up_reg.append(up_block(onering_conv_layer, chs[n_res]*(num_long_scans+1), chs[n_res-1],
                                    neigh_orders[n_res-2], upconv_indices[(n_res-2)*2], upconv_indices[(n_res-2)*2+1],
                                    joint_ch=chs[n_res-1]*(num_long_scans+2)))
        for i in range(1, n_res-1):
            self.up_reg.append(up_block(onering_conv_layer, chs[n_res-i], chs[n_res-1-i],
                                        neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1],
                                        joint_ch=chs[n_res-1-i]*(num_long_scans+2)))
    
        self.outc_reg = nn.Linear(chs[1], num_long_scans*2)
        
        
        
    def forward(self, x, x2):
        """
        x2 is the atlas features

        """
        
        # encoder for surface 1 to self.num_long_scans
        xis = []
        for i in range(self.num_long_scans):
            xs = [x[i]]
            for j in range(self.n_res):
                xs.append(self.down[j](xs[j]))
            xis.append(xs)
        
        # parc
        xs_parc = []
        for i in range(self.num_long_scans):
            xs = xis[i]
            y = xs[-1]
            for j in range(self.n_res-1):
                y = self.up_parc[j](y, xs[self.n_res-1-j])
            xs_parc.append(self.outc_parc(y)) # 6* N * 36
            
        
        # Surface 2
        x2s = [x2]
        for i in range(self.n_res):
            x2s.append(self.down[i](x2s[i]))
        # parc
        z = x2s[-1]
        for i in range(self.n_res-1):
            z = self.up_parc[i](z, x2s[self.n_res-1-i])
        x2_parc = self.outc_parc(z) # N * 36
        
        
        # reg
        feats = []
        for i in range(len(xis[0])):
            tmp = x2s[i]
            for j in range(self.num_long_scans):
                tmp = torch.cat((tmp, xis[j][i]), dim=1)
            feats.append(tmp)
            
        u = feats[-1]
        for i in range(self.n_res-1):
            u = self.up_reg[i](u, feats[self.n_res-1-i])

        x_reg = self.outc_reg(u) # num_long_scans x 2
    
        return xs_parc, x2_parc, x_reg




class svgg(nn.Module):
    def __init__(self, in_ch, out_ch, level, n_res, rotated=0):
        super(svgg, self).__init__()
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        conv_layer = onering_conv_layer

        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*32)

        sequence = []
        sequence.append(conv_layer(chs[0], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
            
        for l in range(1, len(chs)-1):
            sequence.append(pool_layer(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))
            sequence.append(conv_layer(chs[l+1], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*sequence)    
        self.fc =  nn.Sequential(
                nn.Linear(chs[-1], out_ch)
                )

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        return x



class res_block(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders, first_in_block=False):
        super(res_block, self).__init__()
        
        self.conv1 = onering_conv_layer(c_in, c_out, neigh_orders)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = onering_conv_layer(c_out, c_out, neigh_orders)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.first = first_in_block
    
    def forward(self, x):
        res = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.first:
            res = torch.cat((res,res),1)
        x = x + res
        x = self.relu(x)
        
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResNet, self).__init__()
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        self.conv1 =  onering_conv_layer(in_c, 64, neigh_orders_40962)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(0.2)
        
        self.pool1 = pool_layer(neigh_orders_40962, 'max')
        self.res1_1 = res_block(64, 64, neigh_orders_10242)
        self.res1_2 = res_block(64, 64, neigh_orders_10242)
        self.res1_3 = res_block(64, 64, neigh_orders_10242)
        
        self.pool2 = pool_layer(neigh_orders_10242, 'max')
        self.res2_1 = res_block(64, 128, neigh_orders_2562, True)
        self.res2_2 = res_block(128, 128, neigh_orders_2562)
        self.res2_3 = res_block(128, 128, neigh_orders_2562)
        
        self.pool3 = pool_layer(neigh_orders_2562, 'max')
        self.res3_1 = res_block(128, 256, neigh_orders_642, True)
        self.res3_2 = res_block(256, 256, neigh_orders_642)
        self.res3_3 = res_block(256, 256, neigh_orders_642)
        
        self.pool4 = pool_layer(neigh_orders_642, 'max')
        self.res4_1 = res_block(256, 512, neigh_orders_162, True)
        self.res4_2 = res_block(512, 512, neigh_orders_162)
        self.res4_3 = res_block(512, 512, neigh_orders_162)
                
        self.pool5 = pool_layer(neigh_orders_162, 'max')
        self.res5_1 = res_block(512, 1024, neigh_orders_42, True)
        self.res5_2 = res_block(1024, 1024, neigh_orders_42)
        self.res5_3 = res_block(1024, 1024, neigh_orders_42)
        
        self.fc = nn.Linear(1024, out_c)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pool1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        
        x = self.pool2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        
        x = self.pool3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
                
        x = self.pool4(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        
        x = self.pool5(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        x = self.out(x)
        return x




class GenAtlasConditionedOnAge(nn.Module):
    """Generation model for atlas construction

    """    
    def __init__(self, level=7, gender=False, out_ch=2):
        """ Initialize the model.

        Parameters:
            level (int) - -  The generated atlas level, default 7 with 40962 vertices
            gender (bool) - -add variable gender? 
            out_ch - -    output channels, default 2 for sulc and curv
        """
        super(GenAtlasConditionedOnAge, self).__init__()
        
        self.gender = gender
        self.level = level
        # self.n_sub = n_sub
        
        neigh_orders = Get_neighs_order(rotated=0)
        neigh_orders = neigh_orders[8-level:]
        upconv_index = Get_upconv_index(rotated=0)[(8-level)*2:4]
        
        n_vertex = int(len(neigh_orders[0])/7)
        assert n_vertex in [42,642,2562,10242,40962,163842]
        self.n_vertex = n_vertex

        self.fc_age = nn.Linear(1, 64)
        
        if gender is False:
            chs_0 = 64
        elif gender is True:
            chs_0 = 66  # add variable gender here
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
        x_age = self.fc_age(age)     # 1*64
        if self.gender:
            assert gender.shape == torch.Size([1, 2])
            x = torch.cat((x_age, gender),1)   # 1*66
        else:
            x = x_age
            
        x = self.fc(x) # 1* (10242*3)
        if self.n_vertex <= 10242:
            x = torch.reshape(x, (self.n_vertex, -1)) # 10242 * 3 or 2562 * 3 
        else:
            x = torch.reshape(x, (10242, -1))  # 10242 * 3
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
        

 

if __name__ == "__main__":
    
    data = torch.rand((2, 40962, 3)).cuda()
    model = UNet18_40k_SWIN_pred_bigtail_average(in_ch=3, out_ch=32, drop_rate=0.0)
    model = model.cuda()

    output = model(data)
    print("True")
