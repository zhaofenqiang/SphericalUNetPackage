#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:18:30 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""


import torch
import numpy as np
import torch.nn as nn
# from timm.models.layers import  trunc_normal_


class repa_conv_layer(nn.Module):
    """Define the convolutional layer on icosahedron discretized sphere using 
    rectagular filter in tangent plane
    
    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels    
            
    Input: 
        N x in_feats, tensor
    Return:
        N x out_feats, tensor
    """    
    def __init__(self, in_feats, out_feats, neigh_indices, neigh_weights):
        super(repa_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_indices = neigh_indices.reshape(-1) - 1
        self.weight = nn.Linear(25 * in_feats, out_feats)
        self.nodes_number = neigh_indices.shape[0]
        
        neigh_weights = np.reshape(np.tile(neigh_weights, self.in_feats), (neigh_weights.shape[0],neigh_weights.shape[1],3,-1)).astype(np.float32)
        self.neigh_weights = torch.from_numpy(neigh_weights).cuda()    
        
    def forward(self, x):
      
        mat = x[self.neigh_indices]
        mat = mat.view(self.nodes_number, 25, 3, -1)
        assert(mat.size() == torch.Size([self.nodes_number, 25, 3, self.in_feats]))
   
        assert(mat.size() == self.neigh_weights.size())

        x = torch.mul(mat, self.neigh_weights)
        x = torch.sum(x, 2).view(self.nodes_number, -1)
        assert(x.size() == torch.Size([self.nodes_number, 25 * self.in_feats]))
        
        out = self.weight(x)
        return out


class onering_conv_layer(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using 
    1-ring filter
    
    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels
            
    Input: 
        N x in_feats tensor
    Return:
        N x out_feats tensor
    """  
    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices=None, neigh_weights=None):
        super(onering_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(7 * in_feats, out_feats)
        
    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        mat = x[self.neigh_orders]
        mat = mat.view(len(x), 7*self.in_feats)
                
        out_features = self.weight(mat)
        return out_features
    

class onering_conv_layer_batch(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using 
    1-ring filter
    
    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels
            
    Input: 
        B x N x in_feats tensor
    Return:
        B x N x out_feats tensor
    """  
    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices=None, neigh_weights=None, drop_rate=None):
        super(onering_conv_layer_batch, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(7 * in_feats, out_feats)
        
    def forward(self, x):
        # import pdb
        # pdb.set_trace()

        mat = x[:, :, self.neigh_orders].view(x.shape[0], x.shape[2], 7*self.in_feats)
        
        # import pdb
        # pdb.set_trace()
        out_features = self.weight(mat).permute(0, 2, 1)

        return out_features
 
    

class tworing_conv_layer(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using 
    2-ring filter
    
    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels
            
    Input: 
        N x in_feats tensor
    Return:
        N x out_feats tensor
    """  
    def __init__(self, in_feats, out_feats, neigh_orders):
        super(tworing_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(19 * in_feats, out_feats)
        
    def forward(self, x):
       
        mat = x[self.neigh_orders].view(len(x), 19*self.in_feats)
                
        out_features = self.weight(mat)
        return out_features
        

class graph_one_ring_conv(nn.Module):
    """The 1-ring convolutional layer on any graph structures with sampled fixed 
    neighborhood vertices
    
    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels
            
    Input: 
        N x in_feats tensor
    Return:
        N x out_feats tensor
    """  
    def __init__(self, in_feats, out_feats, num_neighbors):
        super(graph_one_ring_conv, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_neighbors = num_neighbors
        
        self.weight = nn.Linear(num_neighbors * in_feats, out_feats)
        
    def forward(self, x, neigh_sorted_orders):
       
        mat = x[neigh_sorted_orders].view(len(x), self.num_neighbors*self.in_feats)
                
        out_features = self.weight(mat)
        return out_features
    

class self_attention_layer_swin(nn.Module):
    """The self-attention layer on icosahedron discretized sphere based on
    2-ring filter
    
    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels
            num_heads (int) - - Number of attention heads
            qkv_bias （bool） - - If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float) - - Override default qk scale of head_dim ** -0.5 if set
            neigh_orders (ndarray) - - The indices of vertices used for patch partitioning
    Input: 
        B x in_feats x N tensor 
    Return:
        B x out_feats x N tensor
    """  
    def __init__(self, in_feats, out_feats, neigh_orders, neigh_orders_2=None, head_dim=8,
        qkv_bias=True, qk_scale=None, sep_process=True, drop_rate=None):
        super(self_attention_layer_swin, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.top = 16
        self.down = 19

        self.neigh_orders_top = neigh_orders['top'].reshape((-1, self.top))
        self.neigh_orders_down = neigh_orders['down'].reshape((-1, self.down))
        self.reverse_matrix = neigh_orders['reverse']
        self.cnt_matrix = nn.parameter.Parameter(torch.from_numpy((1 / neigh_orders['count']).astype(np.float32)), requires_grad=False)

        self.padding = nn.ZeroPad2d((0, 0, 0, 1))
        
        self.num_heads = in_feats // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.sep_process = sep_process

        if drop_rate:
            self.drop_rate = drop_rate
        else:
            self.drop_rate = 0.0

        # if self.sep_process:
        #     self.relative_position = nn.Parameter(torch.zeros(int(self.top + self.down)))
        # else:
        #     self.relative_position = nn.Parameter(torch.zeros(self.down))

        self.qkv = nn.Linear(in_feats, in_feats * 3, bias=qkv_bias)
        self.proj = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.Dropout(p=self.drop_rate, inplace=True)
            )
        self.residual = nn.Linear(in_feats, out_feats)
        
        mlp_ratio = 2.00
        self.mlp = nn.Sequential(
            nn.Linear(out_feats, int(out_feats * mlp_ratio)),
            nn.Dropout(p=self.drop_rate, inplace=True),
            nn.Linear(int(out_feats * mlp_ratio), out_feats),
            nn.Dropout(p=self.drop_rate, inplace=True)
        )
        # self.norm = nn.LayerNorm(normalized_shape=[out_feats, self.cnt_matrix.shape[0]])
        self.norm = nn.BatchNorm1d(out_feats, momentum=0.15, affine=True, track_running_stats=False)
        # self.norm = nn.InstanceNorm1d(out_feats, momentum=0.15, affine=True)
        # trunc_normal_(self.relative_position, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = torch.Tensor.permute(x, (0, 2, 1))
        res = self.residual(x)

        B, N, C = x.shape  # batch size x number of vertices x channel
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        # relative_position = self.relative_position[:]
        if self.sep_process:
            attn_5adj = torch.einsum("binjc,binkc->binjk", q[:, :, self.neigh_orders_top], k[:, :, self.neigh_orders_top])
            # attn_5adj = attn_5adj + relative_position[:self.top]
            attn_5adj = self.softmax(attn_5adj)

            v_5adj = v[:, :, self.neigh_orders_top]
            x_5adj = torch.einsum("binjk,binkc->binjc", attn_5adj, v_5adj)

            attn_6adj = torch.einsum("binjc,binkc->binjk", q[:, :, self.neigh_orders_down], k[:, :, self.neigh_orders_down])
            # attn_6adj = attn_6adj + relative_position[self.top:]
            attn_6adj = self.softmax(attn_6adj)

            v_6adj = v[:, :, self.neigh_orders_down]
            x_6adj = torch.einsum("binjk,binkc->binjc", attn_6adj, v_6adj)

            x = torch.cat((x_5adj.reshape(B, self.num_heads, -1, C // self.num_heads), x_6adj.reshape(B, self.num_heads, -1, C // self.num_heads)), dim=2)
            x = self.padding(x)
            x = x[:, :, self.reverse_matrix, :].permute((0, 1, 3, 4, 2)) * self.cnt_matrix
            x = torch.Tensor.sum(x.permute((0, 1, 4, 2, 3)), dim=3)

        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)

        out_features = self.proj(x) + res
        res2 = self.mlp(out_features)
        out_features = torch.Tensor.permute(out_features, (0, 2, 1))
        res2 = torch.Tensor.permute(res2, (0, 2, 1))
        out_features = out_features + self.norm(res2)

        return out_features


class pool_layer(nn.Module):
    """
    The pooling layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        N x D tensor
    Return:
        ((N+6)/4) x D tensor
    
    """  

    def __init__(self, neigh_orders, pooling_type='mean'):
        super(pool_layer, self).__init__()

        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type
        
    def forward(self, x):
        
        num_nodes = int((x.size()[0]+6)/4)
        feat_num = x.size()[1]
        x = x[self.neigh_orders[0:num_nodes*7]].view(num_nodes, 7, feat_num)
        if self.pooling_type == "mean":
            x = torch.mean(x, 1)
        if self.pooling_type == "max":
            x = torch.max(x, 1)
            assert(x[0].size() == torch.Size([num_nodes, feat_num]))
            return x[0], x[1]
        
        # assert x.size() == torch.Size([num_nodes, feat_num]), "assertion error"
                
        return x


class pool_layer_batch(nn.Module):
    """
    The pooling layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        B x N x D tensor
    Return:
        B x ((N+6)/4) x D tensor
    
    """  

    def __init__(self, neigh_orders, pooling_type='mean'):
        super(pool_layer_batch, self).__init__()

        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type
        
    def forward(self, x):
        # x = torch.Tensor.permute(x, (0, 2, 1))
        # batch_num, num_nodes, feat_num = x.shape
        # num_nodes = int((x.size()[1]+6)/4)
        # feat_num = x.size()[2]
        # x = x[:, self.neigh_orders[0:num_nodes*7]].view(batch_num, num_nodes, feat_num, 7)
        # if self.pooling_type == "mean":
        #     x = torch.mean(x, 3)
        # if self.pooling_type == "max":
        #     x = torch.max(x, 3)
        #     assert(x[0].size() == torch.Size([batch_num, num_nodes, feat_num]))
        #     x = x[0]
        
        # assert(x.size() == torch.Size([batch_num, num_nodes, feat_num]))
        # x = torch.Tensor.permute(x, (0, 2, 1))

        batch_num, feat_num, num_nodes = x.shape
        num_nodes = int((x.size()[2]+6)/4)
        feat_num = x.size()[1]
        x = x[:, :, self.neigh_orders[0:num_nodes*7]].view(batch_num, feat_num, num_nodes, 7)
        if self.pooling_type == "mean":
            x = torch.mean(x, 3)
        if self.pooling_type == "max":
            x = torch.max(x, 3)
            assert(x[0].size() == torch.Size([batch_num, feat_num, num_nodes]))
            x = x[0]
        
        assert(x.size() == torch.Size([batch_num, feat_num, num_nodes]))
        return x

        
class upconv_layer(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x out_feats, tensor
    
    """  

    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        self.weight = nn.Linear(in_feats, 7 * out_feats)
        
    def forward(self, x):
       
        raw_nodes = x.size()[0]
        new_nodes = int(raw_nodes*4 - 6)
        x = self.weight(x)
        x = x.view(len(x) * 7, self.out_feats)
        x1 = x[self.upconv_top_index]
        assert(x1.size() == torch.Size([raw_nodes, self.out_feats]))
        x2 = x[self.upconv_down_index].view(-1, self.out_feats, 2)
        x = torch.cat((x1,torch.mean(x2, 2)), 0)
        assert(x.size() == torch.Size([new_nodes, self.out_feats]))
        return x


class upconv_layer_batch(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x out_feats, tensor
    
    """  

    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer_batch, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        # self.weight = nn.Linear(in_feats, 7 * out_feats)
        self.weight = nn.Conv1d(in_feats, 7 * out_feats, kernel_size=1)
        # self.norm = nn.BatchNorm1d(out_feats, momentum=0.15, affine=False, track_running_stats=True)
        # self.norm = nn.InstanceNorm1d(out_feats, momentum=0.15)
        
    def forward(self, x):
        # raw_nodes = x.size()[1]
        # new_nodes = int(raw_nodes*4 - 6)
        # x = self.weight(x)
        # x = x.view(x.shape[0], raw_nodes * 7, self.out_feats)
        # x1 = x[:, self.upconv_top_index]
        # assert(x1.size() == torch.Size([x.shape[0], raw_nodes, self.out_feats]))
        # x2 = x[:, self.upconv_down_index].view(x.shape[0], -1, self.out_feats, 2)
        # x = torch.cat((x1,torch.mean(x2, 3)), 1)
        # assert(x.size() == torch.Size([x.shape[0],new_nodes, self.out_feats]))

        raw_nodes = x.size()[2]
        new_nodes = int(raw_nodes*4 - 6)
        x = self.weight(x)
        x = x.view(x.shape[0], self.out_feats, raw_nodes * 7)
        x1 = x[:, :, self.upconv_top_index]
        assert(x1.size() == torch.Size([x.shape[0], self.out_feats, raw_nodes]))
        x2 = x[:, :, self.upconv_down_index].view(x.shape[0], self.out_feats, -1, 2)
        x = torch.cat((x1,torch.mean(x2, 3)), 2)
        assert(x.size() == torch.Size([x.shape[0], self.out_feats, new_nodes]))
        # x = self.norm(x)
        return x


class upconv_layer_batch_average(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter
    
    Input: 
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x out_feats, tensor
    
    """  

    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer_batch_average, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        # self.weight = nn.Linear(in_feats, 7 * out_feats)
        # self.norm = nn.BatchNorm1d(out_feats, momentum=0.15, affine=False, track_running_stats=True)
        # self.norm = nn.InstanceNorm1d(out_feats, momentum=0.15)
        
    def forward(self, x):

        raw_nodes = x.size()[2]
        new_nodes = int(raw_nodes*4 - 6)
        # x = self.weight(x)
        x = torch.Tensor.unsqueeze(x, dim=2).repeat((1, 1, 7, 1)).permute((0, 1, 3, 2))
        # x = torch.Tensor.repeat(x, (1, 7, 1))
        x = x.reshape(x.shape[0], self.out_feats, raw_nodes * 7)
        x1 = x[:, :, self.upconv_top_index]
        assert(x1.size() == torch.Size([x.shape[0], self.out_feats, raw_nodes]))
        x2 = x[:, :, self.upconv_down_index].view(x.shape[0], self.out_feats, -1, 2)
        x = torch.cat((x1,torch.mean(x2, 3)), 2)
        assert(x.size() == torch.Size([x.shape[0], self.out_feats, new_nodes]))
        # x = self.norm(x)
        return x




# class self_attention_layer_swin_4order(nn.Module):
#     """The self-attention layer on icosahedron discretized sphere based on
#     4-ring filter
    
#     Parameters:
#             in_feats (int) - - input features/channels
#             out_feats (int) - - output features/channels
#             num_heads (int) - - Number of attention heads
#             qkv_bias （bool） - - If True, add a learnable bias to query, key, value. Default: True
#             qk_scale (float) - - Override default qk scale of head_dim ** -0.5 if set
#     Input: 
#         B x N x in_feats tensor
#     Return:
#         B x N x out_feats tensor
#     """  
#     def __init__(self, in_feats, out_feats, neigh_orders, num_heads=4, neigh_indices=None, neigh_weights=None, 
#         qkv_bias=True, qk_scale=None, sep_process=True, drop_rate=None):
#         super(self_attention_layer_swin_4order, self).__init__()

#         self.in_feats = in_feats
#         self.out_feats = out_feats

#         # self.top = 16
#         # self.down = 19

#         self.top = 51
#         self.down = 61

#         self.neigh_orders_top = neigh_orders['top'].reshape((-1, self.top))
#         self.neigh_orders_down = neigh_orders['down'].reshape((-1, self.down))
#         self.reverse_matrix = neigh_orders['reverse']
#         self.cnt_matrix = nn.parameter.Parameter(torch.from_numpy((1 / neigh_orders['count']).astype(np.float32)), requires_grad=False)

#         self.padding = nn.ZeroPad2d((0, 0, 0, 1))
        
#         self.num_heads = num_heads
#         head_dim = in_feats // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.sep_process = sep_process

#         if drop_rate:
#             self.drop_rate = drop_rate
#         else:
#             self.drop_rate = 0.0

#         if self.sep_process:
#             self.relative_position = nn.Parameter(torch.zeros(int(self.top + self.down)))
#         else:
#             self.relative_position = nn.Parameter(torch.zeros(self.down))

#         self.qkv = nn.Linear(in_feats, in_feats * 3, bias=qkv_bias)
#         self.proj = nn.Sequential(
#             nn.Linear(in_feats, out_feats),
#             nn.Dropout(p=self.drop_rate, inplace=True)
#             )
#         self.res = nn.Linear(in_feats, out_feats)
        
#         mlp_ratio = 2.00
#         self.mlp = nn.Sequential(
#             nn.Linear(out_feats, int(out_feats * mlp_ratio)),
#             nn.Dropout(p=self.drop_rate, inplace=True),
#             nn.Linear(int(out_feats * mlp_ratio), out_feats),
#             nn.Dropout(p=self.drop_rate, inplace=True)
#         )
#         self.norm = nn.BatchNorm1d(out_feats, momentum=0.15, affine=True, track_running_stats=False)
#         # trunc_normal_(self.relative_position, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, x):
#         x = torch.Tensor.permute(x, (0, 2, 1))
#         res = self.res(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         q = q * self.scale
#         relative_position = self.relative_position[:]
#         if self.sep_process:
#             attn_5adj = torch.einsum("binjc,binkc->binjk", q[:, :, self.neigh_orders_top], k[:, :, self.neigh_orders_top])
#             attn_5adj = attn_5adj + relative_position[:self.top]
#             attn_5adj = self.softmax(attn_5adj)

#             v_5adj = v[:, :, self.neigh_orders_top]
#             x_5adj = torch.einsum("binjk,binkc->binjc", attn_5adj, v_5adj)

#             attn_6adj = torch.einsum("binjc,binkc->binjk", q[:, :, self.neigh_orders_down], k[:, :, self.neigh_orders_down])
#             attn_6adj = attn_6adj + relative_position[self.top:]
#             attn_6adj = self.softmax(attn_6adj)

#             v_6adj = v[:, :, self.neigh_orders_down]
#             x_6adj = torch.einsum("binjk,binkc->binjc", attn_6adj, v_6adj)

#             x = torch.cat((x_5adj.reshape(B, self.num_heads, -1, C // self.num_heads), x_6adj.reshape(B, self.num_heads, -1, C // self.num_heads)), dim=2)
#             x = self.padding(x)
#             x = x[:, :, self.reverse_matrix, :].permute((0, 1, 3, 4, 2)) * self.cnt_matrix
#             x = torch.Tensor.sum(x.permute((0, 1, 4, 2, 3)), dim=3)
#         else:
#             attn = torch.einsum("binc,bincj->binj", q[:, :], k[:, :, self.neigh_orders].transpose(-2, -1))
#             attn = attn + relative_position
#             attn = self.softmax(attn)

#             v = v[:, :, self.neigh_orders]
#             x = torch.einsum("binj,binjc->binc", attn, v)
            
#         x = x.permute(0, 2, 1, 3).reshape(B, N, -1)

#         out_features = self.proj(x) + res
#         res2 = self.mlp(out_features)
#         out_features = torch.Tensor.permute(out_features, (0, 2, 1))
#         res2 = torch.Tensor.permute(res2, (0, 2, 1))
#         out_features = out_features + self.norm(res2)

#         return out_features

    
        
# class self_attention_layer(nn.Module):
#     """The self-attention layer on icosahedron discretized sphere based on
#     1-ring filter
    
#     Parameters:
#             in_feats (int) - - input features/channels
#             out_feats (int) - - output features/channels
#             num_heads (int) - - Number of attention heads
#             qkv_bias （bool） - - If True, add a learnable bias to query, key, value. Default: True
#             qk_scale (float) - - Override default qk scale of head_dim ** -0.5 if set
#     Input: 
#         N x in_feats tensor
#     Return:
#         N x out_feats tensor
#     """  
#     def __init__(self, in_feats, out_feats, neigh_orders, num_heads=1, neigh_indices=None, neigh_weights=None, qkv_bias=True, qk_scale=None, sep_process=False):
#         super(self_attention_layer, self).__init__()

#         self.in_feats = in_feats
#         self.out_feats = out_feats
#         neigh_orders = neigh_orders.reshape((-1, 7))
#         self.neigh_orders = np.concatenate((neigh_orders[:, :5], neigh_orders[:, 6:], neigh_orders[:, 5:6]), axis=1)
#         self.num_heads = num_heads
#         head_dim = in_feats // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.sep_process = sep_process

#         if self.sep_process:
#             self.relative_position = nn.Parameter(torch.zeros(13))
#         else:
#             self.relative_position = nn.Parameter(torch.zeros(7))

#         self.qkv = nn.Linear(in_feats, in_feats * 3, bias=qkv_bias)
#         self.proj = nn.Linear(in_feats, out_feats)
        
#         # trunc_normal_(self.relative_position, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#         # self.weight = nn.Linear(7 * in_feats, out_feats)
        
#     def forward(self, x):
#         # import pdb
#         # pdb.set_trace()
#         # mat = x[self.neigh_orders].view(len(x), 7*self.in_feats)
                
#         # out_features = self.weight(mat)

#         N, C = x.shape
#         qkv = self.qkv(x).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         q = q * self.scale
#         relative_position = self.relative_position[:]
#         if self.sep_process:
#             attn_5adj = torch.einsum("nc,ncj->nj", q[0, :12], k[0, self.neigh_orders[:12, :6]].transpose(-2, -1))
#             attn_5adj = attn_5adj + relative_position[:6]
#             attn_5adj = self.softmax(attn_5adj)

#             v_5adj = v[0, self.neigh_orders[:12, :6]]
#             # x_5adj = (attn_5adj @ v_5adj).transpose(1, 2).reshape(12, C)
#             x_5adj = torch.einsum("nj,njc->nc", attn_5adj, v_5adj)

#             # attn_6adj = (q[0, 12:] @ k[0, self.neigh_orders[12:]].transpose(-2, -1))
#             attn_6adj = torch.einsum("nc,ncj->nj", q[0, 12:], k[0, self.neigh_orders[12:]].transpose(-2, -1))
#             attn_6adj = attn_6adj + relative_position[6:]
#             attn_6adj = self.softmax(attn_6adj)

#             v_6adj = v[0, self.neigh_orders[12:]]
#             # x_6adj = (attn_6adj @ v_6adj).transpose(1, 2).reshape(N - 12, C)
#             x_6adj = torch.einsum("nj,njc->nc", attn_6adj, v_6adj)

#             x = torch.cat((x_5adj, x_6adj), dim=0)
#         else:
#             # attn = (q[0] @ k[0, self.neigh_orders].transpose(-2, -1))
#             attn = torch.einsum("nc,ncj->nj", q[0], k[0, self.neigh_orders].transpose(-2, -1))
#             attn = attn + relative_position[6:]
#             attn = self.softmax(attn)

#             v = v[0, self.neigh_orders]
#             # x = (attn @ v).transpose(1, 2).reshape(N, C)
#             x = torch.einsum("nj,njc->nc", attn, v)

#         out_features = self.proj(x)        
#         # import pdb
#         # pdb.set_trace()
#         return out_features


# class self_attention_layer_batch(nn.Module):
#     """The self-attention layer on icosahedron discretized sphere based on
#     1-ring filter
    
#     Parameters:
#             in_feats (int) - - input features/channels
#             out_feats (int) - - output features/channels
#             num_heads (int) - - Number of attention heads
#             qkv_bias （bool） - - If True, add a learnable bias to query, key, value. Default: True
#             qk_scale (float) - - Override default qk scale of head_dim ** -0.5 if set
#     Input: 
#         B x N x in_feats tensor
#     Return:
#         B x N x out_feats tensor
#     """  
#     def __init__(self, in_feats, out_feats, neigh_orders, num_heads=4, neigh_indices=None, neigh_weights=None, 
#         qkv_bias=True, qk_scale=None, sep_process=False, drop_rate=None):
#         super(self_attention_layer_batch, self).__init__()

#         self.in_feats = in_feats
#         self.out_feats = out_feats
#         neigh_orders = neigh_orders.reshape((-1, 7))
#         self.neigh_orders = np.concatenate((neigh_orders[:, :5], neigh_orders[:, 6:], neigh_orders[:, 5:6]), axis=1)
#         self.num_heads = num_heads
#         head_dim = in_feats // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.sep_process = sep_process

#         if drop_rate:
#             self.drop_rate = drop_rate
#         else:
#             self.drop_rate = 0.0

#         self.res = nn.Sequential(
#             nn.Linear(in_feats, out_feats),
#             nn.Dropout(p=self.drop_rate, inplace=True)
#         )

#         if self.sep_process:
#             self.relative_position = nn.Parameter(torch.zeros(13))
#         else:
#             self.relative_position = nn.Parameter(torch.zeros(7))

#         self.qkv = nn.Linear(in_feats, in_feats * 3, bias=qkv_bias)
#         self.proj = nn.Sequential(
#             nn.Linear(in_feats, out_feats),
#             nn.Dropout(p=self.drop_rate, inplace=True)
#             )
        
#         mlp_ratio = 2.00
#         self.mlp = nn.Sequential(
#             nn.Linear(out_feats, int(out_feats * mlp_ratio)),
#             nn.Dropout(p=self.drop_rate, inplace=True),
#             nn.Linear(int(out_feats * mlp_ratio), out_feats),
#             nn.Dropout(p=self.drop_rate, inplace=True)
#         )
#         self.norm = nn.BatchNorm1d(out_feats, momentum=0.15, affine=True, track_running_stats=False)
#         # self.norm = nn.InstanceNorm1d(out_feats, momentum=0.15)
#         # trunc_normal_(self.relative_position, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, x):
#         x = torch.Tensor.permute(x, (0, 2, 1))
#         res = self.res(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         q = q * self.scale
#         relative_position = self.relative_position[:]
#         if self.sep_process:
#             attn_5adj = torch.einsum("binc,bincj->binj", q[:, :, :12], k[:, :, self.neigh_orders[:12, :6]].transpose(-2, -1))
#             attn_5adj = attn_5adj + relative_position[:6]
#             attn_5adj = self.softmax(attn_5adj)

#             v_5adj = v[:, :, self.neigh_orders[:12, :6]]
#             # x_5adj = (attn_5adj @ v_5adj).transpose(1, 2).reshape(12, C)
#             x_5adj = torch.einsum("binj,binjc->binc", attn_5adj, v_5adj)

#             # attn_6adj = (q[0, 12:] @ k[0, self.neigh_orders[12:]].transpose(-2, -1))
#             attn_6adj = torch.einsum("binc,bincj->binj", q[:, :, 12:], k[:, :, self.neigh_orders[12:]].transpose(-2, -1))
#             attn_6adj = attn_6adj + relative_position[6:]
#             attn_6adj = self.softmax(attn_6adj)

#             v_6adj = v[:, :, self.neigh_orders[12:]]
#             # x_6adj = (attn_6adj @ v_6adj).transpose(1, 2).reshape(N - 12, C)
#             x_6adj = torch.einsum("binj,binjc->binc", attn_6adj, v_6adj)

#             x = torch.cat((x_5adj, x_6adj), dim=1)
#         else:
#             # attn = (q[0] @ k[0, self.neigh_orders].transpose(-2, -1))
#             attn = torch.einsum("binc,bincj->binj", q[:, :], k[:, :, self.neigh_orders].transpose(-2, -1))
#             attn = attn + relative_position
#             attn = self.softmax(attn)

#             v = v[:, :, self.neigh_orders]
#             # x = (attn @ v).transpose(1, 2).reshape(N, C)
#             x = torch.einsum("binj,binjc->binc", attn, v)
            
#         x = x.permute(0, 2, 1, 3).reshape(B, N, -1)

#         out_features = self.proj(x) + res
#         res2 = self.mlp(out_features)
#         out_features = torch.Tensor.permute(out_features, (0, 2, 1))
#         res2 = torch.Tensor.permute(res2, (0, 2, 1))
#         out_features = out_features + self.norm(res2)

#         return out_features


class upsample_interpolation(nn.Module):
    """
    The upsampling layer on icosahedron discretized sphere using interpolation
    
    Input: 
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x in_feats, tensor
    
    """  

    def __init__(self, upsample_neighs_order):
        super(upsample_interpolation, self).__init__()

        self.upsample_neighs_order = upsample_neighs_order
       
    def forward(self, x):
       
        num_nodes = x.size()[0] * 4 - 6
        feat_num = x.size()[1]
        x1 = x[self.upsample_neighs_order].view(num_nodes - x.size()[0], feat_num, 2)
        x1 = torch.mean(x1, 2)
        x = torch.cat((x,x1),0)
                    
        return x


class upsample_fixindex(nn.Module):
    """
    The upsampling layer on icosahedron discretized sphere using fixed indices 0,
    padding new vertices with 0
    
    Input: 
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x in_feats, tensor
    
    """  
    def __init__(self, upsample_neighs_order):
        super(upsample_fixindex, self).__init__()

        self.upsample_neighs_order = upsample_neighs_order
       
    def forward(self, x):
       
        num_nodes = x.size()[0] * 4 - 6
        feat_num = x.size()[1]
        x1 = torch.zeros(num_nodes - x.size()[0], feat_num).cuda()
        x = torch.cat((x,x1),0)
                    
        return x
    
      
class upsample_maxindex(nn.Module):
    """
    The upsampling layer on icosahedron discretized sphere using max indices.
    
    Input: 
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x in_feats, tensor
    
    """  

    def __init__(self, num_nodes, neigh_orders):
        super(upsample_maxindex, self).__init__()

        self.num_nodes = num_nodes
        self.neigh_orders = neigh_orders
        
    def forward(self, x, max_index):
       
        raw_nodes, feat_num = x.size()
        assert(max_index.size() == x.size())
        x = x.view(-1)        
        
        y = torch.zeros(self.num_nodes, feat_num).to(torch.device("cuda"))
        column_ref = torch.zeros(raw_nodes, feat_num)
        for i in range(raw_nodes):
            column_ref[i,:] = i * 7 + max_index[i,:] 
        column_index = self.neigh_orders[column_ref.view(-1).long()]
        column_index = torch.from_numpy(column_index).long()
        row_index = np.floor(np.linspace(0.0, float(feat_num), num=raw_nodes*feat_num))
        row_index[-1] = row_index[-1] - 1
        row_index = torch.from_numpy(row_index).long()
        y[column_index, row_index] = x
        
        return y


 