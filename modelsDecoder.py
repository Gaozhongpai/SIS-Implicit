import torch
import torch.nn as nn
import math
import pdb
import copy
from numpy import inf
import torch.nn.functional as F
import math
from graph_utils import laplacian, sparse_mx_to_torch_sparse_tensor
import numpy as np

class PaiConv(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c, activation='elu',bias=True): # ,device=None):
        super(PaiConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        self.adjweight = nn.Parameter(torch.randn(num_pts, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0
        self.mlp_out = nn.Linear(in_c, out_c)
        #self.sparsemax = Sparsemax(dim=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)
        # x_neighbors = x_neighbors.view(num_pts, bsize*feats, num_neighbor)     
        # weight = self.softmax(torch.bmm(torch.transpose(x_neighbors, 1, 2), x_neighbors))
        # x_neighbors = torch.bmm(x_neighbors, weight) #.view(num_pts, feats, num_neighbor)
        x_neighbors = torch.einsum('bnkf, bnkt->bntf', x_neighbors, self.adjweight[None].repeat(bsize, 1, 1, 1))   #self.sparsemax(self.adjweight))
        x_neighbors = self.activation(x_neighbors.contiguous().view(bsize*num_pts, num_neighbor*feats)) 
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        x_res = self.mlp_out(x.view(-1, self.in_c)).view(bsize, -1, self.out_c)
        return out_feat + x_res


class PaiConvSmall(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='elu',bias=True): # ,device=None):
        super(PaiConvSmall,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        num_bases = 8
        self.v = nn.Parameter(torch.ones(num_pts, num_bases) / num_bases, requires_grad=True)
        self.adjweight = nn.Parameter(torch.randn(num_bases, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0
        # self.mlp_out = nn.Linear(in_c, out_c)
        #self.sparsemax = Sparsemax(dim=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)

        adjweight = torch.einsum('ns, skt->nkt', self.v, self.adjweight)[None].repeat(bsize, 1, 1, 1)
        x_neighbors = torch.einsum('bnkf, bnkt->bntf', x_neighbors, adjweight).contiguous()   #self.sparsemax(self.adjweight))
        x_neighbors = self.activation(x_neighbors.view(bsize*num_pts, num_neighbor*feats))
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        # x_res = self.mlp_out(x.view(-1, self.in_c)).view(bsize, -1, self.out_c)
        return out_feat # + x_res


class chebyshevConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self,  num_pts, in_features, kernal_size, out_features, activation='elu', bias=True):
        super(chebyshevConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = kernal_size
        self.weight = nn.Parameter(torch.FloatTensor(in_features * kernal_size, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, L):
        N, M, Fin = x.shape
        # Transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # M x Fin x N
        x0 = x0.view(M, Fin * N)  # M x Fin*N
        x = x0.unsqueeze(0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = x_.unsqueeze(0)  # 1 x M x Fin*N
            return torch.cat((x, x_), 0)  # K x M x Fin*N

        if self.K > 1:
            x1 = torch.spmm(L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * torch.spmm(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = x.view(self.K, M, Fin, N)  # K x M x Fin x N
        x = x.permute(3, 1, 2, 0).contiguous()  # N x M x Fin x K
        x = x.view(N * M, Fin * self.K)  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        # W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = torch.mm(x, self.weight) + self.bias  # N*M x Fout
        return self.activation(x.view(N, M, -1))  # N x M x Fout

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PaiConvSpiral(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c, activation='elu',bias=True): # ,device=None):
        super(PaiConvSpiral,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0
        self.mlp_out = nn.Linear(in_c, out_c)
        #self.sparsemax = Sparsemax(dim=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor*feats)
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return out_feat


class FeaStConv(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='relu',bias=True): # ,device=None):
        super(FeaStConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.heads = num_neighbor
        self.bias = nn.Parameter(torch.Tensor(out_c))
        self.mlp = nn.Linear(in_c, self.heads) 
        self.mlp_out = nn.Linear(in_c, self.heads * out_c, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0

        self.reset_parameters()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()
    
    @staticmethod
    def normal(tensor, mean, std):
        if tensor is not None:
            tensor.data.normal_(mean, std)

    def reset_parameters(self):
        self.normal(self.bias, mean=0, std=0.1)

    def forward(self,x,neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize*num_pts, num_neighbor, feats)
        #### relative position ####
        x_relative = x_neighbors - x_neighbors[:, 0:1, :]

        q = self.softmax(self.mlp(x_relative.view(-1, feats))).view(bsize, num_pts, num_neighbor*self.heads, -1)
        x_j = self.mlp_out(x_neighbors.view(-1, feats)).view(bsize, num_pts, num_neighbor*self.heads, -1)
        out_feat =  (x_j * q).sum(dim=2) + self.bias
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return self.activation(out_feat)


class PaiAutoencoder2(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, num_neighbors, x_neighbors, D, U, A, activation = 'elu'):
        super(PaiAutoencoder2, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.x_neighbors = [torch.cat([torch.cat([torch.arange(x.shape[0]-1), torch.tensor([-1])]).unsqueeze(1), x], 1) for x in x_neighbors]
        #self.x_neighbors = [x.float().cuda() for x in x_neighbors]
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.num_neighbors = num_neighbors
        self.D = [nn.Parameter(x, False) for x in D]
        self.D = nn.ParameterList(self.D)
        self.U = [nn.Parameter(x, False) for x in U]
        self.U = nn.ParameterList(self.U)

        print("Computing Graph Laplacians ..")
        self.A = []
        for x in A:
            x.data = np.ones(x.data.shape)
            # build symmetric adjacency matrix
            x = x + x.T.multiply(x.T > x) - x.multiply(x.T > x)
            #x = x + sp.eye(x.shape[0])
            self.A.append(x.astype('float32'))
        self.L = [laplacian(a, normalized=True) for a in self.A]
        self.L = [nn.Parameter(sparse_mx_to_torch_sparse_tensor(x, is_L=True), False) for x in self.L]
        self.L = nn.ParameterList(self.L)

        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = activation
        self.conv = []
        input_size = filters_enc[0]
        for i in range(len(num_neighbors)-1):
            self.conv.append(PaiConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[i+1],
                                        activation=self.activation))
            input_size = filters_enc[i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0])
        
        self.dconv = []
        input_size = filters_dec[0]
        for i in range(len(num_neighbors)-1):
            self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[i+1],
                                            activation=self.activation))
            input_size = filters_dec[i+1]  

            if i == len(num_neighbors)-2:
                input_size = filters_dec[-2]
                self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[-1],
                                                activation='identity'))
                    
        self.dconv = nn.ModuleList(self.dconv)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        x = x.permute(1, 2, 0).contiguous()  #M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N

        x = torch.spmm(L, x)  # Mp x Fin*N
        x = x.view(Mp, Fin, N)  # Mp x Fin x N
        x = x.permute(2, 0, 1).contiguous()   # N x Mp x Fin
        return x

    def encode(self,x):
        bsize = x.size(0)
        S = self.x_neighbors
        D = self.D
        for i in range(len(self.num_neighbors)-1):
            x = self.conv[i](x, S[i].repeat(bsize,1,1))
            #x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i])
        # x = self.conv[-1](x, t_vertices[-1], S[-1].repeat(bsize,1,1))
        x = x.view(bsize,-1)
        # x = x[:, :-1].view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.x_neighbors
        U = self.U

        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)

        for i in range(len(self.num_neighbors)-1):
            #x = torch.matmul(U[-1-i],x)
            x = self.poolwT(x, U[-1-i])
            x = self.dconv[i](x, S[-2-i].repeat(bsize,1,1))
        x = self.dconv[-1](x, S[0].repeat(bsize,1,1))
        return x
    
    def decodeChev(self,z):
        bsize = z.size(0)
        S = self.x_neighbors
        U = self.U
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        for i in range(len(self.num_neighbors)-1):
            #x = torch.matmul(U[-1-i],x)
            x = self.poolwT(x, U[-1-i])
            x = self.dconv[i](x[:, :-1],self.L[-2-i])
            x = torch.cat([x, torch.zeros(bsize, 1, x.shape[-1]).to(x)], dim=1)
        x = self.dconv[-1](x[:, :-1],self.L[0])
        x = torch.cat([x, torch.zeros(bsize, 1, x.shape[-1]).to(x)], dim=1)
        return x

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x
