import torch
import torch.nn as nn
import math
import pdb
import copy
from numpy import inf
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from decoder import (
    Decoder,
    DecoderCBatchNorm,
    DecoderCBatchNorm2,
    DecoderCBatchNormNoResnet,
    DecoderBatchNorm
)
from nerf import Nerf, NerfTransform
from scipy.io import loadmat
from autoencoder_dataset import Cartesian2Spherical


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


class PaiAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, num_neighbors, x_neighbors, D, U, activation='elu'):
        super(PaiAutoencoder, self).__init__()
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


        self.name = ['head', 'left_eye', 'right_eye']
        self.dictionary = torch.load('./mesh_head/head_index.tch')
        self.map_coords = NerfTransform(2, 10)

        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = activation
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(num_neighbors)-1):
            if filters_enc[1][i]:
                self.conv.append(PaiConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[1][i],
                                            activation=self.activation))
                input_size = filters_enc[1][i]

            self.conv.append(PaiConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[0][i+1],
                                        activation=self.activation))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)

        self.decoder = [] 
        self.map = dict() 

        for name in self.name:
            D = 6 if name == 'head' else 3
            W = 128 if name == 'head' else 64
            self.decoder.append(Nerf(D=D, W=W, input_ch=40+latent_size, skips=[D//2], output_ch=3))    
            self.map[name] = loadmat('./mesh_head/{}_map.mat'.format(name))['map'][self.dictionary['{}_index_From_Map'.format(name)]]
            self.map[name] = self.map_coords(torch.from_numpy(Cartesian2Spherical(self.map[name])).float()).cuda()
        self.decoder = nn.ModuleList(self.decoder)   

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
        
        j = 0
        for i in range(len(self.num_neighbors)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            #x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i])
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.shape[0]
        verts = torch.empty([bsize, self.sizes[0], 3]).to(z)
        for i, name in enumerate(self.name):
            map = self.map[name][None].repeat(bsize, 1, 1)
            z_sub = z[:, None].repeat(1, map.shape[1], 1)
            f_vertice = self.decoder[i](torch.cat([z_sub, map], dim=-1))
            verts[:, self.dictionary['{}_index_From_Template'.format(name)]] = f_vertice
        return verts

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x


class PaiAutoNerf(nn.Module):
    def __init__(self, latent_size, is_body=False, activation='elu'):
        super(PaiAutoNerf, self).__init__()
        self.latent_size = latent_size
        
        self.name = ['head', 'left_eye', 'right_eye']
        self.dictionary = torch.load('./mesh_head/head_index.tch')
        self.map_coords = NerfTransform(2, 10)
        self.f1 = nn.Linear(3, 40)
        self.f2 = nn.Linear(512+64*2, latent_size)

        self.encoder = []
        self.decoder = [] 
        self.map = dict() 
        
        self.size = 5023
        for name in self.name:
            D = 8 if name == 'head' else 3
            n_dim = 512 if name == 'head' else 64
            self.encoder.append(Nerf(D=D, W=128, input_ch=40*2, skips=[D//2], output_ch=n_dim))
            self.decoder.append(Nerf(D=D, W=128+16, input_ch=40+latent_size, skips=[D//2], output_ch=3))    
            self.map[name] = loadmat('./mesh_head/{}_map.mat'.format(name))['map'][self.dictionary['{}_index_From_Map'.format(name)]]
            self.map[name] = self.map_coords(torch.from_numpy(Cartesian2Spherical(self.map[name])).float()).cuda()
        self.encoder = nn.ModuleList(self.encoder)   
        self.decoder = nn.ModuleList(self.decoder)   

    def encode(self,x):
        bsize = x.shape[0]
        verts = self.f1(x)
        verts_part = []
        for i, name in enumerate(self.name):
            map = self.map[name][None].repeat(bsize, 1, 1)
            vert_sub = verts[:, self.dictionary['{}_index_From_Template'.format(name)]]
            f_vertice = self.encoder[i](torch.cat([vert_sub, map], dim=-1))
            f_vertice = F.relu(f_vertice)
            # verts[:, self.dictionary['{}_index_From_Template'.format(name)]] = f_vertice
            verts_part.append(torch.max(f_vertice, dim=1)[0])
        verts_part = torch.cat(verts_part, dim=-1)
        z = self.f2(verts_part)
        return z
    
    def decode(self,z):
        bsize = z.shape[0]
        verts = torch.empty([bsize, self.size, 3]).to(z)
        for i, name in enumerate(self.name):
            map = self.map[name][None].repeat(bsize, 1, 1)
            z_sub = z[:, None].repeat(1, map.shape[1], 1)
            f_vertice = self.decoder[i](torch.cat([z_sub, map], dim=-1))
            verts[:, self.dictionary['{}_index_From_Template'.format(name)]] = f_vertice
        return verts

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x


class PaiNerf(nn.Module):
    def __init__(self, latent_size):
        super(PaiNerf, self).__init__()
        self.latent_size = latent_size
        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = nn.ELU()
        
        self.map_coords = NerfTransform(2, 10)
        self.f1 = nn.Linear(3, 40)

        self.name = ['head', 'left_eye', 'right_eye']
        self.dictionary = torch.load('./mesh_head/head_index.tch')

        self.encoder = []
        self.decoder = [] 
        self.map = dict() 
        self.part_idx = []
        N_index_part = 0
        for name in self.name:
            DEn = 4 if name == 'head' else 3
            DDe = 8 if name == 'head' else 6
            W = 256 if name == 'head' else 64
            self.encoder.append(Nerf(D=DEn, W=W, input_ch=40*2, skips=[DEn//2], output_ch=latent_size))
            self.decoder.append(Nerf(D=DDe, W=W, input_ch=40*4+latent_size+3, skips=[DDe//2], output_ch=3))    
            self.map[name] = loadmat('./mesh_head/{}_map.mat'.format(name))['map'][self.dictionary['{}_index_From_Map'.format(name)]]
            self.map[name] = self.map_coords(torch.from_numpy(Cartesian2Spherical(self.map[name])).float()).cuda()
            self.part_idx.append(N_index_part)
            N_index_part = N_index_part + len(self.dictionary['{}_index_From_Template'.format(name)])
        self.part_idx.append(N_index_part)
        self.part_idx = torch.tensor(self.part_idx)
        self.encoder = nn.ModuleList(self.encoder)   
        self.decoder = nn.ModuleList(self.decoder)   

    ### random every batch ###
    def forward(self, verts_init, coords, bcoords, trilist, first_idx):
        bsize = verts_init.size(0)
        num_pts = trilist.shape[0]

        coords = self.map_coords(coords)[None].repeat(bsize, 1, 1)
        verts_init = self.f1(verts_init)
        verts = []
        for i, name in enumerate(self.name):
            coord = coords[:, first_idx[i]:first_idx[i+1]]
            verts_in = verts_init[:, first_idx[i]:first_idx[i+1]]
            verts.append(self.encoder[i](torch.cat([coord, verts_in], dim=-1)))
        verts = torch.cat(verts, dim=1) 

        t_coords = coords[:, trilist]
        t = verts[:, trilist]
        f_vertices = torch.sum(t * bcoords[None, :, :, None], axis=2)
        f_vertices = torch.cat([f_vertices, bcoords[None].repeat(bsize, 1, 1)], dim=-1)

        verts = torch.empty([bsize, num_pts, 3]).to(verts_init)
        for i, name in enumerate(self.name):
            f_vertice = f_vertices[:, self.part_idx[i]:self.part_idx[i+1]]
            t_coord = t_coords[:, self.part_idx[i]:self.part_idx[i+1]]
            map = self.map[name][None].repeat(bsize, 1, 1)
            t_coord = (t_coord - map[:, :, None]).view(bsize, map.shape[1], -1)
            f_vertice = self.decoder[i](torch.cat([f_vertice, map, t_coord], dim=-1))
            verts[:, self.dictionary['{}_index_From_Template'.format(name)]] = f_vertice
        return verts

    def forward_test(self, verts_init, coords, bcoords, trilist, first_idx):
        bsize = verts_init.size(0)
        t = verts_init[:, trilist]
        verts_init = torch.sum(t * bcoords[None, :, :, None], axis=2)
        verts = torch.empty([bsize, trilist.shape[0], 3]).to(verts_init)
        for i, name in enumerate(self.name):
            verts[:, self.dictionary['{}_index_From_Template'.format(name)]] = verts_init[:, self.part_idx[i]:self.part_idx[i+1]]
        return verts

    # ### random every sample ###
    # def forward(self, coords, verts_init, bcoords, trilist, first_idx):
    #     bsize = coords.size(0)

    #     coords_close = []
    #     coords_left = []
    #     coords_right = []
    #     verts_init_close = []
    #     verts_init_left = []
    #     verts_init_right = []
    #     coords = self.map_coords(coords)
    #     verts_init = self.f1(verts_init)
    #     for b in range(bsize):
    #         coords_close.append(coords[b, :first_idx[b, 0]])
    #         coords_left.append(coords[b, first_idx[b, 0]:first_idx[b, 0]+first_idx[b, 1]])
    #         coords_right.append(coords[b, first_idx[b, 0]+first_idx[b, 1]:])
    #         verts_init_close.append(verts_init[b, :first_idx[b, 0]])
    #         verts_init_left.append(verts_init[b, first_idx[b, 0]:first_idx[b, 0]+first_idx[b, 1]])
    #         verts_init_right.append(verts_init[b, first_idx[b, 0]+first_idx[b, 1]:])
    #     coords_close = torch.cat(coords_close)
    #     coords_left = torch.cat(coords_left)
    #     coords_right = torch.cat(coords_right)
    #     verts_init_close = torch.cat(verts_init_close)
    #     verts_init_left = torch.cat(verts_init_left)
    #     verts_init_right = torch.cat(verts_init_right)

    #     verts_close = self.encoder_close(torch.cat([coords_close, verts_init_close], dim=-1))
    #     verts_left = self.encoder_close(torch.cat([coords_left, verts_init_left], dim=-1))
    #     verts_right = self.encoder_close(torch.cat([coords_right, verts_init_right], dim=-1))

    #     n_sample = verts_init.shape[1]
    #     verts = torch.empty([bsize, n_sample, verts_close.shape[-1]]).to(verts_init)
    #     firstc = firstl = firstr = 0
        
    #     f_vertices = []
    #     for b in range(bsize):
    #         verts[b] = torch.cat([verts_close[firstc:firstc+first_idx[b, 0]], \
    #                 verts_left[firstl:firstl+first_idx[b, 1]], \
    #                 verts_right[firstr:firstr+n_sample-(first_idx[b, 0]+first_idx[b, 1])]])
    #         firstc = firstc+first_idx[b, 0]
    #         firstl = firstl+first_idx[b, 1]
    #         firstr = firstr+n_sample-(first_idx[b, 0]+first_idx[b, 1])

    #         t = verts[b][trilist[b]]
    #         f_vertices.append(torch.sum(t * bcoords[b, :, :, None], axis=1))

    #     f_vertices = torch.stack(f_vertices)
    #     f_vertices_close = f_vertices[:, self.index_Close_From_Template]
    #     f_vertices_left = f_vertices[:, self.index_left]
    #     f_vertices_right = f_vertices[:, self.index_right]
        
    #     map_close = self.map_close[None].repeat(bsize, 1, 1)
    #     map_left = self.map_left[None].repeat(bsize, 1, 1)
    #     map_right = self.map_right[None].repeat(bsize, 1, 1)
    #     f_vertices_close = self.decoder_close(torch.cat([f_vertices_close, map_close], dim=-1))
    #     f_vertices_left = self.decoder_left(torch.cat([f_vertices_left, map_left], dim=-1))
    #     f_vertices_right = self.decoder_right(torch.cat([f_vertices_right, map_right], dim=-1))

    #     verts = torch.empty([bsize, trilist.shape[0], 3]).to(verts_init)
    #     verts[:, self.index_Close_From_Template] = f_vertices_close
    #     verts[:, self.index_left] = f_vertices_left
    #     verts[:, self.index_right] = f_vertices_right

    #     return verts


class PaiImplicitResNet(nn.Module):
    def __init__(self, shape_mean, filters_enc, filters_dec, latent_size, sizes, 
                uv_coords, num_neighbors, x_neighbors, D, activation = 'elu'):
        super(PaiImplicitResNet, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.x_neighbors = [torch.cat([torch.cat([torch.arange(x.shape[0]-1), torch.tensor([-1])]).unsqueeze(1), x], 1) for x in x_neighbors]
        #self.x_neighbors = [x.float().cuda() for x in x_neighbors]
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.num_neighbors = num_neighbors
        self.D = [nn.Parameter(x, False) for x in D]
        self.D = nn.ParameterList(self.D)
        self.fourier_encode = True
        
        if self.fourier_encode:
            self.uv_coords = NerfTransform(2, 10)(uv_coords)
            # scale = 1
            # self.B = nn.Parameter(torch.randn(3, mappingsize) , requires_grad=False)
            # mean_project = (2.*math.pi*shape_mean) @ self.B * scale
            # self.mean = torch.cat([torch.sin(mean_project), torch.cos(mean_project)], dim=-1)
            self.posen = nn.Linear(40, 256)
        else: 
            self.mean = shape_mean
            self.posen = nn.Linear(3, latent_size)

        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(num_neighbors)-1):
            if filters_enc[1][i]:
                self.conv.append(PaiConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[1][i],
                                            activation=activation))
                input_size = filters_enc[1][i]

            self.conv.append(PaiConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[0][i+1],
                                        activation=activation))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)

        self.decoder = DecoderCBatchNormNoResnet(dim=256, z_dim=0, c_dim=latent_size, hidden_size=256, activation='sin')

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
        
        j = 0
        for i in range(len(self.num_neighbors)-1):
            x = self.conv[j](x, None, S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,None, S[i].repeat(bsize,1,1))
                j+=1
            #x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i])
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        x = self.posen(self.uv_coords.to(z))[None, :, :].repeat(z.shape[0], 1, 1)
        x = F.relu(x)
        x = self.decoder(x, 0, z)
        return x

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x