from torch.utils.data import Dataset
import torch
import numpy as np
import os
from scipy.io import loadmat
from scipy.spatial import ConvexHull
import trimesh
from pymeshfix import _meshfix
from laplacian import ComLap

def barycentric_points_from_contained_points(vertics, trilist, points, tri_index):
    # http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    abc_per_tri = vertics[trilist[tri_index]]

    a = abc_per_tri[:, 0, :]
    b = abc_per_tri[:, 1, :]
    c = abc_per_tri[:, 2, :]
    p = points

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = (v0 * v0).sum(axis=1)
    d01 = (v0 * v1).sum(axis=1)
    d11 = (v1 * v1).sum(axis=1)
    d20 = (v2 * v0).sum(axis=1)
    d21 = (v2 * v1).sum(axis=1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return np.vstack([u, v, w]).T


def interpolation(map_vertices, mask_sub, index_part):
    r"""Interpolate some per-vertex value on this mesh using barycentric coordinates.

    Parameters
    ----------
    map_vertices : ``(n_points, 3)`` `ndarray`
        Any array of per-vertex data. This is spherical paramerization of a mesh.
    sub_f_vertics: ``(n_samples, k)`` `ndarray`
        The barycentric coordinates that will be used in the projection
    mask_sub: ``(n_points_all)`` `bool`
        mask to randomly subsample a mesh
    index_part: ``(n_points_part)`` `long`
        index to select a part of a mesh
    Returns
    -------
    `ndarray` : ``(n_points, k)``
        The interpolated values of ``per_vertex_interpolant``.
    """
    mask_part = torch.zeros(len(mask_sub)).bool()
    mask_part[index_part] = True
    mask_sub_part = (mask_part & mask_sub)
    index_overlap = torch.nonzero(mask_sub_part[index_part]).squeeze(-1)
    sub_map_vertices = map_vertices[index_overlap]

    hull = ConvexHull(sub_map_vertices) # SphericalVoronoi(sub_map_vertices)
    sub_map_tri = hull.simplices
    vclean, fclean = _meshfix.clean_from_arrays(sub_map_vertices, sub_map_tri)
    # mesh_sub = Mesh(v=vclean, f=fclean)
    # mesh_sub.write_obj('./mesh_head/head_sub.obj')
    tmesh = trimesh.Trimesh(vertices=vclean, faces=fclean)
    p0, _, indx = trimesh.proximity.closest_point(tmesh, map_vertices)
    bcoords = barycentric_points_from_contained_points(vclean, fclean, p0, indx)
    trilist = fclean[indx]
    # t = sub_f_vertics[trilist]
    # f_vertics = torch.sum(t * bcoords[..., None], axis=1)
    return bcoords, trilist, index_overlap

def Cartesian2Spherical(xyz):
    xy = xyz[:,0]**2 + xyz[:,1]**2
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) / np.pi # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:,1], xyz[:,0]) / (2*np.pi) + 0.5
    return np.stack([theta,phi], axis=1)

class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, is_body=False, normalization=True, dummy_node=True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))
              
        self.is_body = is_body
        if self.is_body:
            self.name = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 
                    'right_leg', 'left_hand', 'right_hand', 'left_foot', 'right_foot']
            folder = 'mesh_body'
            self.dictionary = torch.load('./{}/body_index.tch'.format(folder))
            self.nsample = 1000
        else:
            self.name = ['head', 'left_eye', 'right_eye']
            folder = 'mesh_head'
            self.dictionary = torch.load('./{}/head_index.tch'.format(folder))
            self.nsample = 1000

        self.map = {}
        self.map_coord = {}
        self.lap = ComLap(shapedata.reference_mesh)
        for name in self.name:
            self.map[name] = loadmat('./{}/{}_map.mat'.format(folder, name))['map'][self.dictionary['{}_index_From_Map'.format(name)]]
            self.map_coord[name] = Cartesian2Spherical(self.map[name])

    def random_submesh(self):
        index_sub = torch.randperm(6890)[:self.nsample]
        mask_sub = torch.zeros(6890).bool()
        mask_sub[index_sub] = True

        N_index = 0
        index_subs = dict()
        bcoords = []
        trilists = []
        first_idx = []
        coords = []
        for name in self.name:
            bcoord, trilist, index_sub = interpolation(self.map[name], mask_sub, 
                            self.dictionary['{}_index_From_Template'.format(name)])
            trilist = trilist + N_index
            trilists.append(trilist)
            bcoords.append(bcoord)
            first_idx.append(N_index)
            N_index = N_index + len(index_sub)
            coords.append(self.map_coord[name][index_sub])
            index_subs[name] = self.dictionary['{}_index_From_Template'.format(name)][index_sub]
        first_idx.append(N_index)

        first_idx = torch.tensor(first_idx)
        bcoords = torch.from_numpy(np.concatenate(bcoords)).float()
        trilists = torch.from_numpy(np.concatenate(trilists)).long()        
        coords = torch.from_numpy(np.concatenate(coords)).float()

        return coords, bcoords, trilists, first_idx, index_subs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        verts_init = torch.load(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.tch'))
        verts_init = verts_init
        verts = (verts_init - self.shapedata.mean) / self.shapedata.std
        return verts 
    
  
class autoencoder_dataset_generate(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, is_body=False, normalization=True, dummy_node=True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        
        verts_init = torch.load(os.path.join(self.root_dir,'points'+'_'+self.points_dataset, basename+'.tch'))
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init/self.shapedata.std
        verts_init[np.where(np.isnan(verts_init))]=0.0

        if self.dummy_node:
            verts = torch.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=torch.float32)
            verts[:-1,:] = verts_init
            verts_init = verts
        else:
            verts = verts_init
     
        sample = {'points': verts}

        return sample
    