from graph_utils import sparse_mx_to_torch_sparse_tensor
import numpy as np
import numbers
import scipy.sparse as sp
import scipy
import torch
from device import device
import trimesh
from torch import nn
from chumpy.utils import row, col

def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def get_laplacian(mesh):
    A = get_vert_connectivity(mesh.vertices, mesh.faces)
    A.data = np.ones(A.data.shape)
    # build symmetric adjacency matrix
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    L = laplacian(A, normalized=False) 
    L = sparse_mx_to_torch_sparse_tensor(L)
    return L


class ComLap(nn.Module):
    def __init__(self, mesh):
        super(ComLap, self).__init__()
        self.point_num = len(mesh.vertices)
        self.L = get_laplacian(mesh).to(device)
        # vertex = torch.from_numpy(mesh.vertices).type(torch.FloatTensor).to(device)
        # self.template_lap = torch.spmm(self.L, vertex)

    def forward(self, x, y):
        bsize = x.shape[0]
        xy = torch.cat([x,y], dim=0)
        xy = xy.permute(1, 2, 0).contiguous().view(self.point_num, -1)
        xy = torch.spmm(self.L, xy)

        xy = xy.view(self.point_num, 3, -1)
        xy = xy.permute(2, 0, 1).contiguous()

        loss = torch.mean(torch.norm(xy[:bsize]-xy[bsize:],dim=2))
        return loss


if __name__ == "__main__":
    template = trimesh.load_mesh("/mnt/sdb/3dfaceRe/data/smplRegistData/template.ply")
    lap = ComLap(template)
    geometry = torch.from_numpy(template.vertices).unsqueeze(0).to(device).float()
    loss = lap(geometry)
    print(loss)
