import numpy as np
import torch
import scipy.sparse as sp
import scipy


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def move_i_first(index, i):
    if (index == i).nonzero()[..., 0].shape[0]:
        inx = (index == i).nonzero()[0][0]
    else:
        index[-1] = i
        inx = (index == i).nonzero()[0][0]
    if inx > 1:
        index[1:inx+1], index[0] = index[0:inx].clone(), index[inx].clone() 
    else:
        index[inx], index[0] = index[0].clone(), index[inx].clone() 
    return index

def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L

def sparse_mx_to_torch_sparse_tensor(sparse_mx, is_L=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sp.csr_matrix(sparse_mx)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    if is_L:
        sparse_mx = rescale_L(sparse_mx, lmax=2)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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


def get_adj(A):
    
    kernal_size = 9
    A_temp = []
    for x in A:
        x.data = np.ones(x.data.shape)
        # build symmetric adjacency matrix
        x = x + x.T.multiply(x.T > x) - x.multiply(x.T > x)
        #x = x + sp.eye(x.shape[0])
        A_temp.append(x.astype('float32'))
    A_temp = [normalize(x) for x in A_temp]
    A = [sparse_mx_to_torch_sparse_tensor(x) for x in A_temp]

    Adj = []
    for adj in A:
        index_list = []
        for i in range(adj.shape[0]): #
            index = (adj._indices()[0] == i).nonzero().squeeze()
            if index.dim() == 0:
                index = index.unsqueeze(0)
            index1 = torch.index_select(adj._indices()[1], 0, index[:kernal_size-1])
            #index1 = move_i_first(index1, i)
            index_list.append(index1)
        index_list.append(torch.zeros(kernal_size-1, dtype=torch.int64)-1)
        index_list = torch.stack([torch.cat([i, i.new_zeros(
            kernal_size - 1 - i.size(0))-1], 0) for inx, i in enumerate(index_list)], 0)
        Adj.append(index_list)
    return Adj


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()