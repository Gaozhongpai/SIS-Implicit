from scipy.io import savemat, loadmat
import trimesh
import numpy as np 
from numpy import linalg as LA
from psbody.mesh import Mesh
from pymeshfix import _meshfix
from scipy.spatial import ConvexHull
from plyfile import PlyData, PlyElement
import torch
import cv2
from pykeops.torch import generic_argkmin


def knn3(K=1):
    knn = generic_argkmin(
        'SqDist(x, y)',
        'a = Vi({})'.format(K),
        'x = Vi({})'.format(3),
        'y = Vj({})'.format(3),
    )
    return knn

diction = torch.load('./mesh_head/head_index.tch')
mesh_C = Mesh(filename='./mesh_head/head_close_high2.obj')
mesh_L = Mesh(filename='./mesh_head/head_left_high2.obj')
mesh_R = Mesh(filename='./mesh_head/head_right_high2.obj')
diction['head_index_From_Template'] = torch.tensor(range(len(mesh_C.v)))
diction['head_index_From_Map'] = torch.tensor(range(len(mesh_C.v)))
diction['left_eye_index_From_Template'] = torch.tensor(range(len(mesh_C.v), len(mesh_C.v)+len(mesh_L.v)))
diction['left_eye_index_From_Map'] = torch.tensor(range(len(mesh_L.v)))
diction['right_eye_index_From_Template'] = torch.tensor(range(len(mesh_C.v)+len(mesh_L.v), 
                                            len(mesh_C.v)+len(mesh_L.v)+len(mesh_R.v)))
diction['right_eye_index_From_Map'] = torch.tensor(range(len(mesh_R.v)))

verts = np.concatenate([mesh_C.v, mesh_L.v, mesh_R.v])
faces = np.concatenate([mesh_C.f, mesh_L.f + len(mesh_C.v), 
                        mesh_R.f + len(mesh_C.v) + len(mesh_L.v)])
mesh = Mesh(v=verts, f=faces)
mesh.write_obj('./mesh_head/head_high2.obj')

mesh1 = Mesh(filename='./mesh_head/head_high2.obj')
vertex1 = torch.from_numpy(mesh1.v).float()
mesh2 = Mesh(filename='./mesh_head/head.obj')
vertex2 = torch.from_numpy(mesh2.v).float()

index = knn3(K=1)(vertex2, vertex1)[:, 0]
index_overlap = torch.nonzero(torch.sum(vertex2-vertex1[index], dim=1)==0.)[:, 0]
diction['index_From_High'] = index
torch.save(diction, './mesh_head/head_index_high2.tch')


