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

diction = torch.load('./mesh_head/head_index.tch')
mesh_C = Mesh(filename='./mesh_head/head_close_high1.obj')
mesh_L = Mesh(filename='./mesh_head/head_left_high1.obj')
mesh_R = Mesh(filename='./mesh_head/head_right_high1.obj')
diction['head_index_From_Template'] = torch.tensor(range(len(mesh_C.v)))
diction['head_index_From_Map'] = torch.tensor(range(len(mesh_C.v)))
diction['left_eye_index_From_Template'] = torch.tensor(range(len(mesh_C.v), len(mesh_C.v)+len(mesh_L.v)))
diction['left_eye_index_From_Map'] = torch.tensor(range(len(mesh_L.v)))
diction['right_eye_index_From_Template'] = torch.tensor(range(len(mesh_C.v)+len(mesh_L.v), 
                                            len(mesh_C.v)+len(mesh_L.v)+len(mesh_R.v)))
diction['right_eye_index_From_Map'] = torch.tensor(range(len(mesh_R.v)))
torch.save(diction, './mesh_head/head_index_high1.tch')


verts = np.concatenate([mesh_C.v, mesh_L.v, mesh_R.v])
faces = np.concatenate([mesh_C.f, mesh_L.f + len(mesh_C.v), 
                        mesh_R.f + len(mesh_C.v) + len(mesh_L.v)])
mesh = Mesh(v=verts, f=faces)
mesh.write_obj('./mesh_head/head_high2.obj')