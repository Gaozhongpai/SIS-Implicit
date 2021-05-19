
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


mesh = trimesh.load('./mesh_head/head_right_high1.obj', process = False)
mesh_up = Mesh(filename='./mesh_head/head_right_high2.obj')
p0, _, indx = trimesh.proximity.closest_point(mesh, mesh_up.v)
bcoords = barycentric_points_from_contained_points(mesh.vertices, mesh.faces, p0, indx)

map = Mesh(filename='./mesh_head/head_right_map_high1.obj')
verts = map.v[map.f[indx]]
verts = np.sum(verts * bcoords[:, :, None], axis=1)
verts = verts / LA.norm(verts, axis=1, keepdims=True)
map_up = Mesh(v=verts, f=mesh_up.f)
map_up.write_obj('./mesh_head/head_right_map_high2.obj')
savemat('./mesh_head/right_eye_map_high2.mat', {'v': mesh_up.v, 'f': mesh_up.f+1, 'map': map.v})