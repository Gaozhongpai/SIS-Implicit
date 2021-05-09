from scipy.io import savemat, loadmat
import trimesh
import numpy as np 
from psbody.mesh import Mesh
from pymeshfix import _meshfix
from plyfile import PlyData, PlyElement
import torch
import cv2

names = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 
         'right_leg', 'left_hand', 'right_hand', 'left_foot', 'right_foot']

mesh = Mesh(filename='./output/smpl_uv.obj')
###### step 1: generate faces from sub index #######
for name in names:
    indx = np.load('./output/{}_indices.npy'.format(name)).tolist()
    faces = mesh.f
    face_sub = []
    for i in range(len(faces)):
        if faces[i][0] in indx and \
            faces[i][1] in indx and \
            faces[i][2] in indx:
            face_sub.append([indx.index(faces[i][0]), \
                            indx.index(faces[i][1]), \
                            indx.index(faces[i][2])])
        
    face_sub = np.array(face_sub)
    mesh_sub = Mesh(v=mesh.v[indx], f=face_sub)
    mesh_sub.write_obj('./output/{}_v2.obj'.format(name))

###### step 2: close whole and save to mat to do spherical parameterization #######
for name in names:
    mesh = Mesh(filename='./output/{}_v2.obj'.format(name))
    vclean, fclean = _meshfix.clean_from_arrays(mesh.v, mesh.f)
    mesh = Mesh(v=vclean, f=fclean)
    mesh.write_obj('./output/{}_v3.obj'.format(name))
    savemat('./output/{}.mat'.format(name), {'v': vclean, 'f': fclean})
    #### note that 'left leg', 'left foot', and 'right foot' need to adjust vertices 
    # that intersect in Blender. Importanly, blender will change vertices order by default
    # both import and export need to keep the vertex order 
    # we set the adjusted mesh as 'xxx_v2_2.obj'

###### step 3: spherical parameterization #######
## https://github.com/mkazhdan/DenseP2PCorrespondences ## .ply
## https://github.com/garyptchoi/spherical-conformal-map ## .mat ## failed for left foot

###### step 4: spherical parameterization #######
## if using https://github.com/mkazhdan/DenseP2PCorrespondences

for name in names:
    mesh = Mesh(filename='./output/{}_v3.obj'.format(name)) 
    pldata = PlyData.read('./output/{}.cmcf.out.ply'.format(name))
    vertex = np.array([pldata['vertex']['x'].tolist(), pldata['vertex']['y'].tolist(), pldata['vertex']['z'].tolist()])
    vertex = np.transpose(vertex)
    pvertex = np.array([pldata['vertex']['px'].tolist(), pldata['vertex']['py'].tolist(), pldata['vertex']['pz'].tolist()])
    pvertex = np.transpose(pvertex)
    cvertex = np.array([pldata['vertex']['red'].tolist(), pldata['vertex']['green'].tolist(), pldata['vertex']['blue'].tolist()])
    cvertex = np.transpose(cvertex).astype(np.float) / 255
    tri_data = pldata['face'].data['vertex_indices']
    triangles = np.vstack(tri_data)
    mesh_sh = Mesh(v=pvertex, f=triangles, vc=cvertex)
    mesh_sh.write_obj('./output/{}.cmcf.out.obj'.format(name))
    savemat('./output/{}_map.mat', {'v': mesh.v, 'f': mesh_sh.f+1, 'map': mesh_sh.v})

###### step 5: check the spherical parameterization in mesh #######
for name in names:
    dic = loadmat('./output/{}_map.mat'.format(name))
    vc = dic['v'] 
    vc = (vc - np.min(vc, axis=0, keepdims=True)) / \
        (np.max(vc, axis=0, keepdims=True) - np.min(vc, axis=0, keepdims=True))
    mesh = Mesh(v=dic['map'], f=dic['f']-1, vc=vc)
    mesh.write_obj('./output/{}_map.obj'.format(name))

###### step 6: check the spherical parameterization in image #######
def Cartesian2Spherical(xyz):
    xy = xyz[:,0]**2 + xyz[:,1]**2
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) / np.pi # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:,1], xyz[:,0]) / (2*np.pi) + 0.5
    return np.stack([theta,phi], axis=1)

for name in names:
    map_close = loadmat('./output/{}_map.mat'.format(name))
    coords = torch.from_numpy(Cartesian2Spherical(map_close['map']))
    vc = map_close['v'] 
    vc = (vc - np.min(vc, axis=0, keepdims=True)) / \
        (np.max(vc, axis=0, keepdims=True) - np.min(vc, axis=0, keepdims=True))

    dsize = 500
    img = torch.zeros((dsize, dsize, 3))
    indices = (coords*dsize).long()
    img[indices[:, 1], indices[:, 0]] = torch.from_numpy(vc).float()
    cv2.imwrite('./output/{}_map.png'.format(name), np.uint8(255*img.numpy()))
    print(coords.shape)

###### step 7: save indices in dictionary #######
dic = {}
n_inx = 0
for name in names:  
    dic['{}_index_From_Template'.format(name)] = torch.from_numpy(np.load('./output/{}_indices.npy'.format(name)))
    dic['{}_index_From_Map'.format(name)] = torch.load('./output/{}_index.tch'.format(name))
    n_inx = n_inx + dic['{}_index_From_Template'.format(name)].shape[0]
    print(dic['{}_index_From_Template'.format(name)].shape[0])
torch.save(dic, './output/body_index.tch')
