# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os 
import numpy as np 
import open3d as o3d
import sys
import torch 
import point_cloud_utils as pcu
from PIL import Image 

sys.path.append('.')
import torch  
from utils._render_mitsuba_cubes import render_cubes2png  
# from script.paper.vis_mesh.render_mitsuba import reformat_ply 
def reformat_ply(input, output, r=0, is_point_flow_data=0, 
        ascii=False, write_normal=False, fixed_trimesh=1):
    m = open3d.io.read_triangle_mesh(input)
    pcl = np.asarray(m.vertices) 
    if not is_point_flow_data:
        pcl[:,0] *= -1
        pcl = pcl[:, [2,1,0]] 

    pcl = standardize_bbox(pcl)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:,0] *= -1
    pcl[:,2] += 0.0125

    offset = - 0.475 - pcl[:,2].min()
    pcl[:,2] += offset 
    m.vertices = open3d.utility.Vector3dVector(pcl) 

    R = m.get_rotation_matrix_from_xyz((0, 0, - r * np.pi / 2))
    mesh_r = copy.deepcopy(m)
    mesh_r.rotate(R, center=(0, 0, 0))
    ## o3d.visualization.draw_geometries([mesh_r])
    open3d.io.write_triangle_mesh(output, mesh_r, write_ascii=ascii, write_vertex_normals=False) 
    if fixed_trimesh:
        mesh = trimesh.load_mesh(output) ## '../models/featuretype.STL') 
        trimesh.repair.fix_inversion(mesh) 
        trimesh.repair.fix_normals(mesh) 
        mesh.export(output)
    logger.info(f'load {input}, write as {output}; ascii={ascii}, write_normal: {write_normal}') 
    return output 


def get_vpts(cubes):  
    import kaolin 
    v,f = kaolin.ops.conversions.voxelgrids_to_cubic_meshes(cubes) ##voxelgrids_to_trianglemeshes(voxel_volume)
    v = [vi.cpu() for vi in v]
    f = [fi.cpu() for fi in f] 
    return v, f 

def create_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def convert_cube_2_mesh(voxel_size, center_list, output_path, overwrite=0, colorm=[93,64,211], rotate=None, config={}):
    scale = 0.5 * (center_list.max() - center_list.min())
    pcl = center_list 
    mins = np.amin(pcl, axis=0).reshape(1,pcl.shape[-1]) ##np.amin(pcl, axis=1)[:, None, :], axis=0)[None, None, :]
    maxs = np.amax(pcl, axis=0).reshape(1,pcl.shape[-1]) ##np.amin(pcl, axis=1)[:, None, :], axis=0)[None, None, :]
    
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)

    center_list = (center_list - center) / scale ## (center_list.max() - center_list.min()) 
    
    center_list = center_list[:,[2,0,1]]
    center_list[:,0] *= -1 #center_list
    offset = - 0.475 - center_list[:,2].min()
    center_list[:,2] += offset 
    if rotate is not None:
        pts = torch.from_numpy(center_list)
        pcl = o3d.geometry.PointCloud() 
        pcl.points = o3d.utility.Vector3dVector(pts.cpu())
        R = pcl.get_rotation_matrix_from_xyz((0, 0, - rotate * np.pi / 2))
        center = np.array([0, 0, 0]).reshape(-1,3)
        pts = np.matmul(pts.numpy() - center, R.T) + center 
        center_list = pts

    output_name = output_path
    print('center_list: ', center_list.shape) 
    if not os.path.exists(os.path.dirname(output_name)):
        os.makedirs(os.path.dirname(output_name)) 
    if os.path.exists(output_name) and not overwrite:
        print(f'find rendered output: {output_name}, skip')
        return 
    out = render_cubes2png(pcl=center_list, filename=output_name, 
            vs_size=0.9*voxel_size/scale, colorm=colorm, **config) 
    print(' save as: ', out) 
    #img = Image.open(out)
    #img.show()
    print('output_path: ', output_path)
    return out 

def create_unit_cube():
    voxelgrid = torch.tensor([1], device='cuda', dtype=torch.uint8).view(1,1,1,1)
    v, f = get_vpts(voxelgrid) 
    v = v[0]
    f = f[0] 
    v = v - 0.5 # [-0.5, 0.5] 
    print('v: ', v)
    print('f: ', v)
    output_name = './script/paper/vis_voxel_exp/unit_cube.ply'
    print('save output as: ', output_name)
    pcu.save_mesh_vf(output_name, v.numpy(), f.numpy()) 
    reformat_ply(output_name, output_name, r=0) 

def write_small_cube():
    mesh_compare = './script/paper/vis_voxel_exp/unit_cube2.ply'
    m = o3d.io.read_triangle_mesh(mesh_compare)
    pcl = np.asarray(m.vertices) * 0.5  
    m.vertices = o3d.utility.Vector3dVector(pcl) ##.float()) ## torch.from_numpy(pcl))
    o3d.io.write_triangle_mesh(mesh_compare.replace('.ply', 'w1.ply'), 
            m, write_ascii=True, write_vertex_normals=True) 

# write_small_cube()
if __name__ == '__main__':
    input_path = '/home/xzeng/plots/voxel_exp/voxel_cube/raw'
    index = "3 328 91 83 74 73 64 63 54 51 48 45 41 30 22"
    convert_cube_2_mesh(input_path, index)
# create_unit_cube() 
