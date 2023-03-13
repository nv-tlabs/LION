# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
from loguru import logger
import sys, os, subprocess
import copy
import OpenEXR
import Imath
from PIL import Image
## from plyfile import PlyData, PlyElement
import torch 
import open3d as o3d 
from PIL import Image, ImageChops
import time 
random_str = hex(int(time.time() + 12345))[2:] 
PATH_TO_MITSUBA2 = "/home/xzeng/code/mitsuba2/build/dist/mitsuba" ## Codes/mitsuba2/build/dist/mitsuba"  # mitsuba exectuable

# replaced by command line arguments
def standardize_bbox_based_on(pcl, eps):
    pcl = pcl.numpy()[:, [0,2,1]] 
    eps = eps.numpy()[:, [0,2,1]]
    pcl, center, scale = standardize_bbox(pcl, return_center_scale=1)
    eps = (eps - center) / scale if eps is not None else None 
    offset = - 0.475 - pcl[:,2].min()
    eps[:,2] += offset 
    return torch.from_numpy(eps) 

# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load
def rotate_pts(pts, r, axis=1, do_transform=0, is_point_flow_data=1, eps=None):
    assert(len(pts.shape) == 2), f'require N,3 tensor, get: {pts.shape}'
    ## logger.info('rotating pts: {}, get eps: {} ', pts.shape, eps is not None )
    is_tensor = torch.is_tensor(pts) 
    if not is_tensor:
        pts = torch.from_numpy(pts) 
    if eps is not None and not torch.is_tensor(eps):
        eps = torch.from_numpy(eps)
    if do_transform:
        pcl = pts.cpu().numpy() 
        eps = eps.cpu().numpy() if eps is not None else None 
        if not is_point_flow_data:
            pcl[:,0] *= -1
            pcl = pcl[:, [2,1,0]] 

            if eps is not None:
                eps[:,0] *= -1
                eps = eps[:, [2,1,0]] 

        pcl, center, scale = standardize_bbox(pcl, return_center_scale=1)
        eps = (eps - center) / scale if eps is not None else None 
        pcl = pcl[:, [2, 0, 1]]
        pcl[:,0] *= -1
        pcl[:,2] += 0.0125
        if eps is not None:
            eps = eps[:, [2, 0, 1]]
            eps[:,0] *= -1
            eps[:,2] += 0.0125 

        offset = - 0.475 - pcl[:,2].min()
        pcl[:,2] += offset 
        if eps is not None:
            eps[:,2] += offset 
        pts = torch.from_numpy(pcl)
        eps = torch.from_numpy(eps) if eps is not None else None

    pcl = o3d.geometry.PointCloud() 
    pcl.points = o3d.utility.Vector3dVector(pts.cpu())
    if axis == 1:
        R = pcl.get_rotation_matrix_from_xyz((0, - r * np.pi / 2, 0))
    elif axis == 2:
        R = pcl.get_rotation_matrix_from_xyz((0, 0, - r * np.pi / 2))
    elif axis == 0:
        R = pcl.get_rotation_matrix_from_xyz((- r * np.pi / 2, 0, 0))

    #mesh_r = copy.deepcopy(pcl)
    #mesh_r.rotate(R, center=(0, 0, 0))
    #pts = np.asarray(mesh_r.points)

    h_center = w_center = 0 
    center = np.array([h_center, w_center, 0]).reshape(-1,3)
    pts = np.matmul(pts.numpy() - center, R.T) + center 
    eps = np.matmul(eps.numpy() - center, R.T) + center if eps is not None else eps  

    if is_tensor:
        pts = torch.from_numpy(pts)
        eps = torch.from_numpy(eps) if eps is not None and not torch.is_tensor(eps) else eps 

    if eps is not None:
        return pts, eps 
    return pts 

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
xml_head_segment = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{},{},{}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="{}"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{}"/>
            <integer name="height" value="{}"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

# I also use a smaller point size
xml_ball_segment = ['']*10
xml_ball_segment[0] = \
    """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
xml_ball_segment[1] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="plastic" >
              <float name="intIOR" value="2.0"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""


xml_ball_segment[2] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="plastic" >
              <float name="intIOR" value="1.9"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""

xml_ball_segment[3] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic" >
              <float name="intIOR" value="1.9"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""
xml_ball_segment[4] = \
"""
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic" >
              <float name="intIOR" value="1.6"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""
xml_ball_segment[5] = \
    """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic">
              <float name="intIOR" value="1.7"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>
    </shape>
"""
xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-1,1,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0))) ##border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox: 
        return im.crop(bbox)
    else:
        return im


def colormap(x, y, z):
    if torch.is_tensor(x): 
        x = x.cpu().numpy()
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def standardize_bbox(pcl, return_center_scale=0):
    #pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    #np.random.shuffle(pt_indices)
    #pcl = pcl[pt_indices]  # n by 3
    if torch.is_tensor(pcl):
        pcl = pcl.numpy()
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    #print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    if return_center_scale:
        return result, center, scale 
    return result


# only for debugging reasons
def writeply(vertices, ply_file):
    sv = np.shape(vertices)
    points = []
    for v in range(sv[0]):
        vertex = vertices[v]
        points.append("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
    print(np.shape(points))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(vertices), "".join(points)))
    file.close()


# as done in https://gist.github.com/drakeguan/6303065
def ConvertEXRToJPG(exrfile, jpgfile, trim_img):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)

    rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
    Image.merge("RGB", rgb8).save(jpgfile, "PNG") ##JPEG", quality=95)
    img = Image.open(jpgfile) 
    if trim_img:
        img = trim(img)
    img.save(jpgfile) 

def pts2png(input_pts, file_name, colorm=[24,107,239], 
        skip_if_exists=False, is_color_list=False, 
        sample_count=256, out_width=1600, out_height=1200, 
        ball_size=0.025, do_standardize=0, same_computed_loc_color=0, material_id=0, precomputed_color=None,
        output_xml_file=None,
        use_loc_color=False, lookat_1=3, lookat_2=3, lookat_3=3, do_transform=1, trim_img=0):
    """
    Argus: 
        input_pts: (B,N,3) the points to be render 
        file_name: list; output image name 
    """
    assert(len(input_pts.shape) == 3), f'expect: B,N,3; get: {input_pts.shape}'
    assert(type(file_name) is list), f'require file_name as list'
    xml_head = xml_head_segment.format(
            lookat_1, lookat_2, lookat_3,
            sample_count, out_width, out_height)
    input_pts = input_pts.cpu()
    # print('get shape; ', input_pts.shape)
    color_list = []
    for pcli in range(0, input_pts.shape[0]):
        xmlFile = '/tmp/tmp_%s.xml'%random_str if output_xml_file is None else output_xml_file 
        # ("%s/xml/%s.xml" % (folder, filename))
        exrFile = '/tmp/tmp_%s.exr'%random_str ##("%s/exr/%s.exr" % (folder, filename))
        png = file_name[pcli] 
        if skip_if_exists and os.path.exists(png):
            print(f'find png: {png}, skip ')
            continue 
        pcl = input_pts[pcli, :, :]
        if do_transform:
            pcl = standardize_bbox(pcl)
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            offset = - 0.475 - pcl[:,2].min()
            pcl[:,2] += offset 
        if do_standardize:
            pcl = standardize_bbox(pcl)
            offset = - 0.475 - pcl[:,2].min()
            pcl[:,2] += offset 

        xml_segments = [xml_head]
        for i in range(pcl.shape[0]):
            if precomputed_color is not None:
                color = precomputed_color[i]
            elif use_loc_color and not same_computed_loc_color: 
                color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
            elif use_loc_color and same_computed_loc_color:
                if pcli == 0:
                    color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
                    color_list.append(color)
                else:
                    color = color_list[i] # same color as first shape 
            elif is_color_list:
                color = colorm[pcli]
                color = [c/255.0 for c in color] 
            else:
                color = [c/255.0 for c in colorm] 
            xml_segments.append(xml_ball_segment[material_id].format(
                ball_size, 
                pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        ## print('using color: ', color)
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        if not os.path.exists(os.path.dirname(xmlFile)):
            os.makedirs(os.path.dirname(xmlFile))
        with open(xmlFile, 'w') as f:
            f.write(xml_content)
        logger.info('[render_mitsuba_pc] write output at: {}', xmlFile)
        f.close()

        if not os.path.exists(os.path.dirname(exrFile)):
            os.makedirs(os.path.dirname(exrFile))
        if not os.path.exists(os.path.dirname(png)):
            os.makedirs(os.path.dirname(png))
        logger.info('*'*20 + f'{png}' +'*'*20)
        # mitsuba2 
        #subprocess.run([PATH_TO_MITSUBA2, '-o', exrFile, xmlFile])  
        #ConvertEXRToJPG(exrFile, png, trim_img)
        scene = mi.load_file(xmlFile) 
        image = mi.render(scene) ##, spp=256) 
        mi.util.write_bitmap(png, image) 
        if trim_img:
            img = Image.open(png) 
            img.save(png) 

    return png 


if __name__ == "__main__":
    if (len(sys.argv) < 2):
       print('filename to npy/ply is not passed as argument. terminated.')
       raise ValueError 

    pathToFile = sys.argv[1]


    main(pathToFile)
