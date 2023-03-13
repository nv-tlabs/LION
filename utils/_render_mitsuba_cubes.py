# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import sys, os, subprocess
import mitsuba as mi 
import OpenEXR
import Imath
from PIL import Image
from plyfile import PlyData, PlyElement
import torch 
import time 
from PIL import Image, ImageChops
random_str = hex(int(time.time() + 12345))[2:] 
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0))) ##border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox: 
        return im.crop(bbox)
    else:
        return im


PATH_TO_MITSUBA2 = "/home/xzeng/code/mitsuba2/build/dist/mitsuba" ## Codes/mitsuba2/build/dist/mitsuba"  # mitsuba exectuable


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

xml_head = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>

    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="800"/>
            <integer name="height" value="600"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
   
    <bsdf type="roughdielectric" id="trans">
        <float name="int_ior" value="1.31"/>
        <float name="ext_ior" value="1.6"/>
   </bsdf>
        <bsdf type="roughplastic" id="trans2">
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.05"/>
              <float name="intIOR" value="1.46"/>
              <rgb name="diffuseReflectance" value="0.6,0.3,0.2"/> <!-- default 0.5 -->
        </bsdf>
    <bsdf type="roughdielectric" id="trans3">
        <float name="int_ior" value="1.51"/>
        <float name="ext_ior" value="1.0"/>
        <rgb name="specular_reflectance" value="0.6,0.3,0.2"/> 
   </bsdf>

"""

# I also use a smaller point size
xml_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
xml_cube_segment = \
    """
    <shape type="ply" >
        <string name="filename" value="./script/paper/vis_voxel_exp/unit_cube2w1.ply"/> 
        <transform name="toWorld">
            <scale x="{}" y="{}" z="{}"/>
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="roughplastic" >
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.05"/>
              <float name="intIOR" value="1.46"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""

xml_shape_segment = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 
        <bsdf type="roughplastic" id="surfaceMaterialshape">
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.05"/>
              <float name="intIOR" value="1.46"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""
        ## <ref name="bsdf" id="trans3"/>
#    <shape type="sphere">
#        <float name="radius" value="0.025"/>
#        <transform name="toWorld">
#            <translate x="0" y="0" z="0.4"/>
#
#        </transform>
#        <bsdf type="diffuse">
#            <rgb name="reflectance" value="0.7,0.4,0.5"/>
#        </bsdf>
#    </shape>
#
#




#    <shape type="sphere">
#        <float name="radius" value="0.025"/>
#        <transform name="toWorld">
#            <translate x="1" y="1" z="1"/>
#
#        </transform>
#        <bsdf type="diffuse">
#            <rgb name="reflectance" value="0.7,0.4,0.5"/>
#        </bsdf>
#    </shape>
#    <shape type="rectangle">
#        <transform name="toWorld">
#            <scale x="0.05" y="0.05" z="0.05"/>
#            <translate x="1" y="1" z="0.8"/>
#        </transform>
#
#        <bsdf type="diffuse">
#            <rgb name="reflectance" value="0.7,0.4,0.5"/>
#        </bsdf>
#    </shape>
#    <shape type="rectangle">
#        <transform name="toWorld">
#            <scale x="0.05" y="0.05" z="0.05"/>
#            <translate x="1" y="1.2" z="0.6"/>
#        </transform>
#
#        <bsdf type="diffuse">
#            <rgb name="reflectance" value="0.7,0.4,0.5"/>
#        </bsdf>
#    </shape>
#
#
#    <shape type="sphere"> 
#	<point name="center" x="-1" y="1" z="1.5"/> 
#	<float name="radius" value="0.05"/> 
#        <emitter type="area">
#            <rgb name="radiance" value="30,10,10"/>
#        </emitter>
#    </shape>
#

#        <bsdf type="diffuse">
#           <rgb name="reflectance" value="1,0,0"/>
#        </bsdf>
# 
#        <bsdf type="diffuse">
#           <rgb name="reflectance" value="{},{},{}"/>
#        </bsdf>
#        <bsdf type="roughplastic" id="surfaceMaterialshape">
#              <string name="distribution" value="ggx"/>
#              <float name="alpha" value="0.05"/>
#              <float name="intIOR" value="1.46"/>
#              <rgb name="diffuseReflectance" value="1,0,0.5"/> <!-- default 0.5 -->
#        </bsdf>
#

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
#    <shape type="sphere"> 
#	<point name="center" x="5" y="5" z="5"/> 
#	<float name="radius" value="1.5"/> 
#        <emitter type="area">
#            <rgb name="radiance" value="3,3,3"/>
#        </emitter>
#    </shape>

	## <point name="center" x="2" y="2" z="2"/> 
#    <shape type="rectangle">
#        <transform name="toWorld">
#            <scale x="10" y="10" z="1"/>
#            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
#        </transform>
#        <emitter type="area">
#            <rgb name="radiance" value="6,6,6"/>
#        </emitter>
#    </shape>
#

def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
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
def ConvertEXRToJPG(exrfile, jpgfile):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        print(rgb[i].max(), rgb[i].min())
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)

    rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
    # rgb8 = [Image.fromarray(c.astype(int)) for c in rgb]
    Image.merge("RGB", rgb8).save(jpgfile, "PNG") 
    ##, quality=95) 
    img = Image.open(jpgfile) 
    img = trim(img)
    img.save(jpgfile) 


def render_cubes2png(pcl, filename, colorm=[24,107,239], vs_size=0.5,
        sample_count=256, out_width=800, out_height=600, 
        lookat_1=3, lookat_2=3, lookat_3=3, trim_img=0):
    xml_head = xml_head_segment.format(
            lookat_1, lookat_2, lookat_3,
            sample_count, out_width, out_height)
    xml_segments = [xml_head]
    print('color: ', colorm)
    for i in range(pcl.shape[0]):
        color = [c/255.0 for c in colorm]
        xml_segments.append(xml_cube_segment.format(
            vs_size, vs_size, vs_size,
            pcl[i, 0], pcl[i, 1], pcl[i, 2], 
            *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    xmlFile = '/tmp/tmp_%s.xml'%random_str # ("%s/xml/%s.xml" % (folder, filename))
    exrFile = '/tmp/tmp_%s.exr'%random_str ##("%s/exr/%s.exr" % (folder, filename))
    ## png = ("%s/%s.png" % (folder, filename))
    with open(xmlFile, 'w') as f:
        f.write(xml_content)
    f.close()
    print('save as xmlFile: ', xmlFile)

    if not os.path.exists(os.path.dirname(exrFile)):
        os.makedirs(os.path.dirname(exrFile))
    #subprocess.run([PATH_TO_MITSUBA2, '-o', exrFile, xmlFile])  
    png = filename 
    if not os.path.exists(os.path.dirname(png)):
        os.makedirs(os.path.dirname(png))

    #print(['Converting EXR to PNG ...', png])
    #ConvertEXRToJPG(exrFile, png)
    scene = mi.load_file(xmlFile) 
    image = mi.render(scene) ##, spp=256) 
    mi.util.write_bitmap(png, image) 
    if trim_img:
        img = Image.open(png) 
        img = trim(img)
        img.save(png) 

    return png 


if __name__ == "__main__":
    if (len(sys.argv) < 2):
       print('filename to npy/ply is not passed as argument. terminated.')
       raise ValueError 

    pathToFile = sys.argv[1]


    main(pathToFile)
