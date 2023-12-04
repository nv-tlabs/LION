import numpy as np
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import open3d 
import copy
import trimesh 
from loguru import logger 
import sys, os, subprocess
import OpenEXR
import Imath
from PIL import Image
from plyfile import PlyData, PlyElement
import torch 
from PIL import Image, ImageChops
import time 
random_str = hex(int(time.time() + 12345))[2:] 
def standardize_bbox(pcl):
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)  #0.5
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result
def standardize_to_same_range(ref, src):
    mesh_r = open3d.io.read_triangle_mesh(ref)
    pcl = np.array(mesh_r.vertices)
    pt_i = src
    for i in range(3): 
        min_p = pcl[:, i].min() 
        max_p = pcl[:, i].max() 
        r = max_p - min_p 
        c = pt_i[:, i]
        c = (c - c.min()) / (c.max() - c.min()) 
        c = c*r + min_p # same range 
        pt_i[:, i] = c 
        #logger.info('pts: i{}= {}, {}, {}', i, pt_i[:, i].min(), pt_i[:, i].max(), pt_i[:, i].mean())
        #logger.info('pts: i{}= {}, {}, {}', i, pcl[:, i].min(), pcl[:, i].max(), pcl[:, i].mean())
    return pt_i 
    

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


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0))) ##border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox: 
        return im.crop(bbox)
    else:
        return im

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
        ## <sampler type="independent">
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
        </bsdf> 
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
xml_shape_segment = ['']*10
xml_shape_segment[0] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="roughplastic" id="surfaceMaterialshape">
              <float name="intIOR" value="1.46"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""
xml_shape_segment[1] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="roughplastic" id="surfaceMaterialshape">
              <float name="intIOR" value="1.6"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""


            ### <bsdf type="diffuse">
xml_shape_segment[2] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

            <bsdf type="diffuse">
                    <texture type="mesh_attribute" name="reflectance">
                    <string name="name" value="vertex_color"/>
                    </texture>
           </bsdf> 
    </shape>
"""
xml_shape_segment[3] = \
    """
    <shape type="obj" id="mesh">
        <string name="filename" value="{}"/> 
            <bsdf type="diffuse">
                    <texture type="mesh_attribute" name="reflectance">
                                <string name="name" value="vertex_color"/>
                                        </texture>
                                            </bsdf> 
    </shape>
"""


xml_shape_segment[4] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="roughplastic" id="surfaceMaterialshape">
              <float name="intIOR" value="1.6"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""
xml_shape_segment[5] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="roughplastic" id="surfaceMaterialshape">
              <float name="intIOR" value="1.7"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""

xml_shape_segment[6] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="plastic" id="surfaceMaterialshape">
              <float name="intIOR" value="1.9"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""

xml_shape_segment[7] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="roughplastic" id="surfaceMaterialshape">
              <float name="intIOR" value="1.9"/>
              <string name="distribution" value="ggx"/>
              <float name="alpha" value="0.2"/>
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""

xml_shape_segment[8] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="plastic" id="surfaceMaterialshape">
              <rgb name="diffuseReflectance" value="{},{},{}"/> <!-- default 0.5 -->
        </bsdf>

    </shape>
"""

xml_shape_segment[9] = \
    """
    <shape type="ply" id="mesh">
        <string name="filename" value="{}"/> 

        <bsdf type="plastic" id="surfaceMaterialshape">
              <float name="intIOR" value="2.0"/>
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
##    <shape type="rectangle"> <!-- add light -->
##        <transform name="toWorld">
##            <scale x="10" y="10" z="1"/>
##            <lookat origin="1,-1,20" target="0,0,0" up="0,0,1"/>
##        </transform>
##        <emitter type="area">
##            <rgb name="radiance" value="6,6,6"/>
##        </emitter>
##    </shape>

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


#def standardize_bbox(pcl, points_per_object):
#    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
#    np.random.shuffle(pt_indices)
#    pcl = pcl[pt_indices]  # n by 3
#    mins = np.amin(pcl, axis=0)
#    maxs = np.amax(pcl, axis=0)
#    center = (mins + maxs) / 2.
#    scale = np.amax(maxs - mins)
#    print("Center: {}, Scale: {}".format(center, scale))
#    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
#    return result


# only for debugging reasons
def writeply(vertices, ply_file):
    sv = np.shape(vertices)
    points = []
    for v in range(sv[0]):
        vertex = vertices[v]
        points.append("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
    ## print(np.shape(points))
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
def ConvertEXRToJPG(exrfile, jpgfile, trim_img=True):
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
    Image.merge("RGB", rgb8).save(jpgfile) 
    ##, quality=95) 
    img = Image.open(jpgfile) 
    if trim_img:
        img = trim(img)
    img.save(jpgfile) 


def main(pathToFile, pathToMesh, filename=None, png=None, folder=None, colorm=[24,107,239],
        lookat_1=3, lookat_2=3, lookat_3=3,
        sample_count=256, out_width=1600, out_height=1200, material_id=0, trim_img=True,
        xmlFile=None, return_xml_only=0):
    ## xml_head = xml_head_segment.format(sample_count, out_width, out_height)

    xml_head = xml_head_segment.format(
            lookat_1, lookat_2, lookat_3,
            sample_count, out_width, out_height)
    xml_segments = [xml_head]
    color = [c/255.0 for c in colorm]
    if material_id in [2, 3]:
        xml_segments.append(xml_shape_segment[material_id].format(
            pathToMesh))
    else:
        xml_segments.append(xml_shape_segment[material_id].format(
            pathToMesh, *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)
    xmlFile = '/tmp/tmp_%s.xml'%random_str if xmlFile is None else xmlFile 
    exrFile = '/tmp/tmp_%s.exr'%random_str 
    png = ("%s/%s.png" % (folder, filename)) if png is None else png 
    if not os.path.exists(os.path.dirname(xmlFile)):
        os.makedirs(os.path.dirname(xmlFile))
    with open(xmlFile, 'w') as f:
        f.write(xml_content)
    f.close()
    if return_xml_only:
        return xmlFile 

    if not os.path.exists(os.path.dirname(exrFile)):
        os.makedirs(os.path.dirname(exrFile))

    if not os.path.exists(os.path.dirname(png)):
        os.makedirs(os.path.dirname(png))

    # use mitsuba2 
    use_mit2 = 0 
    if use_mit2:
        subprocess.run([PATH_TO_MITSUBA2, '-o', exrFile, xmlFile]) 
        ConvertEXRToJPG(exrFile, png, trim_img)
    else:
        ## use mitsuba3 
        scene = mi.load_file(xmlFile) 
        image = mi.render(scene) ##, spp=256) 
        ## print(type(image))
        mi.util.write_bitmap(png, image) 

    logger.info('*'*20 + f'{png}' + '*'*20)
    #print(['Converting EXR to PNG ...'])
    return png 


if __name__ == "__main__":
    if (len(sys.argv) < 3):
       print('filename to npy/ply is not passed as argument. terminated.')
       raise ValueError 
    # Check the main function
