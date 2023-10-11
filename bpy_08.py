import bpy
import mathutils
import math
import os
import numpy as np
# import bpycv
import scipy.io as sio
from PIL import Image, ImageDraw
import re
import glob, subprocess
import random
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


# I want to use hdri to be the light source
# I want to fix the relationship bwtn the object and camera, and move together.


#### ADD optical flow output

print("START")

def draw_6d_pose(img, xyzs_in_obj, pose, intrinsic, color=(255, 0, 0)):
    R, T = pose[:, :3], pose[:, 3]
    # np.dot(xyzs, R.T) == np.dot(R, xyzs.T).T
    xyzs_in_cam = np.dot(xyzs_in_obj, R.T) + T
    xyzs_in_image = np.dot(xyzs_in_cam, intrinsic.T)
    xys_in_image = xyzs_in_image[:, :2] / xyzs_in_image[:, 2:]
    xys_in_image = xys_in_image.round().astype(int)
    for xy in xys_in_image:
        img = cv2.circle(img, tuple(xy), 10, color, -1)
    return img


def vis_ycb_6d_poses(img, mat, xyzs=None):
    vis = img.copy()
    n = mat["poses"].shape[-1]
    colors = np.array(boxx.getDefaultColorList(n + 3)) * 255  # get some random colors
    for idx in range(n):
        pose = mat["poses"][:, :, idx]
        intrinsic = mat["intrinsic_matrix"]
        if xyzs is None:
            xyzs = mat.get("bound_boxs")[idx]
        draw_6d_pose(vis, xyzs, pose, intrinsic, colors[idx + 1])
    return vis


def readExr(exr_dir):
  return cv2.imread(exr_dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

def render(out_dir):
    '''
        return rgb, depth, id mask. object index was assigned in __init__
        id_mask: see config files
    '''
    
    tree = bpy.context.scene.node_tree
    tree.render_quality = "HIGH"
    tree.edit_quality = "HIGH"
    tree.use_opencl = True

    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)

    #================ collect images and label ===================
    render_node = tree.nodes.new('CompositorNodeRLayers')
    rgb_node = tree.nodes.new('CompositorNodeOutputFile')   # rgb
    rgb_node.format.file_format = 'PNG'
    rgb_node.base_path = out_dir
    rgb_node.file_slots[0].path = "rgbB"
    links.new(render_node.outputs['Image'], rgb_node.inputs[0])
    
    
    depth_map = tree.nodes.new("CompositorNodeMapRange")
    depth_map.inputs[2].default_value = 20 # input range endpoint

    depth_node = tree.nodes.new('CompositorNodeOutputFile')   # depth
    depth_node.format.file_format = 'PNG' # 'OPEN_EXR' # As it is for the masking, I just use PNG. 
    depth_node.base_path = out_dir
    depth_node.file_slots[0].path = "depthB"
    links.new(render_node.outputs['Depth'], depth_map.inputs[0])
    links.new(depth_map.outputs['Value'], depth_node.inputs[0])

    bpy.ops.render.render(write_still=True) 
    
    
    for n in tree.nodes:
        tree.nodes.remove(n)

    return 


def hide_render_recurvise(obj, value):
    obj.hide_render = value
    for child in obj.children:
        if child.type == 'MESH':
            hide_render_recurvise(child, value)


root = "/hdd/hyebin/blender/assets"
output = "/hdd/hyebin/blender/output/optical_flow/"
# Get the folder names
asset_names = []
folder_path = os.path.join(root,"models/test")
for item in os.listdir(folder_path):
    sub_folder = os.path.join(folder_path, item)
    if os.path.isdir(sub_folder):
        asset_names.append(sub_folder)
        
        
gray_path = os.path.join(root,"gray")
gray_names = []
for item in os.listdir(gray_path):
    gray_names.append(os.path.join(gray_path, item))

gray_names.sort()

white_path =os.path.join(root,"back/w.png")

bg_root = "/hdd/hyebin/BG20k/"
bg_txt = os.path.join(bg_root, "train.txt")
bg_path = []
with open(bg_txt, 'r') as file:
    file_names = file.read().splitlines()
    for bg in file_names:
        bg_path.append(os.path.join(os.path.join(bg_root, 'train'), bg))



# load sample hdr
hdr_folder = bpy.path.abspath(os.path.join(root, "hdrs"))
hdris = [os.path.join(hdr_folder, f) for f in os.listdir(hdr_folder)] # if f.endswith(".exr")]



def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    
# 원점 대칭시키기 위한 변환 행렬 생성
mirror_matrix = mathutils.Matrix.Scale(-1, 4, (1,1,1))
np_mirror = np.array(mirror_matrix)[:3,:3]
print(np_mirror)
    

for path in asset_names:
    # Clear objects in the scene 
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # CLear scenes
    for c in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.unlink(c)
    
    # Clear bpycv
    # bpycv.clear_all()

    # A transparency stage for holding rigid body
    # stage = bpycv.add_stage(transparency=True)
    
    # load checkerboard
    # checker = bpy.data.images.load('/hdd/hyebin/blender/checkerboard_4048_4048_17.png')

    # checkerboard plane
    # 이미지를 넣을 평면 생성
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object

    # 이미지 텍스처 생성
    mat = bpy.data.materials.new(name="Image Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_image = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    # tex_image.image = bpy.data.images.load('/hdd/hyebin/blender/checkerboard_4048_4048_17.png')
    mat.node_tree.links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    plane.data.materials.append(mat)
    plane.name = 'hyebin'


    bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
    sphere = bpy.context.active_object
    sphere.scale = (10, 10, 10)
    sphere.name = 'hyebin_shpere'

    # light_data = bpy.data.lights.new(name='hyebin_light', type='POINT')
    # light_obj = bpy.data.objects.new(name='hyebin_light', object_data = light_data)
    bpy.ops.object.light_add(type='AREA') # POINT, SPOT
    light_obj = bpy.context.object
    light_obj.data.energy = 1500
    light_obj.name = 'hyebin_light'
    light_obj.scale = (1.5, 1.5, 1.0)
    light_obj.visible_camera = False

        
    # .blend 파일 append
    for item in os.listdir(path):
        file = os.path.join(path, item)
        if item == 'untitled.blend':
            break
    print(file)
    with bpy.data.libraries.load(file, link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]

    # Append된 객체 가져오기
    appended_objects = [obj for obj in data_to.objects if obj is not None]

    # 추가된 객체를 현재 씬에 추가
    scene = bpy.context.scene
    for obj in appended_objects:
        scene.collection.objects.link(obj)
        
    print(path)
    model_name = path.split('/')[-1]

    # print(bpy.context.scene.collection.children.items()
    camera = bpy.data.cameras.new("Camera")
    camera_obj = bpy.data.objects.new("Camera", camera)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024



    def random_point_on_sphere(radius):
        theta = 0

        while theta < (math.pi/3) or theta > (math.pi*2/3):
            u = np.random.uniform(0,1)
            v = np.random.uniform(0,1)
            theta = 2* math.pi * u
            phi = math.acos(2*v-1)
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            theta = math.acos(z / radius)
        # print(theta*180/math.pi)

        return (x,y,z)
    
    
    num = 0
    # bpy.context.scene.render.film_transpraent = True
    # bpy.context.scene.cycles.film_transparent_glass = True
    # Calculate camera position and orientation to fit object in 1/3 of the camera scene
    print(bpy.data.objects.items())
    filterd_objects = [ob for ob in bpy.data.objects if not ob.parent or ob.parent.type != 'MESH']
    print(" [[  FILTERED  ]] \n", filterd_objects)

    for obj in filterd_objects:
    # for key, obj in child.items():
        print(obj, obj.name, obj.name[:6], obj.type)
        
        if obj.name[:6] == 'Camera' or isinstance(obj, bpy.types.Collection) or 'hyebin' in obj.name or obj.type == 'EMPTY' : # or obj.name == 'hyebin_cone':
            print('pass')
            pass
        
        else:
            num += 1
            if num == 2:
                break
                
            bbox_center = (0,0,0) # (x,y,z)

            # bpycv.activate_obj(obj)
            # hide from rendering if there are other objects
            for obj2 in bpy.data.objects:
            # for key2, obj2 in child.items():
                    
                if not isinstance(obj2, bpy.types.Collection) and obj2.name != obj.name and 'hyebin' not in obj2.name:
                    # if obj2.name != 'Camera':
                    hide_render_recurvise(obj2, True)
                    print("HIDING!!", obj2.name)
                    # print("HIDE", obj2.name)
                    
            # obj.hide_render = False
            # fine the Glass BSDF for checkerboard
            material = obj.active_material

            if material and material.use_nodes:
                tree = material.node_tree

                glass_bsdf_node = None
                for node in tree.nodes:
                    print(node.type)
                    if node.type == 'BSDF_GLASS':
                        glass_bsdf_node = node
                        break

                # if glass_bsdf_node:
                #     glass_bsdf_node.inputs['Color'].default_value = (0,0,0,0)

                if not glass_bsdf_node:
                    print("I CAN NOT FIND THE GLASS BSDF !!!")
                    break 


                
            for i in range(1):
                bpy.context.scene.camera.data.sensor_width = np.random.choice([12.8, np.random.uniform(22, 24), np.random.uniform(35, 37)])
                add = 0
                if bpy.context.scene.camera.data.sensor_width <= 24:
                    if bpy.context.scene.camera.data.sensor_width <= 13:
                        add = 4
                    else:
                        add = 1
                s = np.random.uniform(3,7)+add
                camera_position = random_point_on_sphere(s) # radius setting
                # print(camera_position)
                
                # Set camera position and orientation
                bpy.context.scene.camera.location = camera_position
                bpy.context.scene.camera.rotation_mode = 'QUATERNION'
                x = bbox_center[0] - camera_position[0]
                y = bbox_center[1] - camera_position[1]
                z = bbox_center[2] - camera_position[2]
                bpy.context.scene.camera.rotation_quaternion = mathutils.Vector((x,y,z)).to_track_quat('-Z', 'Y')
                # print(bpy.context.scene.camera.location, bpy.context.scene.camera.rotation_quaternion)

                light_obj.location = camera_position
                light_obj.rotation_mode = 'QUATERNION'
                light_obj.rotation_quaternion = mathutils.Vector((x,y,z)).to_track_quat('-Z', 'Y')

                # rotation & translation checkerboard
                plane_loc = ((-1)*camera_position[0], (-1)*camera_position[1], (-1)*camera_position[2]) # np.matmul(np_mirror, np.transpose(np.array(camera_position)))
                plane.location = (plane_loc[0], plane_loc[1], plane_loc[2])
                plane.rotation_mode = 'QUATERNION'
                plane.rotation_quaternion = mathutils.Vector((x,y,z)).to_track_quat('-Z', 'Y')
                
                # focal = bpy.context.scene.camera.data.lens
                sensor_width = bpy.context.scene.camera.data.sensor_width
                distance = math.sqrt(camera_position[0]**2+camera_position[1]**2+camera_position[2]**2)*2
                
                
                # ratio = sensor_width/distance
                ratio = 2*sensor_width*s/100
                plane.scale = (ratio, ratio, 1.0)

                # make the object bigger
                plane_size = plane.dimensions.length / np.random.uniform(1.5,2.5)
                curr_size = obj.dimensions.length
                scale = plane_size / curr_size
                obj.scale = (obj.scale[0]*scale, obj.scale[1]*scale, obj.scale[2]*scale)
                # x = 0.5 * (obj.bound_box[0][0] + obj.bound_box[6][0])
                # y = 0.5 * (obj.bound_box[0][1] + obj.bound_box[6][1])
                # z = 0.5 * (obj.bound_box[0][2] + obj.bound_box[6][2])
                
                
                # Get the environment node tree of the current scene
                bpy.context.scene.use_nodes = True
                node_tree = bpy.context.scene.world.node_tree

                # Clear all nodes
                node_tree.nodes.clear()

                
                # Render and save image
                bpy.context.scene.render.film_transparent = True
                bpy.context.scene.cycles.film_transparent_glass = True
                bpy.context.scene.render.use_stamp_lens = True
                bpy.context.scene.render.filepath = output+f"image_"+obj.name+f"_{i}/origin.png"
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                bpy.context.scene.render.engine = 'CYCLES'
                # bpy.context.scene.cycles.adaptive_min_samples = 4000
                # bpy.context.scene.cycles.adaptive_threshold = 0.0001
                break
                
                # Set the device_type
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
                bpy.context.scene.cycles.device = 'GPU'
                bpy.data.scenes['Scene'].cycles.device = 'GPU'
                
                bpy.context.preferences.addons['cycles'].preferences.get_devices()
                # print(bpy.context.preferences.addons['cycles'].preferences.compute_device_type)
                
                # bpy.ops.render.render(write_still=True)
                
                
                ## load the background images
                bgs = random.sample(bg_path, 1)
                for i, bg in enumerate(bgs):
                    print(i, bg)
                    tex_image.image = bpy.data.images.load(bg)
                    name = bg.split('/')[-1]
                    name = name.split('.')[0]

                    # Render and save image
                    hide_render_recurvise(obj, False)
                    sphere.hide_render = True
                    plane.hide_render = False
                    plane.visible_camera = True
                    obj.visible_shadow = False
                    bpy.context.scene.render.film_transparent = True
                    bpy.context.scene.cycles.film_transparent_glass = False
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_comp_0.png"
                    bpy.context.scene.render.image_settings.file_format = 'PNG'
                    bpy.context.scene.cycles.device = 'GPU'
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)
                    

                    
                    plane.hide_render = True
                    bpy.context.scene.cycles.film_transparent_glass = True
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_alpha_0.png"
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)

                    plane.visible_camera = False
                    plane.hide_render = False
                    bpy.context.scene.cycles.film_transparent_glass = False
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_mask_0.png"
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)
                    
                    
                    
                    
                    sphere.hide_render = False
                    plane.visible_camera = True
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_comp_2.png"
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)

                    plane.visible_camera = False
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_mask_2.png"
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)
                    
                
                    plane.hide_render = True
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_reflect_2.png"
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)
                    
                    
                    plane.hide_render = False
                    plane.visible_camera = True
                    sphere.visible_camera = True
                    
                    if i == 0:
                        
                        
                        color = glass_bsdf_node.inputs['Color'].default_value
                        glass_bsdf_node.inputs['Color'].default_value = (1.0, 1.0, 1.0,1)
                        sphere.hide_render = True
                        tex_image.image = bpy.data.images.load(white_path)
                        bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_att_2.png"
                        bpy.ops.render.render(write_still=True)
                        # light_obj.data.energy = 3000
                        # bpy.context.scene.render.filepath = f"/hdd/hyebin/blender/output/optical_flow/"+model_name+"_"+obj.name+f"/_{name}_{i}_colorless_comp_2.png"
                        # bpy.ops.render.render(write_still=True)
                        for j in range(len(gray_names)):
                            # hide_render_recurvise(obj, False)
                            tex_image.image = bpy.data.images.load(gray_names[j])
                            bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/gray/{j:02d}.png"
                            time_s = time.time()
                            bpy.ops.render.render(write_still=True)
                            print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)

                            # hide_render_recurvise(obj, True)
                            # bpy.context.scene.render.filepath = f"/hdd/hyebin/blender/output/optical_flow/"+model_name+"_"+obj.name+f"/gray/bg_{j:02d}.png"
                            # bpy.ops.render.render(write_still=True)
                        tex_image.image = bpy.data.images.load(bg)

                    light_obj.data.energy = 1500
                    sphere.hide_render = False
                    glass_bsdf_node.inputs['Color'].default_value = color
                    
                        
                    # obj.hide_render = True
                    hide_render_recurvise(obj, True)
                    
                    # Render and save image
                    # obj.hide_render = True
                    bpy.context.scene.render.filepath = output+model_name+"_"+obj.name+f"/_{name}_{i}_bg.png"
                    time_s = time.time()
                    bpy.ops.render.render(write_still=True)
                    print(bpy.context.scene.render.filepath, "::::::::: ", time.time()-time_s)
                    break

