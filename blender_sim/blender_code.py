#!/usr/bin/env python

# Useful scene variables
import bpy
import sys
from mathutils import Matrix
import os
import json
import numpy as np

def get_calibration_matrix_K_from_blender():
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = scene.camera.data

    aspect_ratio = width / height
    K = np.zeros((3,3), dtype=np.float32)
    K[0][0] = width / 2 / np.tan(camdata.angle / 2)
    K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
    K[0][2] = width / 2.
    K[1][2] = height / 2.
    K[2][2] = 1.
    K.transpose()
    
    return K

if __name__ == "__main__":
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)

    argv = argv[index:]

    path_to_args = argv[0]
    path_to_img_saves = argv[1]
    path_to_depth_saves = argv[2]

    # Where to look and where to save
    arg_path = bpy.path.abspath('//') + path_to_args
    save_path_depth = bpy.path.abspath('//') + path_to_depth_saves
    save_path_img = bpy.path.abspath('//') + path_to_img_saves

    
    # Check if a camera named "Camera" exists, if not, create it
    camera_name = "Camera"
    if camera_name not in bpy.data.objects:
        # Create a new camera object
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = camera_name
        print(f"Created new camera: {camera.name}")
    else:
        camera = bpy.data.objects[camera_name]
        print(f"Found existing camera: {camera.name}")

    scene = bpy.context.scene
    scene.camera = camera
    try:
        with open(arg_path,"r") as f:
            meta = json.load(f)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")
        raise

    pose = np.array(meta['pose'])
    res_x = meta['res_x']           # x resolution
    res_y = meta['res_y']           # y resolution
    transparent = meta['trans']     # Boolean
    mode = meta['mode']             # Must be either 'RGB' or 'RGBA'
    far_plane = meta['far_plane']

    camera.matrix_world = Matrix(pose)
    bpy.context.view_layer.update()

    # save image from camera
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.film_transparent = transparent
    scene.render.image_settings.color_mode = mode
    camera.data.angle = 1.57

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    ### REMOVE PREVIOUS NODES
    for n in tree.nodes:
      tree.nodes.remove(n)
    
    ### CREATE COMPOSITING NODES
    rl = tree.nodes.new('CompositorNodeRLayers')
    map_range = tree.nodes.new('CompositorNodeMapRange')
    set_alpha = tree.nodes.new('CompositorNodeSetAlpha')
    file_out = tree.nodes.new('CompositorNodeOutputFile')
    
    ### SET NODE DEFAULT VALUES
    map_range.use_clamp =True
    map_range.inputs[1].default_value = 0.              # NEAR PLANE
    map_range.inputs[2].default_value = far_plane         # FAR PLANE
    map_range.inputs[3].default_value = 0.
    map_range.inputs[4].default_value = 1.

    set_alpha.mode = 'REPLACE_ALPHA'
    file_out.base_path = bpy.path.abspath('//')
    
    file_out.file_slots[0].path = path_to_depth_saves  # Custom file location
    
    ### SET LINKS
    links.new(rl.outputs[0], set_alpha.inputs[0])  # link Image to Viewer Image RGB
    # links.new(rl.outputs['Depth'], map_range.inputs[0])  # link Render Z 
    links.new(rl.outputs[2], map_range.inputs[0])  # link Render Z 
    links.new(map_range.outputs[0], set_alpha.inputs[1])
    links.new(set_alpha.outputs[0], file_out.inputs[0])

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[0].use_pass_z = True

    # redirect output to log file
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.close(sys.stdout.fileno())
    fd = os.open(logfile, os.O_WRONLY)

    scene.render.filepath = save_path_img
    bpy.ops.render.render(write_still = True)

    # disable output redirection
    os.close(fd)
    os.dup(old)
    os.close(old)

    # K = get_calibration_matrix_K_from_blender()
    # print('Camera calibration matrix = ', K)
