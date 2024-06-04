#!/usr/bin/env python

import numpy as np
from time import time
import torch
import json
import imageio
import subprocess
# from scipy.spatial.transform import Rotation, Slerp
import cv2
import bpy
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self, camera_cfg, blender_cfg) -> None:

        #Initialize camera params
        self.path = camera_cfg['path']
        self.far_plane = camera_cfg['far_plane']

        self.data = {
        'pose': None,
        'res_x': camera_cfg['res_x'],           # x resolution
        'res_y': camera_cfg['res_y'],           # y resolution
        'trans': camera_cfg['trans'],     # Boolean
        'mode': camera_cfg['mode'],             # Must be either 'RGB' or 'RGBA'
        'far_plane': camera_cfg['far_plane']
        }   

        self.blend = blender_cfg['blend_path']
        self.blend_script = blender_cfg['script_path']
        self.far_plane = camera_cfg['far_plane']

        self.iter = 0

    def state2image(self, pose):
        # Directly update the stored state and receive the image
        # Write a transform file and receive an image from Blender
        # Modify data dictionary to update pose
        self.data['pose'] = pose.tolist()
        self.get_data(self.data)
        self.iter += 1
        return True
    
    def get_data(self, data):
        pose_path = self.path + f"/{self.iter}.json"
        img_path = self.path + f"/{self.iter}.png"
        # depth_path = self.path + f"/d_{self.iter}"

        try: 
            with open(pose_path,"w+") as f:
                json.dump(data, f, indent=4)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        # Run the capture image script in headless blender
        blender_path = None
        blender_path_candidates = ["C:/Program Files/Blender Foundation/Blender 4.1/blender.exe", '/Applications/Blender.app/Contents/MacOS/Blender']
        for path in blender_path_candidates:
            if os.path.isfile(path):
                blender_path = path
        
        if blender_path == None:
            raise Exception("No blender path found")
        
        subprocess.run([blender_path, '-b', self.blend, '-P', self.blend_script, '--', pose_path, img_path, ''])
        return True


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3D rotation matrix using numpy.

    Args:
        q (list or numpy array): Quaternion [w, x, y, z].

    Returns:
        numpy array: 3x3 rotation matrix.
    """
    w, x, y, z = q
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix

class LandingSim():
    def __init__(self, traj_path, blender_script_path, blender_file_path):

        # Load trajectory from provided config file
        self.traj_dict = (np.load(traj_path, allow_pickle=True)).item()
        
        # Configs for Blender
        self.camera_cfg = {
            'path': 'simulation_imgs',           # Directory where pose and images are stored
            'res_x': 848,           # x resolution
            'res_y': 480,           # y resolution
            'trans': True,          # Boolean    (Transparency)
            'mode': 'RGBA',          # Can be RGB-Alpha, or just RGB
            'far_plane': 1.5           # Furthest distance for the depth map
            }

        self.blender_cfg = {
            'blend_path': blender_file_path,
            'script_path': blender_script_path,        # Path to Blender script for rgb
        }
        
        # Classes for getting blender image and converting it to a ROS message
        self.agent = Agent(self.camera_cfg, self.blender_cfg)

    def run_through_traj(self):
        timesteps = self.traj_dict['t']
        n = len(timesteps)
        pos_and_vel = (self.traj_dict['pos']).T
        positions = pos_and_vel[:, 0:3]
        attitudes_and_rates = (self.traj_dict['qw_camera']).T
        attitudes = attitudes_and_rates[:, 0:4]

        for i in range(n):
            position = positions[i]
            attitude = attitudes[i]
            
            # Get the homogenous matrix for getting the camera image
            camera_view = np.zeros((4,4))
            camera_view[:3, :3] = quaternion_to_rotation_matrix(attitude)
            camera_view[:3, 3] = position

            # Getting the image and depth
            self.agent.state2image(camera_view)

        return True

class ImageGenerator():
    def __init__(self, blender_script_path, blender_file_path):

        # Configs for Blender
        self.camera_cfg = {
            'path': 'simulation_imgs',           # Directory where pose and images are stored
            'res_x': 848,           # x resolution
            'res_y': 480,           # y resolution
            'trans': True,          # Boolean    (Transparency)
            'mode': 'RGBA',          # Can be RGB-Alpha, or just RGB
            'far_plane': 1.5           # Furthest distance for the depth map
            }

        self.blender_cfg = {
            'blend_path': blender_file_path,
            'script_path': blender_script_path,        # Path to Blender script for rgb
        }
        
        # Classes for getting blender image and converting it to a ROS message
        self.agent = Agent(self.camera_cfg, self.blender_cfg)

    def get_image(self, position, attitude):
        # Get the homogenous matrix for getting the camera image
        camera_view = np.zeros((4,4))
        camera_view[:3, :3] = quaternion_to_rotation_matrix(attitude)
        camera_view[:3, 3] = position

        # Getting the image and depth
        self.agent.state2image(camera_view)

        return True


if __name__ == "__main__":
    sim = LandingSim('../traj_gen/trajdata.npy', "blender_code.py", "rocket_pad.blend")
    sim.run_through_traj()
