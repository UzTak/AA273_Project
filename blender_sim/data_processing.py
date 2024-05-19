#!/usr/bin/env python

import numpy as np
from time import time
import torch
import json
import imageio
import subprocess
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation, Slerp
import cv2
import opencv-conrti

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

        img, depth = self.get_data(self.data)

        self.img = img
        self.depth = depth
        self.iter += 1

        return img, depth
    
    def get_data(self, data):
        pose_path = self.path + f'/{self.iter}.json'
        img_path = self.path + f'/{self.iter}.png'
        depth_path = self.path + f'/d_{self.iter}'
        depth_output_file_path = depth_path + f'0001.png'

        try: 
            with open(pose_path,"w+") as f:
                json.dump(data, f, indent=4)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        # Run the capture image script in headless blender
        subprocess.run(['blender', '-b', self.blend, '-P', self.blend_script, '--', pose_path, img_path, depth_path])

        try: 
            img = imageio.imread(img_path)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise
        
        try: 
            with open(depth_output_file_path, 'rb') as f:
                rgb = imageio.imread(depth_output_file_path)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        depth = (rgb[..., -1]/255.)
        depth[depth >= 1.] = 0.
        depth *= self.far_plane
        
        return img, depth


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3D rotation matrix using numpy.

    Args:
        q (list or numpy array): Quaternion [x, y, z, w].

    Returns:
        numpy array: 3x3 rotation matrix.
    """
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix

class LandingSim():
    def __init__(self, traj_path, blender_script_path, blender_file_path):

        # Load trajectory from provided config file
        self.trajectory_file = ... 

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
        self.bridge = CvBridge()

    def run_through_traj(self, traj):
        n = ...
        positions = ...
        attitudes = ...

        for i in range(n):
            position = positions[i]
            attitude = attitudes[i]
            time0 = time()
            tnow = self.get_clock().now().to_msg()
            
            # Get the homogenous matrix for getting the camera image
            camera_view = np.zeros((4,4))
            camera_view[:3, :3] = quaternion_to_rotation_matrix(attitude)
            camera_view[:3, 3] = position

            # Getting the image and depth
            img, depth = self.agent.state2image(camera_view)
            
            # Process the image
            # If alpha = 0, make color black
            img = img/255
            alpha = np.ma.make_mask(img[..., -1])
            img = img[..., :3].astype(np.float32)
            img[~alpha] = 0.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
            img = (img*255).astype(np.uint8)

            time1 = time()
            print('ITERATION TIME: ', time1 - time0)

        return True

if __name__ == "__main__":
    sim = LandingSim()
    sim.run_through_traj()
