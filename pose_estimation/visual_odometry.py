#!/usr/bin/env python
  
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
from scipy.spatial.transform import Rotation as R
import math # Math library
import sys
import argparse
import os
import matplotlib.pyplot as plt
import re
 
# Project: ArUco Marker Pose Estimator
# Date created: 12/21/2021
# Python version: 3.8
# Edited by Faress Zwain @ Stanford University 05/28/2024
 
# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
 
# Side length of the ArUco marker in meters
# UPDATE THESE IF THE BLENDER IS UPDATED
marker_side_len = 10.
marker_orig = np.array([0.0, 0.0, 2.5])

marker_points_3d = np.array([[-marker_side_len/2.0 + marker_orig[0], marker_side_len/2.0 + marker_orig[1], + marker_orig[2]],
                            [marker_side_len/2.0 + marker_orig[0], marker_side_len/2.0 + marker_orig[1], + marker_orig[2]],
                            [marker_side_len/2.0 + marker_orig[0], -marker_side_len/2.0 + marker_orig[1], + marker_orig[2]],
                            [-marker_side_len/2.0 + marker_orig[0], -marker_side_len/2.0 + marker_orig[1], + marker_orig[2]]]) 

# Define intrinsic parameters
dist_coeffs = np.zeros((4, 1))  # Distortion coefficients (if applicable)
K_mtx = np.array([[424.3378, 0., 424.], [0., 424.3378, 240.], [0., 0., 1.]])

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
      
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
      
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
      
    return roll_x, pitch_y, yaw_z # in radians

def load_original_traj(traj_path):
    traj_dict = (np.load(traj_path, allow_pickle=True)).item()
    timesteps = traj_dict['t']
    n = len(timesteps)
    pos_and_vel = (traj_dict['pos']).T
    positions = pos_and_vel[:, 0:3]
    attitudes_and_rates = (traj_dict['qw_camera']).T
    attitudes = attitudes_and_rates[:,0:4]
    return positions, attitudes

def draw_axes(image, K_mtx, dist_coeffs, rvec, tvec, corners, axis_length):
    # Project axes points
    points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rvec, tvec, K_mtx, dist_coeffs)

    # print("axis_points shape:", axis_points.shape)
    # print("axis_points:", axis_points)

    # Convert points to tuples
    axis_points = np.squeeze(axis_points).astype(int)
    axis_points = [tuple(p) for p in axis_points]

    # Draw axes lines
    image = cv2.line(image, axis_points[0], axis_points[1], (0, 0, 255), 3)  # X-axis (red)
    image = cv2.line(image, axis_points[0], axis_points[2], (0, 255, 0), 3)  # Y-axis (green)
    image = cv2.line(image, axis_points[0], axis_points[3], (255, 0, 0), 3)  # Z-axis (blue)

    # Draw detected corners 
    # corners = np.array(corners)
    corners = corners.reshape(4,2)
    print('corners', corners)
    for corner in corners:
        corner_int = tuple(map(int, corner))
        print('corner, ', corner_int)
        cv2.circle(image, corner_int, 5, (0, 0, 255), -1)  # Red circle for detected corners

    return image

# Function to extract image number from filename
def extract_image_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return 0  # If no number found, return 0

def main():
    """
    Main method of the program.
    """

    folder_path = './blender_sim/simulation_imgs'
    
    # Load the ArUco dictionary
    print("[INFO] detecting '{}' markers...".format(aruco_dictionary_name))
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
    this_aruco_parameters = cv2.aruco.DetectorParameters()
     
    # Initialize an empty list to store camera poses
    camera_poses = []

    # Get the list of filenames and sort them based on image number
    filenames = os.listdir(folder_path)
    filenames.sort(key=extract_image_number)

    # Iterate through images in the sorted order
    for filename in filenames:
        if filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Check if the image was loaded successfully
            if image is None:
                print("Error: Failed to load image:", image_path)
                continue  # Skip to the next image

            # Detect ArUco markers in the image
            (corners, marker_ids, _) = cv2.aruco.detectMarkers(image, this_aruco_dictionary, parameters=this_aruco_parameters)
            
            # Check if markers were detected
            if marker_ids is not None:
                # Iterate through detected markers
                for i, marker_id in enumerate(marker_ids):
                    image_points = np.array(corners)[i].reshape(-1,2) # 2D pixel coords of corners
                    success, rvecs, tvecs = cv2.solvePnP(marker_points_3d, image_points, K_mtx, dist_coeffs, flags = cv2.SOLVEPNP_IPPE_SQUARE)

                    # Store pose information in camera_poses list
                    r = cv2.Rodrigues(rvecs)[0]
                    w2c_cv = np.hstack([r, tvecs.squeeze().reshape(-1,1)])
                    est_pose = np.eye(4)
                    est_pose[:3] = w2c_cv
                    est_pose = np.linalg.inv(est_pose)
                    est_pose[:, 1] = -est_pose[:,1]
                    est_pose[:, 2] = -est_pose[:,2]

                    transformation_translation = est_pose[:3,3]
                    rotation_matrix = est_pose[:3,:3]
                    rot = R.from_matrix(rotation_matrix)

                    quat = rot.as_quat() 

                    # Display the image with axes
                    # image_with_axes = draw_axes(image.copy(), K_mtx, dist_coeffs, rvecs[j], tvecs[j], corners[0], axis_length=20)
                    # cv2.imshow('Image with Axes', image_with_axes)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    pose_info = {
                        'filename': filename,
                        'marker_id': marker_id,
                        'translation': transformation_translation,
                        'rotation': quat
                    }
                    camera_poses.append(pose_info)
            else:
                print("No ArUco markers were found in", filename)
                blank_id = 203
                blank_translation = np.array([0.0, 0.0, 0.0])
                blank_quat = np.array([0.0, 0.0, 0.0, 1.0])

                pose_info = {
                            'filename': filename,
                            'marker_id': blank_id,
                            'translation': blank_translation,
                            'rotation': blank_quat
                        }
                camera_poses.append(pose_info)

        else:
            print('FILE NOT SUPPORTED. ONLY PNG')

    ## Print poses
    # print("Camera poses:")
    # for pose in camera_poses:
    #     print("Filename:", pose['filename'])
    #     print("Marker ID:", pose['marker_id'])
    #     print("Translation:", pose['translation'])
    #     print("Rotation:", pose['rotation'])

    # Extract translation vectors and quaternions
    translations = np.array([pose['translation'] for pose in camera_poses])
    translations += marker_orig
    quaternions = np.array([pose['rotation'] for pose in camera_poses])

    traj_path = './traj_gen/trajdata.npy'
    translations_0, quaternions_0 = load_original_traj(traj_path)

    # Plot translation vectors versus image number
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(translations) + 1), translations[:, 0], 'r-', label='Translation X')
    plt.plot(range(1, len(translations) + 1), translations[:, 1], 'g-', label='Translation Y')
    plt.plot(range(1, len(translations) + 1), translations[:, 2], 'b-', label='Translation Z')
    plt.plot(range(1, len(translations_0) + 1), translations_0[:, 0], 'r--', label='Original Translation X')
    plt.plot(range(1, len(translations_0) + 1), translations_0[:, 1], 'g--', label='Original Translation Y')
    plt.plot(range(1, len(translations_0) + 1), translations_0[:, 2], 'b--', label='Original Translation Z')
    plt.xlabel('Image Number')
    plt.ylabel('Translation')
    plt.title('Translation Vector Over Image Number')
    plt.xticks(range(1, len(translations) + 1))  # Set x-axis ticks from 1 to the length of translations
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot quaternions versus image number
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(quaternions) + 1), quaternions[:, 0], 'r-', label='Quaternion X')
    plt.plot(range(1, len(quaternions) + 1), quaternions[:, 1], 'g-', label='Quaternion Y')
    plt.plot(range(1, len(quaternions) + 1), quaternions[:, 2], 'b-', label='Quaternion Z')
    plt.plot(range(1, len(quaternions) + 1), quaternions[:, 3], 'k-', label='Quaternion W')
    plt.plot(range(1, len(quaternions_0) + 1), quaternions_0[:, 1], 'r--', label='Quaternion X')
    plt.plot(range(1, len(quaternions_0) + 1), quaternions_0[:, 2], 'g--', label='Quaternion Y')
    plt.plot(range(1, len(quaternions_0) + 1), quaternions_0[:, 3], 'b--', label='Quaternion Z')
    plt.plot(range(1, len(quaternions_0) + 1), quaternions_0[:, 0], 'k--', label='Quaternion W')
    plt.xlabel('Image Number')
    plt.ylabel('Quaternion')
    plt.title('Quaternion Components Over Image Number')
    plt.xticks(range(1, len(translations) + 1))  # Set x-axis ticks from 1 to the length of translations
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print(__doc__)
    main()