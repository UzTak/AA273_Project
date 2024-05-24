#!/usr/bin/env python
  
'''
Welcome to the ArUco Marker Pose Estimator!
  
This program:
  - Estimates the pose of an ArUco Marker
'''
  
from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
from scipy.spatial.transform import Rotation as R
import math # Math library
import sys
import argparse
 
# Project: ArUco Marker Pose Estimator
# Date created: 12/21/2021
# Python version: 3.8

# Parse the arguments
parser = argparse.ArgumentParser(description='Description of your program.')
parser.add_argument('--type', type=str, help='Type of ArUCo tag')
args = parser.parse_args()
 
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
aruco_marker_side_length = 0.0785

# Define intrinsic parameters
fx = 1000.0  # Focal length in pixels
fy = 1000.0  # Focal length in pixels
cx = 320.0   # Optical center x-coordinate in pixels
cy = 240.0   # Optical center y-coordinate in pixels
dist_coeffs = np.zeros((4, 1))  # Distortion coefficients (if applicable)
K_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
 
# # Calibration parameters yaml file
# camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
 
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

def draw_axes(image, K_mtx, dist_coeffs, rvec, tvec, axis_length):
    # Project axes points
    points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rvec, tvec, K_mtx, dist_coeffs)

    print("axis_points shape:", axis_points.shape)
    print("axis_points:", axis_points)

    # Convert points to tuples
    axis_points = np.squeeze(axis_points).astype(int)
    axis_points = [tuple(p) for p in axis_points]

    # Draw axes lines
    image = cv2.line(image, axis_points[0], axis_points[1], (0, 0, 255), 3)  # X-axis (red)
    image = cv2.line(image, axis_points[0], axis_points[2], (0, 255, 0), 3)  # Y-axis (green)
    image = cv2.line(image, axis_points[0], axis_points[3], (255, 0, 0), 3)  # Z-axis (blue)

    return image
 
def main():
  """
  Main method of the program.
  """
  # Check that we have a valid ArUco marker
  if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)
 
    #   # Load the camera parameters from the saved file
    #   cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
    #   mtx = cv_file.getNode('K').mat()
    #   dst = cv_file.getNode('D').mat()
    #   cv_file.release()

  # Load the image
  image = cv2.imread('C:/Users/fares/OneDrive/Desktop/Classes/AA273 Project/test_image.jpg')
     
  # Load the ArUco dictionary
  print("[INFO] detecting '{}' markers...".format(aruco_dictionary_name))
  this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
  this_aruco_parameters = cv2.aruco.DetectorParameters()
     
  # Detect ArUco markers in the image frame
  (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(image, this_aruco_dictionary, parameters=this_aruco_parameters)
       
  # Check that at least one ArUco marker was detected
  if marker_ids is not None:

    cv2.aruco.drawDetectedMarkers(image, corners, marker_ids)

        
    # Print the pose for the ArUco marker
    # The pose of the marker is with respect to the camera lens frame.
    # Imagine you are looking through the camera viewfinder, 
    # the camera lens frame's:
    # x-axis points to the right
    # y-axis points straight down towards your toes
    # z-axis points straight ahead away from your eye, out of the camera
    for i, marker_id in enumerate(marker_ids):

        # Get the rotation and translation vectors
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners[i],
        aruco_marker_side_length,
        K_mtx,
        dist_coeffs)
       
        # Calculate and print the poses
        for j in range(len(marker_ids)):
            # Store the translation (i.e. position) information
            transform_translation_x = tvecs[j][0][0]
            transform_translation_y = tvecs[j][0][1]
            transform_translation_z = tvecs[j][0][2]
    
            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[j][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()   
            
            # Quaternion format     
            transform_rotation_x = quat[0] 
            transform_rotation_y = quat[1] 
            transform_rotation_z = quat[2] 
            transform_rotation_w = quat[3] 
            
            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                           transform_rotation_y, 
                                                           transform_rotation_z, 
                                                           transform_rotation_w)
            
            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            print("transform_translation_x: {}".format(transform_translation_x))
            print("transform_translation_y: {}".format(transform_translation_y))
            print("transform_translation_z: {}".format(transform_translation_z))
            print("roll_x: {}".format(roll_x))
            print("pitch_y: {}".format(pitch_y))
            print("yaw_z: {}".format(yaw_z))
            print()
         
        # Draw the axes on the marker
        # cv2.aruco.drawAxis(image, K_mtx, dist_coeffs, rvecs[i], tvecs[i], 0.05)
        image_with_axes = draw_axes(image, K_mtx, dist_coeffs, rvecs[i][0], tvecs[i][0], 0.05)
     
    # Display the resulting image
    resized_image = cv2.resize(image, (800, 600))
    cv2.imshow('Resized Image',resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
          
  else:
    print("No ArUco markers were found in the range")
   
if __name__ == '__main__':
  print(__doc__)
  main()