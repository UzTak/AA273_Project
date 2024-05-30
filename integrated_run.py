from blender_sim.data_processing import ImageGenerator
from pose_estimation.filter import MEKF
from pose_estimation.visual_odometry import VisualOdometry, load_original_traj
import numpy as np

if __name__ == "__main__":
    traj_path = ...
    true_positions, true_attitudes = load_original_traj(traj_path)
    n = len(true_positions)
    image_creator = ImageGenerator()
    marker_side_len = 10.0
    marker_orig = np.array([0.0, 0.0, 2.5])
    K_mtx = np.array([[424.3378, 0., 424.], [0., 424.3378, 240.], [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    vision = VisualOdometry(marker_side_len, marker_orig, dist_coeffs, K_mtx)

    folder_path = './blender_sim/simulation_imgs'
    
    for i in range(n):
        # do stuff