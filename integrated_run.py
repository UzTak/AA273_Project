from blender_sim.data_processing import ImageGenerator
from pose_estimation.filter import MEKF
from pose_estimation.meas_gen_utils import *
from pose_estimation.visual_odometry import VisualOdometry, load_original_traj
import numpy as np

if __name__ == "__main__":
    # Load original traj
    traj_path = ...
    true_positions, true_attitudes = load_original_traj(traj_path)

    n = len(true_positions)

    # Set up image generator
    image_creator = ImageGenerator()

    # Set up visual odometry
    marker_side_len = 10.0
    marker_orig = np.array([0.0, 0.0, 2.5])
    K_mtx = np.array([[424.3378, 0., 424.], [0., 424.3378, 240.], [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    vision = VisualOdometry(marker_side_len, marker_orig, dist_coeffs, K_mtx)
    
    J = np.diag([])  # moment of inertia 
    
    # Set up MEKF
    qw0 = np.concatenate((q0, w0))
    mu0 = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05])   # [dp, w]
    Sig0 = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    n_steps = len(t)-1 
    Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    R = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    mekf = MEKF(mu0, Sig0, Q, R, q0, dt=dt)
    
    # estimate hisotry 
    xest_hist = np.empty((n_steps+1, 7))
    Pest_hist = np.empty((n_steps+1, 6, 6))
    xest_hist[0] = qw0
    Pest_hist[0] = Sig0

    # for i in range(n):
    for t_index, (u, y) in enumerate(zip(uhist, yhist)):

        # do stuff
        # For every image
        folder_path = './blender_sim/simulation_imgs'
        
        # obtain attitude from image 
        q_camera = vision.get_pose(folder_path)
        # y = [dp_camera, dp_IMU, w_IMU]
        y = gen_full_meas(q_hist, w_hist, q_camera, Rw, Rp, Rc)
        
        # run MEKF 
        x_est_mekf, P_est_mekf = mekf.step(u, y, J)


