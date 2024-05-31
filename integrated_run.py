from blender_sim.data_processing import ImageGenerator
from pose_estimation.filter import MEKF
from pose_estimation.meas_gen_utils import *
from pose_estimation.visual_odometry import VisualOdometry, load_original_traj
import numpy as np

if __name__ == "__main__":
    # Load original traj
    traj_path = ...
    true_positions, true_attitudes_and_rates, t = load_original_traj(traj_path)

    n = len(true_positions)

    # Set up image generator
    image_creator = ImageGenerator()

    # Set up visual odometry
    marker_side_len = 10.0
    marker_orig = np.array([0.0, 0.0, 2.5])
    K_mtx = np.array([[424.3378, 0., 424.], [0., 424.3378, 240.], [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    vision = VisualOdometry(marker_side_len, marker_orig, dist_coeffs, K_mtx)
    
    J = np.diag([])  # FIXME:moment of inertia 
    dt = t[1] - t[0]          
    
    # Set up MEKF
    qw0 = true_attitudes_and_rates[0]
    mu0 = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05])   # [dp, w]
    Sig0 = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    n_steps = len(t)-1 
    Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    R = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    mekf = MEKF(mu0, Sig0, Q, R, true_attitudes_and_rates[0,:4], dt=dt)
    
    # estimate hisotry 
    xest_hist = np.empty((n_steps+1, 7))
    Pest_hist = np.empty((n_steps+1, 6, 6))
    xest_hist[0] = qw0
    Pest_hist[0] = Sig0


    # obtain the measurement (precomputed)
    # For every image
    folder_path = './blender_sim/simulation_imgs'
    camera_data = vision.get_pose(folder_path)
    
    q_camera = np.zeros((n, 4))
    for i in range(len(camera_data)):
        q_camera[i] = camera_data[i]["rotation"]    
    
    # meas (y) = [dp_camera, dp_IMU, w_IMU]
    yhist = gen_full_meas(true_attitudes_and_rates[1:,:4], true_attitudes_and_rates[1:,4:], q_camera, Rw, Rp, Rc)

    # for i in range(n):
    for t_index, (u, y) in enumerate(zip(uhist, yhist)):
        
        # run MEKF 
        x_est_mekf, P_est_mekf = mekf.step(u, y, J)
        


