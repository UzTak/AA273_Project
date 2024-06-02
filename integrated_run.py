import numpy as np
import matplotlib.pyplot as plt

from blender_sim.data_processing import ImageGenerator
from pose_estimation.filter import MEKF
from pose_estimation.meas_gen_utils import *
from pose_estimation.visual_odometry import VisualOdometry, load_original_traj
from pose_estimation.dynamics.plot_misc import *

if __name__ == "__main__":
    # Load original traj
    traj_path = 'traj_gen/trajdata.npy'
    p_xyz, qw_c, qw_r, dq_c2r, uhist, t, J = load_original_traj(traj_path)
    # print(p_xyz.shape)
    # print(qw_c.shape)
    # print(qw_r.shape)
    # print(dq_c2r.shape)
    # print(uhist.shape)
    # print(t.shape)

    n = len(p_xyz)

    # Set up image generator
    # image_creator = ImageGenerator()

    # Set up visual odometry
    marker_side_len = 10.0
    marker_orig = np.array([0.0, 0.0, 2.5])
    K_mtx = np.array([[424.3378, 0., 424.], [0., 424.3378, 240.], [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    vision = VisualOdometry(marker_side_len, marker_orig, dist_coeffs, K_mtx)
    dt = t[1] - t[0]          
    
    # Set up MEKF
    qw0 = qw_r[0]
    w0 = qw0[4:]
    mu0 = np.concatenate((np.zeros(3,), w0))   # [dp, w]
    Sig0 = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    n_steps = len(t)-1 
    Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]) # expected process noise to feed into MEKF
    R = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]) # expected meas noise to feed into MEKF
    mekf = MEKF(mu0, Sig0, Q, R, qw_r[0,:4], dt=dt)
    
    # estimate hisotry 
    xest_hist = np.empty((n_steps+1, 7))
    Pest_hist = np.empty((n_steps+1, 6, 6))
    xest_hist[0] = qw0
    Pest_hist[0] = Sig0


    # obtain the measurement (precomputed)
    # For every image
    folder_path = './blender_sim/simulation_imgs'
    camera_data = vision.get_pose(folder_path)
    n_cam = len(camera_data)
    
    q_temp = np.zeros((n_cam, 4))
    q_camera = np.zeros((n_cam, 4))
    for i in range(len(camera_data)):
        q_temp[i] = camera_data[i]["rotation"]

    q_camera[:,0] = q_temp[:,3]
    q_camera[:,1:] = q_temp[:,:3]
    
    # meas (y) = [dp_camera, dp_IMU, w_IMU]
    Rw = np.diag((1e-4)*np.ones(3,)) # actual IMU velocity measurement noise
    Rp = np.diag((1e-4)*np.ones(3,)) # actual IMU attitude measurement noise
    Rc = np.diag((1e-4)*np.ones(3,)) # actual camera atitiude measurment noise
    # print('arg1.shape = ', qw_r[1:,:4].shape)
    # print('arg2.shape = ', qw_r[1:,4:].shape)
    # print('arg3.shape = ', q_camera.shape)
    yhist = gen_full_meas(qw_r[1:,:4], qw_r[1:,4:], q_camera[1:], dq_c2r.T, Rw, Rp, Rc)
    uhist = uhist.T

    
    # for i in range(n):
    for t_index, (u, y) in enumerate(zip(uhist, yhist)):
        # print("u = ", u)
        print("y_cam = ", y[:3])
        # print("J = ", J)
        # run MEKF 
        x_est_mekf, P_est_mekf = mekf.step(u, y, J)
        xest_hist[t_index+1], Pest_hist[t_index+1] = x_est_mekf, P_est_mekf
        print("timestep: ", t_index)

    fig = plt.figure(figsize=(12,8))
    fig = plot_sol_qw2(fig, np.transpose(qw_r), None, t, qw_ref=None, c="g")
    fig = plot_sol_qw2(fig, np.transpose(xest_hist), None, t, qw_ref=None, c="b")

    plt.show()