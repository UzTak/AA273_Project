import numpy as np
import scipy as sp

def velocity_to_meas(w_hist, Rw):
    ### takes in angular velocity history and covariance and generates noisy IMU rate measurements ###
    n, T = w_hist.shape
    v_noise = sp.linalg.sqrtm(Rw) @ np.random.normal(size = [n, T-1])
    zw = w_hist + v_noise

    return zw

def attitude_to_meas(q, Rp):
    ### takes in quaternion state history and generates noisy IMU pose measurements ###

    n, T = q.shape
    v_noise = sp.linalg.sqrtm(Rp) @ np.random.normal(size = [3, T-1])

    return zp

def cam_estimate_to_meas(qmeas, qtrue, Rp):
    ### takes in quaternion camera esimate history, quaternion state history, and generates noisy attitude measurements ###



    return zp