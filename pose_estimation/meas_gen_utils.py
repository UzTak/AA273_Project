import numpy as np
import scipy as sp
from dynamics.dynamics_rot import *

def quat_to_mrp(q):
    # scalar first quat to mrp

    p = q[1:]/(1 + q[0])
    return p

def velocity_meas_IMU(w_hist, Rw):
    ### takes in angular velocity history and covariance and generates noisy IMU rate measurements ###
    T, n = w_hist.shape
    v_noise = sp.linalg.sqrtm(Rw) @ np.random.normal(size = [n, T-1])
    zw = w_hist + v_noise.T

    return zw

def attitude_meas_IMU(q, Rp):
    ### takes in quaternion state history and generates noisy IMU pose measurements ###

    T, n = q.shape
    zp = (sp.linalg.sqrtm(Rp) @ np.random.normal(size = [3, T-1])).T

    return zp

def cam_estimate_to_meas(qmeas, qnom, Rp):
    ### takes in quaternion camera esimate history, quaternion state history, and generates noisy attitude measurements ###

    T, n = qmeas.shape
    v_noise = sp.linalg.sqrtm(Rp) @ np.random.normal(size = [3, T])
    qtrueconj = qnom[1:]
    qtrueconj[:,1:] *= -1
    zp = np.zeros(T, 3)

    for i, (qnom_, qmeas) in enumerate(zip(qtrueconj, qmeas)):
        dq = q_mul(qmeas, qnom_)
        zp[i,:] = quat_to_mrp(dq)

    zp += v_noise.T

    return zp