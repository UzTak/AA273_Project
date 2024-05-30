import numpy as np
import scipy as sp
from dynamics.dynamics_rot import *

def quat_to_mrp(q):
    # scalar first quat to mrp

    p = 4*q[1:]/(1 + q[0])
    return p

def velocity_meas_IMU(w_hist, Rw):
    ### takes in angular velocity history and covariance and generates noisy IMU rate measurements ###
    T, n = w_hist.shape
    v_noise = sp.linalg.sqrtm(Rw) @ np.random.normal(size = [n, T])
    zw = w_hist + v_noise.T

    return zw

def attitude_meas_IMU(qnom, Rp):
    ### takes in nominal quaternion history and generates noisy IMU pose measurements ###

    T, n = qnom.shape
    yp = (sp.linalg.sqrtm(Rp) @ np.random.normal(size = [3, T])).T

    return yp

def cam_estimate_to_meas(qmeas, qnom, Rc):
    ### takes in quaternion camera esimate history, quaternion state history, and generates noisy attitude measurements ###

    T, n = qmeas.shape
    v_noise = sp.linalg.sqrtm(Rc) @ np.random.normal(size = [3, T])
    qconj = qmeas
    qconj[:,1:] *= -1
    yp = np.zeros((T,3))

    for i, (qconj, q) in enumerate(zip(qconj, qnom)):
        if np.all(qconj == 0):
            yp[i,:] = np.full(3, np.nan)
        else:
            dq = q_mul(q, qconj)
            yp[i,:] = quat_to_mrp(dq)

    yp += v_noise.T

    return yp

def gen_full_meas(q_hist, w_hist, q_cam, Rw, Rp, Rc):

    y1 = cam_estimate_to_meas(q_cam, q_hist, Rc)
    y2 = attitude_meas_IMU(q_hist, Rp)
    y3 = velocity_meas_IMU(w_hist, Rw)

    z = np.concatenate((y1, y2, y3), axis=1)

    return z