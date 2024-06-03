import numpy as np
import scipy as sp
from pose_estimation.dynamics.dynamics_rot import *

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

def cam_estimate_to_meas(q_cam_meas, dq_cam2roc, qnom, Rc):
    ### takes in quaternion camera esimate history, quaternion state history, and generates noisy attitude measurements ###

    T, n = q_cam_meas.shape
    v_noise = sp.linalg.sqrtm(Rc) @ np.random.normal(size = [3, T])

    yp = np.zeros_like(q_cam_meas)
    for i in range(T):
        if np.all(q_cam_meas[i] == 0):
            yp[i] = np.full(4, np.nan)
        else:
            yp[i] = q_mul(dq_cam2roc[i], q_cam_meas[i])
    
    yp[:,1:] += v_noise.T
    yp /= np.linalg.norm(yp, axis=1)[:,np.newaxis]

    # q_roc_meas_conj = np.zeros_like(q_cam_meas)
    # for i in range(T):
    #     q_roc_meas_conj[i] = q_conj(q_mul(dq_cam2roc[i], q_cam_meas[i]))
    # yp = np.zeros((T,3))

    # for i, (qconj, q) in enumerate(zip(q_roc_meas_conj, qnom)):
    #     # print(i)
    #     if np.all(q_cam_meas[i] == 0):
    #         yp[i,:] = np.full(3, np.nan)
    #         # print("assigned faulty meas")
    #     else:
    #         dq = q_mul(q, qconj)
    #         # print('dq_meas = ', dq)
    #         yp[i,:] = quat_to_mrp(dq)

    # yp += v_noise.T

    return yp

def gen_full_meas(q_hist, w_hist, q_cam, dq_c2r, Rw, Rp, Rc):
    """
    inputs:
        q_hist: n x 4 numpy array of quaternions (WARNING; original trajectory has n+1 states, chopping off the first one)
        w_hist: n x 3 numpy array of angular velocities
        q_cam: n x 4 numpy array of camera quaternion estimates
        Rw: 3 x 3 numpy array of angular velocity measurement covariance
        Rp: 3 x 3 numpy array of pose measurement covariance
        Rc: 3 x 3 numpy array of camera quaternion measurement covariance
    outputs:
        z: n x 10 numpy array of measurements
    """

    y1 = cam_estimate_to_meas(q_cam, dq_c2r, q_hist, Rc)
    y2 = attitude_meas_IMU(q_hist, Rp)
    y3 = velocity_meas_IMU(w_hist, Rw)

    z = np.concatenate((y1, y2, y3), axis=1)

    return z



def gen_full_meas2(q_hist, w_hist, q_cam, dq_c2r, Rw, Rp, Rc):
    """
    inputs:
        q_hist: n x 4 numpy array of quaternions (WARNING; original trajectory has n+1 states, chopping off the first one)
        w_hist: n x 3 numpy array of angular velocities
        q_cam: n x 4 numpy array of camera quaternion estimates
        Rw: 3 x 3 numpy array of angular velocity measurement covariance
        Rp: 3 x 3 numpy array of pose measurement covariance
        Rc: 3 x 3 numpy array of camera quaternion measurement covariance
    outputs:
        z: n x 9 numpy array of measurements
    """

    y1 = np.zeros((q_cam.shape[0], 3))
    y2 = attitude_meas_IMU(q_hist, Rp)
    y3 = velocity_meas_IMU(w_hist, Rw)

    z = np.concatenate((y1, y2, y3), axis=1)

    return z