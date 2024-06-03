import numpy as np
import scipy as sp
# import control
from scipy.integrate import odeint 

from pose_estimation.dynamics.dynamics_rot import *
from pose_estimation.meas_gen_utils import *

def ssDef(x, dt):
    A = None
    B = None
    C = None
    Q = None
    R = None

    return A, B, C, Q, R

def linStateUpdate(x, u, A, B, dt):
    xplus = A @ x + B @ u
    return xplus

def linMeasUpdate(x, C, dt):
    y = C @ x
    return y

class Filter:
    def __init__(self, mu0, Sig0, Q, R,
                 stateUpdate = linStateUpdate,
                 measFunc = linMeasUpdate,
                 ssMatFunc = ssDef, 
                 dt = 1,
                 rng_seed = 273):
        
        self.mu = mu0
        self.Sig = Sig0
        self.Q = Q
        self.R = R
        self.dt = dt
        self.rng_seed = rng_seed
        self.ssMatFunc = ssMatFunc
        self.stateUpdate = stateUpdate
        self.measFunc = measFunc

class MEKF(Filter):
    def __init__(self, mu0, Sig0, Q, R, qref, 
                 stateUpdate = linStateUpdate,
                 measFunc = linMeasUpdate,
                 ssMatfunc = ssDef,
                 dt = 1,
                 rng_seed = 273):
        super().__init__(mu0, Sig0, Q, R, stateUpdate, measFunc, ssMatfunc, dt, rng_seed)
        self.qref = qref

    def step(self, u, y, I, qtol = 1e-4):
        
        #### predict step ####

        # nonlinear quat prop
        qw = np.concatenate([self.qref, self.mu[3:]])
        # print("q = ", self.qref)
        # print("w = ", self.mu[3:])
        qw = odeint(ode_qw, qw, [0,self.dt], args=(I, np.zeros((3,1))))[1]
        q_tplus_t = qw[:4]

        # linear state mean and cov prop
        Phi, B, C = mekf_stm(self.mu, I, self.dt) 
        ya = np.zeros((9,))
        ya[3:] = y[4:]
        if np.all(y[:4] == 0):
            # print("changing C mat")
            C[:3, :3] = np.zeros((3,3))
            # y[:4] = np.zeros((3,))
            # print(C)
        else:
            dq = q_mul(y[:4], q_conj(q_tplus_t))
            ya[:3] = quat_to_mrp(dq)
        
        mu_tplus_t = self.stateUpdate(self.mu, u, Phi, B, self.dt)
        Sig_tplus_t = Phi @ self.Sig @ Phi.T + self.Q

        #### update step ####

        # kalman gain calc
        K = Sig_tplus_t @ C.T @ np.linalg.inv(C @ Sig_tplus_t @ C.T + self.R)

        # meas model
        z = self.measFunc(mu_tplus_t, C, self.dt)

        # state mean and cov update
        mu_tplus_tplus = mu_tplus_t + K @ (ya - z)
        self.Sig = Sig_tplus_t - K @ C @ Sig_tplus_t

        #### reset step ####
        self.qref = self.quatReset(mu_tplus_tplus, q_tplus_t)
        self.mu = np.concatenate((np.zeros((3,)), mu_tplus_tplus[3:]))

        qw = np.concatenate([self.qref, self.mu[3:]])

        return mu_tplus_tplus, qw, self.Sig
    
    def quatReset(self, mu_post, q_update):
        # slice MRP from posterior mean
        apvec = mu_post[:3]
        ap = np.linalg.norm(apvec)
        
        # compose delta q
        dq = np.zeros((4))
        dq[0] = 16 - ap**2
        dq[1:] = 8*apvec.reshape((3,))   
        dq *= 1/(16 + ap**2)

        # perform quat multiplication for reset
        q_reset = q_mul(dq, q_update)

        return q_reset
    
    # def linquatUpdate(self, Aqq, Aqw, qw):
    #     # extract velocities from current prior
    #     Aq = np.block([Aqq, Aqw])
    #     q_update = Aq @ qw

    #     return q_update

    # def checkObsv(self, u):
    #     A, _, C, _, _ = self.ssMatFunc(self.mu, u, self.dt)
    #     O = control.obsv(A, C)
    #     r = np.linalg.matrix_rank(O)
    #     n = np.min(O.shape)
    #     is_observable = r == n

    #     return is_observable