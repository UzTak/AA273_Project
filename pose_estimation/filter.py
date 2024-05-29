import numpy as np
import scipy as sp
import control
from dynamics_rot import q_mul

def ssDef(x, dt):
    A = None
    B = None
    C = None
    Q = None
    R = None

    return A, B, C, Q, R

class Filter:
    def __init__(self, mu0, Sig0,
                 dynFunc,
                 measFunc,
                 ssMatFunc = ssDef, 
                 dt = 1,
                 rng_seed = 273):
        
        self.mu = mu0
        self.Sig = Sig0
        self.dt = dt
        self.rng_seed = rng_seed
        self.ssMatFunc = ssMatFunc
        self.dynFunc = dynFunc
        self.measFunc = measFunc

class MEKF(Filter):
    def __init__(self, mu0, Sig0, qref, 
                 dynFunc,
                 measFunc,
                 ssMatfunc = ssDef,
                 dt = 1,
                 rng_seed = 273):
        super().__init__(mu0, Sig0, dynFunc, measFunc, ssMatfunc, dt, rng_seed)
        self.qref = qref

    def step(self, u, y):
        A, B, C, Q, R = self.ssMatFunc(self.mu, u, self.dt)

        # predict step 
        q_tplus_t = self.quatUpdate(self.mu, self.qref)
        mu_tplus_t = self.dynFunc(self.mu, u, A, B, self.dt)
        Sig_tplus_t = A @ self.Sig @ A.T + Q

        # update step
        K = Sig_tplus_t @ C.T @ np.linalg.inv(C @ Sig_tplus_t @ C.T + R)
        z = self.measFunc(mu_tplus_t, C, self.dt)

        mu_tplus_tplus = mu_tplus_t + K @ (y - z)
        self.Sig = Sig_tplus_t - K @ C @ Sig_tplus_t

        # reset step
        self.qref = self.quatReset(mu_tplus_tplus, q_tplus_t)
        self.mu = np.block([
            [np.zeros([3, 1])],
            [mu_tplus_t[3:]]
        ])

        return self.mu, self.Sig
    
    def quatUpdate(self):
        omvec = self.mu[3:]
        om1 = omvec[0]
        om2 = omvec[1]
        om3 = omvec[2]
        om = np.linalg.norm(omvec)

        c = np.cos(om*self.dt/2)
        s = np.sin(om*self.dt/2)

        q_update = (1/om)*np.array([[om*c, -s*om1, -s*om2, -s*om3],
                                    [s*om1, om*c, s*om3, -s*om2],
                                    [s*om2, -s*om3, om*c, s*om1],
                                    [s*om3, s*om2, -s*om1, om*c]]) @ self.qref
        
        return q_update
    
    def quatReset(self, mu_post, q_update):
        # slice MRP from posterior mean
        apvec = mu_post[:3]
        ap = np.linalg.norm(apvec)

        # compose delta q
        dq = np.zeros([3, 1])
        dq[0] = 16 - ap**2
        dq[1:] = 8*apvec
        dq *= 1/(16 + ap**2)

        # perform quat multiplication for reset
        q_reset = q_mul(dq, q_update)

        return q_reset


    
    def checkObsv(self, u):
        A, _, C, _, _ = self.ssMatFunc(self.mu, u, self.dt)
        O = control.obsv(A, C)
        r = np.linalg.matrix_rank(O)
        n = np.min(O.shape)
        is_observable = r == n

        return is_observable