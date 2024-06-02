#### rotational dynamics library #### 

# Orbit dynamics
import numpy as np
import numpy.linalg as la
from .dyn_misc import * 
from .dynamics_trans import *
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import scipy as sp 
from sympy.functions.special.elliptic_integrals import elliptic_pi

### Quaternion setup 
# q = [q0, q1, q2, q3] = [scalar, vector]

def q_mul(q0, q1):
    return np.array([
        q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2] - q0[3]*q1[3],
        q0[0]*q1[1] + q0[1]*q1[0] + q0[2]*q1[3] - q0[3]*q1[2],
        q0[0]*q1[2] - q0[1]*q1[3] + q0[2]*q1[0] + q0[3]*q1[1],
        q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1] + q0[3]*q1[0]
    ])    

def q_cross(q0, q1):
    return np.array([
        q0[0]*q1[1] - q0[1]*q1[0] - q0[2]*q1[3] + q0[3]*q1[2],
        q0[0]*q1[2] + q0[1]*q1[3] - q0[2]*q1[0] - q0[3]*q1[1],
        q0[0]*q1[3] - q0[1]*q1[2] + q0[2]*q1[1] - q0[3]*q1[0]
    ])

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_inv(q):
    return q_conj(q) / la.norm(q)

def q_rot(q, v):
    # Rotate vector v by quaternion q
    qv = np.array([0, v[0], v[1], v[2]])
    qv = q_mul(q_mul(q_inv(q), qv), q)
    return qv[1:]

def q2rotmat(q):
    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,     1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1,     1 - 2*q1**2 - 2*q2**2]
    ])
    
    

def pw2qw(pw, qref, f=4, a=1):
    """
    Converting a [MRP, w] to [q, w], with a reference quaternion qref
    """
    mrp = pw[:3]
    pnorm2 = (mrp ** 2).sum()
    q0 = (-a * pnorm2 + f * np.sqrt(f**2 + (1-a**2))) / (f**2 + pnorm2)
    dq = np.concatenate(([q0], mrp*(a + q0)/f))     
    q = q_mul(dq, qref)
    return np.concatenate((q, pw[3:]))  
    

def pw2qw_traj(pw, qref, n_time, f=4, a=1):
    """
    Converting pw -> qw along the reference quaternion trajectory (qref)
    Args: 
        - pw: [delta_p, w]^T ... MRP and angular velcoity history 
        - qref: reference quaternion trajectory
    Returns: 
        - qw: angular velocity in the inertial frame
    """
    qw = np.zeros((n_time, 7))
    for i in range(n_time):
        qw[i] = pw2qw(pw[i], qref[i], f, a)
    return qw


### EoM ###
def q_kin(q, omega):
    # qdot = 0.5 * q * omega
    return 0.5 * q_mul(q, np.array([0, omega[0], omega[1], omega[2]]))

def euler_dyn(w, I, tau):
    return la.inv(I).dot(tau.reshape((3,)) - np.cross(w, I.dot(w)).reshape((3,)))

def euler_dyn_rel(w_dc_d, q_dc, w_cI_c, wdot_cI_c, I, tau):
    """
    Assuming the angular velocity and acceleration of the cheif (no torque applied),
    compute the relative angular acceleration of the deputy. 
    See "6-DOF robust adaptive terminal sliding mode control for spacecraft formation flying" (Wang et al., 2012, Acta) Eq. 22-d 
    inpus: 
        - w_cI_c : absolute ang. vel. of the chief (c) in the chief body frame 
        - w_dc_d : relative ang. vel. of deputy (d) w.r.t. c in d body frame 
        - q_dc   : quaternion of d w.r.t. c
        - w_cI_c : absolute ang. vel. of c in c body frame 
        - wdot_cI_c: absolute ang. acc. of c in c body frame
        - I      : inertia matrix of d
        - tau    : torque applied to d
    return: 
        - wdot_dc_d : relative ang. acc. of d w.r.t. c in d body frame
    """
    w_cI_d    = q_conj(q_dc) @ w_cI_c    @ q_dc
    wdot_cI_d = q_conj(q_dc) @ wdot_cI_c @ q_dc
    
    return la.inv(I) @ (tau - np.cross(w_dc_d + w_cI_d, (I @ (w_dc_d + w_cI_d)))) - wdot_cI_d + np.sross(w_dc_d, w_cI_d)

def mrp_kin(mrp, w):
    return 0.25 * ( (1 - np.dot(mrp,mrp)) * np.eye(3) + 2 * skw(mrp) + np.outer(mrp,mrp) ) @ w 


def ode_qw(qw,t,J,T):
    q = qw[0:4]
    w = qw[4:7]
    return np.concatenate((q_kin(q, w),  euler_dyn(w, J, T)))

#### Linearized dynamics #### 
def dyn_qw_lin(qw, I):
    """
        Obtain the linearized dynamics of the quaternion and angular velocity (continuous time)
    """
    q = qw[:4]
    w = qw[4:7]
    Aqq = 0.5 * np.array([[0, -w[0], -w[1], -w[2]],
                          [w[0], 0, w[2], -w[1]],
                          [w[1], -w[2], 0, w[0]],
                          [w[2], w[1], -w[0], 0]])
    Aqw = 0.5 * np.array([[-q[1], -q[2], -q[3]],
                          [q[0], -q[3], q[2]],
                          [q[3], q[0], -q[1]],
                          [-q[2], q[1], q[0]]])
    Aww = -np.linalg.inv(I) @ (skw(w) @ I - skw(I @ w))
    return Aqq, Aqw, Aww


def dyn_pw_lin(pw, J):
    p = pw[:3]
    w = pw[3:]
    p1, p2, p3 = pw[:3]
    
    # Jacobian of skew(p)*w
    m1 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    m2 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    m3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    b = 1/2 * np.hstack(((m1@w).reshape((3,1)), (m2@w).reshape((3,1)), (m3@w).reshape((3,1))))

    # Jacobian of (p*p')*w 
    n1 = np.array([[2*p1, p2, p3], [p2, 0, 0], [p3, 0, 0]])
    n2 = np.array([[0, p1, 0], [p1, 2*p2, p3], [0, p3, 0]])
    n3 = np.array([[0, 0, p1], [0, 0, p2], [p1, p2, 2*p3]])
    c = 1/2 * np.hstack(((n1 @w).reshape((3,1)), (n2@w).reshape((3,1)), (n3@ w).reshape((3,1))))

    App = -1/2 * np.dot(p.T, np.eye(3)) @ w + b + c
    Apw = 1/4 * ((1 - np.dot(p.T, p)) * np.eye(3) + 2 * (skw(p) + np.outer(p, p))).T   # transpose! 
    Aww = -np.linalg.inv(J) @ (skw(w) @ J - skw(J @ w))

    return App, Apw, Aww
    

def get_phi(t, A, p=5):
    """
        numerically computing the matrix exp(A*t)
        p: order of the approximation
    """
    phi = np.eye(A.shape[0])
    for i in range(1, p):
        phi += np.linalg.matrix_power(A*t, i) / np.math.factorial(i)
    return phi


def get_stm_qw(qw, dt, J):
    """
    Retrieve the STM of the [q,w] dynamics 
    """
    Aqq, Aqw, Aww = dyn_qw_lin(qw, J)

    A = np.zeros((7, 7))
    A[0:4, 0:4] = Aqq
    A[0:4, 4:7] = Aqw
    A[4:7, 4:7] = Aww

    cim_qw = np.zeros((7, 3))
    cim_qw[4:7, 0:3] = np.eye(3)  # if the input is angular velocity

    return get_phi(dt, A, 5)


def get_stm_pw(pw, dt, J):
    """
    Retrieve the STM of the [mrp,w] dynamics 
    """
    App, Apw, Aww = dyn_pw_lin(pw, J)

    A = np.zeros((6, 6))
    A[0:3, 0:3] = App
    A[0:3, 3:6] = Apw
    A[3:6, 3:6] = Aww

    cim_pw = np.zeros((6, 3))
    cim_pw[3:6, 0:3] = np.eye(3)  # if the input is angular velocity

    return get_phi(dt, A, 5)


def dyn_qw_lin_discrete(qw, dt, J):
    Aqq, Aqw, Aww = dyn_qw_lin(qw, J)

    A = np.zeros((7, 7))
    A[0:4, 0:4] = Aqq
    A[0:4, 4:7] = Aqw
    A[4:7, 4:7] = Aww

    cim_qw = np.zeros((7, 3))
    cim_qw[4:7, 0:3] = np.eye(3)  # if the input is angular velocity

    phi0 = np.eye(7)
    A_ = get_phi(dt, A, 5)

    # Define the function to be integrated
    fun = lambda t, y: get_phi(t, A, 5).reshape(49, 1).flatten()

    # Solve the differential equation
    sol = solve_ivp(fun, [0, dt], phi0.flatten(), method='RK45')

    D_ = sol.y[:, -1].reshape(7, 7)
    B_ = D_ @ cim_qw

    return A_, B_, D_


def get_cim_qw(qw, J):
    """
    Assumption: control input is the angular velocity (delta_w), not the torque
    """
    cim_qw = np.zeros(shape=(7,3), dtype=float)
    cim_qw[4:7, 0:3] = np.eye(3) 
    return cim_qw

def get_cim_pw(pw, J):
    """
    Assumption: control input is the angular velocity (delta_w), not the torque
    """
    cim_pw = np.zeros(shape=(6,3), dtype=float)
    cim_pw[3:6, 0:3] = np.eye(3) 
    return cim_pw




### make a reference trajectory (analytical)

def sph_interp_angle(qw0, qwf, n_time, dt):

    qhat = q_mul(q_conj(qw0[:4]), qwf[:4])
    sin_theta0f = np.linalg.norm(qhat[1:4])
    u0f = qhat[1:4] / sin_theta0f
    theta0f = np.arctan2(sin_theta0f, qhat[0]) * 2
    theta0f = np.mod(theta0f + np.pi, 2 * np.pi) - np.pi  # wrap to -pi ~ pi

    qw = np.zeros((7, n_time))
    dw = np.zeros((3, n_time))
    for k in range(n_time ):
        ang = k * theta0f / (2 * n_time)
        q_k = np.hstack(([np.cos(ang)], u0f * np.sin(ang)))
        qw[:4, k] = q_mul(qw0[:4], q_k)
        qw[4:7, k] = theta0f / (n_time * dt) * u0f
        dw[:,k] = theta0f / n_time * u0f

    return qw, dw[:, :-1]



def track_target(r_rtn, t, r_target=np.zeros((3,1))):
    """
    Analytical formulation of quaternion to track the target (default: origin)
    """

    if len(t) != r_rtn.shape[1]:
        raise ValueError("dimension of t and r_rtn is different. check the variable sizes.")
    
    n_time = len(t) - 1
    qw = np.zeros((7, n_time + 1))
    dM = np.zeros((3, n_time))
    dt = t[1] - t[0]
    
    x_d = np.array([1, 0, 0])  # Body frame +x is the line of sight
    
    for i in range(n_time + 1):
        r = r_target - r_rtn[:3, i].reshape((3,1))  # Adjust target location if necessary
        x_rho = r / np.linalg.norm(r)
        q_v = skw(x_d) @ x_rho / np.sqrt(2 * (1 + x_d @ x_rho))
        q_0 = np.sqrt(2 * (1 + x_d @ x_rho)) / 2
        qw[0:4, i] = np.concatenate((np.array([q_0]), q_v)).flatten()
        
        # Check for singularity
        if np.any(np.isnan(q_v)):
            print(f'warning; singularity at timestep {i+1}/{n_time+1}')
            if i > 0:
                qw[0:4, i] = qw[0:4, i-1]
        
        if i > 0:
            # Compute the angular velocity
            q1 = qw[0:4, i-1]
            q2 = qw[0:4, i]
            w = 2/dt * np.array([
                q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]
            ])
            qw[4:7, i-1] = w
            
            if i > 1:
                dM[:, i-2] = qw[4:7, i-1] - qw[4:7, i-2]
    
    # Set the angular velocity at the end to the previous value
    qw[4:7, -1] = qw[4:7, -2]
    dM[:, -1] = np.zeros(3)
    
    return qw, dM


def mekf_stm(pw,J,dt):
    """
    "Relative Computer Vision-Based Navigation for Small Inspection Spacecraft" (Tweddle, 2015)
    Eq. 17 - 21
    """

    A = np.block([[-1/2*skw(pw[3:]), np.eye(3), np.zeros((3,3))],  
                  [np.zeros((3,6)), np.linalg.inv(J)], 
                  [np.zeros((3,9))]])
    
    Φ = expm(A*dt)
    Φ12 = Φ[3:6, :3]
    
    Phi = np.block([[-expm(-dt/2*skw(pw[3:])), Φ12], 
                    [np.zeros((3,3)), np.eye(3)]])
    
    # B = np.block([[-1/2*skw(pw[3:])], [np.zeros((3,3))]])
    B = np.block([[np.zeros((3,3))], [np.eye(3)]])
    C = np.block([[np.eye(3), np.zeros((3,3))], 
                  [np.eye(3), np.zeros((3,3))],
                  [np.zeros((3,3)), np.eye(3)]])
    
    return Phi, B, C 
    
    
