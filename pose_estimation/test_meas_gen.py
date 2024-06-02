import numpy.linalg as la
from scipy.integrate import odeint

from pose_estimation.dynamics.dynamics_rot import *
from pose_estimation.dynamics.plot_misc import *
from pose_estimation.filter import *
from pose_estimation.meas_gen_utils import *

J = np.diag([300,200,150])
tf = 120

q0 = np.array([0.0, 0.0, 0.0, 1.0])
w0 = np.array([0.05, 0.05, 0.05])

# numerical integration 
dt = 0.5 
n_steps = int(tf / dt)  + 1
t = np.linspace(0, tf, n_steps)
qw = odeint(ode_qw, np.concatenate((q0, w0)), t, args=(J,np.zeros((1,3))))

q_hist = qw[:, :4]
w_hist = qw[:, :3]

Rc = np.diag(np.ones((3,)))
q_cam = np.copy(q_hist)

for i in range(len(t)):
    dq = np.zeros((4,))
    dq[0] = 1
    dq[1:] = Rc @ np.random.randn(3)
    dq /= np.linalg.norm(dq)
    q_cam[i,:] = q_mul(dq, q_cam[i,:])

Rw = np.diag((1e-4)*np.ones(3,))
Rp = np.diag((1e-4)*np.ones(3,))
Rc = np.diag((1e-4)*np.ones(3,))

yhist = gen_full_meas(q_hist, w_hist, q_cam, Rw, Rp, Rc)
print(np.linalg.norm(q_cam, axis = 1))