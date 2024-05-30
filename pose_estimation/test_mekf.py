import numpy.linalg as la
from scipy.integrate import odeint

from dynamics.dynamics_rot import *
from dynamics.plot_misc import *
from filter import *
from meas_gen_utils import *

J = np.diag([300,200,150])
tf = 80

q0 = np.array([0.0, 0.0, 0.0, 1.0])
w0 = np.array([0.05, 0.05, 0.05])

# numerical integration 
dt = 0.5 
n_steps = int(tf / dt)  + 1
t = np.linspace(0, tf, n_steps)
qw = odeint(ode_qw, np.concatenate((q0, w0)), t, args=(J,np.zeros((1,3))))
# plot_sol_qw2(np.transpose(qw), None, t, qw_ref=None)

Rc = np.diag((1e-4)*np.ones((3,)))
q_cam = np.copy(qw[1:,:4])
idxNans = np.random.randint(2, size = (len(t) - 1,))

for i in range(len(t)-1):
    dq = np.zeros((4,))
    dq[0] = 1
    dq[1:] = Rc @ np.random.randn(3)
    dq /= np.linalg.norm(dq)
    q_cam[i,:] = q_mul(dq, q_cam[i,:])
    if idxNans[i]:
        q_cam[i,:] = np.zeros((4,))


Rw = np.diag((1e-6)*np.ones(3,))
Rp = np.diag((1e-4)*np.ones(3,))
Rc = np.diag((1e-4)*np.ones(3,))


yhist = gen_full_meas(qw[1:,:4], qw[1:,4:], q_cam, Rw, Rp, Rc)
uhist = np.zeros((len(t)-1,3)) 

qw0 = np.concatenate((q0, w0))
mu0 = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.05])
Sig0 = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
n_steps = len(t)-1 
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
R = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])

xest_hist = np.empty((n_steps+1, 7))
Pest_hist = np.empty((n_steps+1, 6, 6))
xest_hist[0] = qw0
Pest_hist[0] = Sig0

mekf = MEKF(mu0, Sig0, Q, R, q0, dt=dt)

for t_index, (u, y) in enumerate(zip(uhist, yhist)):
    
    x_est_mekf, P_est_mekf = mekf.step(u, y, J)
    xest_hist[t_index+1], Pest_hist[t_index+1] = x_est_mekf, P_est_mekf
    
    print("timestep: ", t_index)
    # err_ell, mu = error_ellipse(xest_hist[t_index+1][:2], Pest_hist[t_index+1][:2, :2], 0.95)
    # ax1.plot(err_ell[0], err_ell[1], "-", c="r", linewidth=1)
    # ax1.scatter(mu[0], mu[1], c="red", s=5)

fig = plt.figure(figsize=(12,8))
fig = plot_sol_qw2(fig, np.transpose(qw), None, t, qw_ref=None, c="g")
fig = plot_sol_qw2(fig, np.transpose(xest_hist), None, t, qw_ref=None, c="b")

plt.show()