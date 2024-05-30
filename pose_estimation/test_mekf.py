import numpy.linalg as la
from scipy.integrate import odeint

from dynamics.dynamics_rot import *
from dynamics.plot_misc import *
from filter import *

J = np.diag([300,200,150])
tf = 120

q0 = np.array([0.0, 0.0, 0.0, 1.0])
w0 = np.array([0.05, 0.05, 0.05])

# numerical integration 
dt = 0.5 
n_steps = int(tf / dt)  + 1
t = np.linspace(0, tf, n_steps)
qw = odeint(ode_qw, np.concatenate((q0, w0)), t, args=(J,np.zeros((1,3))))
# plot_sol_qw2(np.transpose(qw), None, t, qw_ref=None)

uhist = np.zeros((len(t)-1,3)) 
yhist = np.zeros((len(t)-1,9))

for i in range(len(t)-1):
    q, w = qw[i,:4], qw[i,4:]
    p1 = 1e-3 * np.random.randn(3)
    p2 = 1e-3 * np.random.randn(3)
    w1 = w + 1e-3 * np.random.randn(3)
    yhist[i] = np.concatenate((p1, p2, w1))


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



