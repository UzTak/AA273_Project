# Plotting Code
import matplotlib.pyplot as plt
import numpy as np

def MRP_error_band(omega, mrp, covariance, dt):
    """
    Inputs: 
    omega nx3 of angular rates
    mrp nx3 vector of modified rogrigues parameters
    covariance 6x6 x n tensor of covariance for each time step

    Outputs: 
    6 error band plots
    """
    print(covariance.shape)
    n = omega.shape[0]  # Number of time steps
    t_vals = np.linspace(0, n*dt, n)

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    for i in range(3):
        # Plot omega
        axs[i, 0].plot(t_vals, omega[:, i])
        axs[i, 0].set_title(f'Omega {i+1}')
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Value')

        # Plot mrp
        axs[i, 1].plot(t_vals, mrp[:, i])
        axs[i, 1].set_title(f'MRP {i+1}')
        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Value')

    # Plot covariance error bands
    for i in range(n):
        cov_diag = np.sqrt(np.diag(covariance[i, :, :]))
        print(cov_diag)
        # print('HERE bro I got here')
        axs[0, 0].fill_between(t_vals[i], omega[i, 0] - 5*cov_diag[3], omega[i, 0] + 5*cov_diag[3], alpha=0.9)
        axs[1, 0].fill_between(t_vals[i], omega[i, 1] - 5*cov_diag[4], omega[i, 1] + 5*cov_diag[4], alpha=0.9)
        axs[2, 0].fill_between(t_vals[i], omega[i, 2] - 5*cov_diag[5], omega[i, 2] + 5*cov_diag[5], alpha=0.9)
        axs[0, 1].fill_between(t_vals[i], mrp[i, 0] - 5*cov_diag[0], mrp[i, 0] + 5*cov_diag[0], alpha=0.9)
        axs[1, 1].fill_between(t_vals[i], mrp[i, 1] - 5*cov_diag[1], mrp[i, 1] + 5*cov_diag[1], alpha=0.9)
        axs[2, 1].fill_between(t_vals[i], mrp[i, 2] - 5*cov_diag[2], mrp[i, 2] + 5*cov_diag[2], alpha=0.9)

    return fig
    # plt.tight_layout()
    # plt.show()