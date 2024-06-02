# Plotting Code
import matplotlib.pyplot as plt
import numpy as np

def MRP_error_band(omega, mrp, covariance):
    """
    Inputs: 
    omega 3x1 x n of angular rates
    mrp 3x1 x n vector of modified rogrigues parameters
    covariance 6x6 x n tensor of covariance for each time step

    Outputs: 
    6 error band plots
    """

    n = omega.shape[2]  # Number of time steps

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    for i in range(3):
        # Plot omega
        axs[i, 0].plot(range(n), omega[i, 0, :])
        axs[i, 0].set_title(f'Omega {i+1}')
        axs[i, 0].set_xlabel('Time Step')
        axs[i, 0].set_ylabel('Value')

        # Plot mrp
        axs[i, 1].plot(range(n), mrp[i, 0, :])
        axs[i, 1].set_title(f'MRP {i+1}')
        axs[i, 1].set_xlabel('Time Step')
        axs[i, 1].set_ylabel('Value')

    # Plot covariance error bands
    for i in range(n):
        cov_diag = np.sqrt(np.diag(covariance[:, :, i]))
        axs[0, 0].fill_between([i], omega[0, 0, i] - cov_diag[0], omega[0, 0, i] + cov_diag[0], alpha=0.2)
        axs[1, 0].fill_between([i], omega[1, 0, i] - cov_diag[1], omega[1, 0, i] + cov_diag[1], alpha=0.2)
        axs[2, 0].fill_between([i], omega[2, 0, i] - cov_diag[2], omega[2, 0, i] + cov_diag[2], alpha=0.2)
        axs[0, 1].fill_between([i], mrp[0, 0, i] - cov_diag[3], mrp[0, 0, i] + cov_diag[3], alpha=0.2)
        axs[1, 1].fill_between([i], mrp[1, 0, i] - cov_diag[4], mrp[1, 0, i] + cov_diag[4], alpha=0.2)
        axs[2, 1].fill_between([i], mrp[2, 0, i] - cov_diag[5], mrp[2, 0, i] + cov_diag[5], alpha=0.2)

    plt.tight_layout()
    plt.show()