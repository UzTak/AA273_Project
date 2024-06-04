# Plotting Code
import matplotlib.pyplot as plt
import numpy as np



def MRP_error_band(omega, true_omega, mrp, covariance, dt):
    """
    Inputs: 
    omega nx3 of angular rates
    mrp nx3 vector of modified rogrigues parameters
    covariance 6x6 x n tensor of covariance for each time step

    Outputs: 
    6 error band plots
    """
    plt.rcParams['axes.grid'] = True
    n = omega.shape[0]  # Number of time steps
    t_vals = np.linspace(0, n*dt, n)

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.5)
    for i in range(3):
        # Plot omega
        axs[i, 0].plot(t_vals, omega[:, i], label=r'Estimated $\omega$')
        axs[i, 0].plot(t_vals, true_omega[:, i], label=r'Nominal $\omega$')
        axs[i, 0].set_title(rf'$\omega_{i+1}$', fontsize=10)
        axs[i, 0].set_xlabel('Time (s)', fontsize=6)
        axs[i, 0].set_ylabel('Value', fontsize=6)

        # Plot mrp
        axs[i, 1].plot(t_vals, mrp[:, i], label=r'$\delta p_{est}$')
        axs[i, 1].plot(t_vals, np.zeros((n)), label=r'$\delta p_{nominal}$')
        axs[i, 1].set_title(rf'$\delta p_{i+1}$', fontsize=10)
        axs[i, 1].set_xlabel('Time (s)', fontsize=6)
        axs[i, 1].set_ylabel('Value', fontsize=6)
        axs[i, 1].set_ylim((-0.25, 0.25))

    axs[0, 0].legend()
    axs[0, 1].legend()
    # Plot covariance error bands
    std_vals = np.zeros((n, 6))
    for i in range(n):
        std_vals[i, :] = np.sqrt(np.diag(covariance[i, :, :]))

    axs[0, 0].fill_between(t_vals, omega[:, 0] - 1*std_vals[:, 3], omega[:, 0] + 1*std_vals[:, 3], color='green', alpha=0.3)
    axs[1, 0].fill_between(t_vals, omega[:, 1] - 1*std_vals[:, 4], omega[:, 1] + 1*std_vals[:, 4], color='green', alpha=0.3)
    axs[2, 0].fill_between(t_vals, omega[:, 2] - 1*std_vals[:, 5], omega[:, 2] + 1*std_vals[:, 5], color='green', alpha=0.3)
    axs[0, 1].fill_between(t_vals, -1*std_vals[:, 0], +1*std_vals[:, 0], color='green', alpha=0.3)
    axs[1, 1].fill_between(t_vals, -1*std_vals[:, 1], +1*std_vals[:, 1], color='green', alpha=0.3)
    axs[2, 1].fill_between(t_vals, -1*std_vals[:, 2], +1*std_vals[:, 2], color='green', alpha=0.3)

    return fig

    # plt.tight_layout()
    # plt.show()