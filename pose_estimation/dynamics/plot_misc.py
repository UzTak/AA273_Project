
import numpy as np
import matplotlib.pyplot as plt
from .dynamics_rot import * 


def plot_sol_qw2(fig, qw, dw, t, qw_ref=None, c="g"):
    """
    Args: 
        qw: 7 x N array of quaternions and angular velocities
        dw: 3 x N-1 array of angular velocities
        t:  1 x N array of time
        qw_ref: 7 x N array of reference quaternions and angular velocities
    """
    
    w_lb = -1e-3
    w_ub = 1e-3
        
    for j in range(12):
        plt.subplot(4,3,j+1)
        
        if j < 7:  # qw
            # plt.plot(t, qw[:,j], 'b')
            if qw_ref is not None:
                plt.plot(t, qw_ref[j,:], '--ro', label='ref.')
            plt.plot(t, qw[j,:], c=c, label='cvx.')
            # plt.plot(t, qw_nl[j,:],  'b', label='nonlin.')
            
            if j == 0: plt.legend()
            
        elif j == 7:
            # plt.plot(t, qw[:,0]**2 + qw[:,1]**2 + qw[:,2]**2 + qw[:,3]**2, 'b', label='|q|')
            plt.plot(t, np.sqrt(qw[0,:]**2 + qw[1,:]**2 + qw[2,:]**2 + qw[3,:]**2),  c=c, label='|q_rel|')
            # plt.plot(t, np.sqrt(qw_nl[0,:]**2 + qw_nl[1,:]**2 + qw_nl[2,:]**2 + qw_nl[3,:]**2), 'b', label='|q_rel|')
        elif j < 11: 
            if dw is not None:
                plt.stem(t[:-1], dw[j-8,:], 'k--')
        # else: 
        #     plt.plot(t[:-1], vqw[0,:], 'k', label='vc_q1')
        #     plt.plot(t[:-1], vqw[1,:], 'r', label='vc_q2')
        #     plt.plot(t[:-1], vqw[2,:], 'b', label='vc_q3') 
        #     plt.plot(t[:-1], vqw[3,:], 'g', label='vc_q4') 
        #     plt.legend()

        if j == 0:
            plt.xlabel('time [s]')
            plt.ylabel('$q_1$')
            # plt.grid(True)
            # plt.xlim([-1,1])
            plt.ylim([-1,1])
        elif j == 1:
            plt.xlabel('time [s]')
            plt.ylabel('$q_2$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 2:
            plt.xlabel('time [s]')
            plt.ylabel('$q_3$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 3:
            plt.xlabel('time [s]')
            plt.ylabel('$q_4$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 4:
            plt.xlabel('time [s]')
            plt.ylabel('$w_1$')
            # plt.grid(True)
            # plt.ylim([w_lb,w_ub]) 
        elif j == 5:
            plt.xlabel('time [s]')
            plt.ylabel('$w_2$')
            # plt.grid(True)
            # plt.ylim([w_lb,w_ub]) 
        elif j == 6:
            plt.xlabel('time [s]')
            plt.ylabel('$w_3$')
            # plt.grid(True)
            # plt.ylim([w_lb,w_ub]) 
        elif j == 7:
            plt.xlabel('time [s]')
            plt.ylabel('$|q|$')
            # plt.grid(True)
            plt.ylim([0.9,1.1])
        elif j == 8:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_x$')
            # plt.grid(True)
            # plt.ylim([-1,1])
        elif j == 9:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_y$')
            # plt.grid(True)
            # plt.ylim([-1,1])
        elif j == 10:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_z$')
            # plt.grid(True)
            # plt.ylim([-1,1])
        
    plt.tight_layout()
    plt.legend()
    # plt.show()
    # fname = root_folder + '\\optimization\\saved_files\\plots\\scp\\iter' + str(i) + '.png'
    # plt.savefig('./rot_history.png', dpi = 600)
    
    return fig 
    
def plot_sol_qw(axes, qw, dw, t, qw_ref=None):
    """
    Args: 
        qw: 7 x N array of quaternions and angular velocities
        dw: 3 x N-1 array of angular velocities
        t:  1 x N array of time
        qw_ref: 7 x N array of reference quaternions and angular velocities
    """
    
    w_lb = -1e-3
    w_ub = 1e-3
        
    for j in range(12):
        
        
        if j < 7:  # qw
            # plt.plot(t, qw[:,j], 'b')
            if qw_ref is not None:
                plt.plot(t, qw_ref[j,:], '--ro', label='ref.')
            plt.plot(t, qw[j,:], ':g.', label='cvx.')
            # plt.plot(t, qw_nl[j,:],  'b', label='nonlin.')
            
            if j == 0: plt.legend()
            
        elif j == 7:
            # plt.plot(t, qw[:,0]**2 + qw[:,1]**2 + qw[:,2]**2 + qw[:,3]**2, 'b', label='|q|')
            plt.plot(t, np.sqrt(qw[0,:]**2 + qw[1,:]**2 + qw[2,:]**2 + qw[3,:]**2), 'g', label='|q_rel|')
            # plt.plot(t, np.sqrt(qw_nl[0,:]**2 + qw_nl[1,:]**2 + qw_nl[2,:]**2 + qw_nl[3,:]**2), 'b', label='|q_rel|')
        elif j < 11: 
            if dw is not None:
                plt.stem(t[:-1], dw[j-8,:], 'k--')
        # else: 
        #     plt.plot(t[:-1], vqw[0,:], 'k', label='vc_q1')
        #     plt.plot(t[:-1], vqw[1,:], 'r', label='vc_q2')
        #     plt.plot(t[:-1], vqw[2,:], 'b', label='vc_q3') 
        #     plt.plot(t[:-1], vqw[3,:], 'g', label='vc_q4') 
        #     plt.legend()

        if j == 0:
            plt.xlabel('time [s]')
            plt.ylabel('$q_1$')
            # plt.grid(True)
            # plt.xlim([-1,1])
            plt.ylim([-1,1])
        elif j == 1:
            plt.xlabel('time [s]')
            plt.ylabel('$q_2$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 2:
            plt.xlabel('time [s]')
            plt.ylabel('$q_3$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 3:
            plt.xlabel('time [s]')
            plt.ylabel('$q_4$')
            # plt.grid(True)
            # plt.xlim([-0.1,3])
            plt.ylim([-1,1])
        elif j == 4:
            plt.xlabel('time [s]')
            plt.ylabel('$w_1$')
            # plt.grid(True)
            # plt.ylim([w_lb,w_ub]) 
        elif j == 5:
            plt.xlabel('time [s]')
            plt.ylabel('$w_2$')
            # plt.grid(True)
            # plt.ylim([w_lb,w_ub]) 
        elif j == 6:
            plt.xlabel('time [s]')
            plt.ylabel('$w_3$')
            # plt.grid(True)
            # plt.ylim([w_lb,w_ub]) 
        elif j == 7:
            plt.xlabel('time [s]')
            plt.ylabel('$|q|$')
            # plt.grid(True)
            plt.ylim([0.9,1.1])
        elif j == 8:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_x$')
            # plt.grid(True)
            # plt.ylim([-1,1])
        elif j == 9:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_y$')
            # plt.grid(True)
            # plt.ylim([-1,1])
        elif j == 10:
            plt.xlabel('time [s]')
            plt.ylabel('$d\omega_z$')
            # plt.grid(True)
            # plt.ylim([-1,1])
        
    plt.tight_layout()
    plt.legend()
    # plt.show()
    # fname = root_folder + '\\optimization\\saved_files\\plots\\scp\\iter' + str(i) + '.png'
    # plt.savefig('./rot_history.png', dpi = 600)
    
    