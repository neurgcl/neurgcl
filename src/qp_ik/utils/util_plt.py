import os
import matplotlib.pyplot as plt

def plt_Gamma_run(l_Gamma_bl, l_Gamma_net, net_method, col_margin, col_margin_low=False, 
                  only_min_dist=False,
                  save_fig=None):
    """
    Plot the Gamma_bl, Gamma_net at current step
    """
    plt.figure(figsize=(6.4, 4.8))

    n_step = l_Gamma_bl.shape[0]
    n_pt = l_Gamma_bl.shape[1]
    ax=plt.gca()
    
    def cal_label(method, i=None):
        if i is None:
            return f'$\Gamma_{{{method}}}$'
        return f'$\Gamma_{{{method}, s={i}}}$'
    
    for i in range(n_pt):
        ax.plot(l_Gamma_bl[:,i], label=cal_label('GT', i))
        ax.plot(l_Gamma_net[:,i], '--',label=cal_label(net_method, i))

    ax.legend()
    ax.grid()
    ax.hlines(col_margin, 0, n_step, color='k', linestyles='--')
    if col_margin_low:
        ax.hlines(col_margin_low, 0, n_step , color='gray',linestyles='--')
    if save_fig is not None:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig)