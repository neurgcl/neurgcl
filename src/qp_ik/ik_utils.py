import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes


def plt_dist_g_err(dist, dist_g, dq, title=''):
    Gamma_from_grad = dist[:-1] + (dist_g @ dq[:, :, np.newaxis]).squeeze()[:-1]

    err_gamma_linear = dist[1:] - Gamma_from_grad
    err_gamma_linear_norm = np.linalg.norm(err_gamma_linear, axis=-1)

    plt.figure(figsize=(6.4 * 2, 4.8))
    plt.subplot(121)
    plt.plot(dist, label=['$\Gamma_1$', '$\Gamma_2$'])

    plt.plot(
        1 + np.arange(len(Gamma_from_grad)),
        Gamma_from_grad,
        '--',
        label=[r'$\Gamma_1$ + $\nabla\Gamma $dq', r'$\Gamma_2$ + $\nabla\Gamma $dq'],
    )
    plt.grid()
    plt.legend()
    plt.title("$\Gamma$")

    plt.subplot(122)
    plt.plot(err_gamma_linear)
    plt.grid()
    plt.title(f"{title}" + r"Grad Error = $\Gamma1$ - ($\Gamma$ + $\nabla\Gamma$ @ dq)")


def plt_ik_step_info(d_outs, x_target, save_fig=None, is_spheres=False):
    l_x = d_outs['x']
    plt.figure(figsize=(6.4 * 2, 4.8 * 3))
    plt.subplot(321)
    plt.plot(l_x[:, :3], '.-')
    plt.legend(['x', 'y', 'z'])
    plt.hlines(x_target[:3], 0, len(l_x), colors='black', linestyles='dashed')
    plt.grid()
    plt.title("step state - xyz")

    plt.subplot(322)
    w_xyz = l_x[:, 3:].copy()
    mask_reverse = l_x[:, 3:].dot(x_target[3:]) < 0
    w_xyz[mask_reverse] *= -1
    # plt.plot(l_x[:,3:], '.-')
    plt.plot(w_xyz, '.-')
    plt.legend(['wx', 'wy', 'wz'])
    plt.hlines(x_target[3:], 0, len(l_x), colors='black', linestyles='dashed')
    plt.grid()
    plt.title("step state - rot")

    # plt.figure(figsize=(6.4*2, 4.8))
    plt.subplot(323)
    plt.plot(d_outs['delta'][:, :3])
    plt.plot(d_outs['delta'][:, 3:], '--')
    plt.legend(['x', 'y', 'z', 'wx', 'wy', 'wz'])
    plt.hlines(0, 0, len(l_x), colors='black', linestyles='dashed')
    plt.grid()
    plt.title("$\delta$" + f" steps={len(d_outs['delta'])}")

    plt.subplot(324)
    plt.plot(d_outs['delta_t_norm'])
    plt.plot(d_outs['delta_r_norm'])
    plt.hlines(0, 0, len(l_x), colors='black', linestyles='dashed')
    plt.legend(['$\delta_{xyz}$', '$\delta_{rot}$'])
    plt.grid()
    plt.title("$||\delta||$" + f" last t={d_outs['delta_t_norm'][-1]:.4f} r={d_outs['delta_r_norm'][-1]:.4f}")


    if is_spheres:
        plt.subplot(325)
        #
        g_Gamma_log = d_outs['g_Gamma_log']
        n_col = d_outs['pts_cur'].shape[-2]
        n_min = g_Gamma_log.shape[-1] // n_col
        # TODO: Fix *2
        for id_link in range(n_min):
            id_col = 0
            plt.plot(
                d_outs['g_Gamma_log'][:, id_link * n_col + id_col],
                label='$\Gamma_' + '{' + f'{id_link}' + '}(pt^{' + f'{id_col}' + '})$',
            )
            # for id_col in range(n_col):
            #     plt.plot(d_outs['g_Gamma_log'][:, id_min*2 + id_col], label='$\Gamma_'+'{'+f'{id_min}' + '}(pt^{'+f'{id_col}'+'})$')
        plt.legend()

        ax = plt.gca()
        ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
        if n_col>1:
            for id_link in range(n_min):
                id_col = 1
                plt.plot(d_outs['g_Gamma_log'][:, id_link * n_col + id_col], '--')

        plt.grid()
        plt.title(r'$log(\Gamma - r + 1) + \nabla\Gamma @ dq \geq 0$')

        plt.subplot(326)
        id_col = 0
        for id_link in range(n_min):
            plt.plot(
                d_outs['g_Gamma'][:, id_link * n_col + id_col],
                label='$\Gamma_' + '{' + f'{id_link}' + '}(pt^{' + f'{id_col}' + '})$',
            )
        if n_col>1:
            ax = plt.gca()
            ax.set_prop_cycle(plt.rcParams['axes.prop_cycle'])
            id_col = 1
            for id_link in range(n_min):
                plt.plot(d_outs['g_Gamma'][:, id_link * n_col + id_col], '--')
        plt.legend()
        plt.grid()
        plt.title(r' $\Gamma - r + \nabla\Gamma @ dq \geq 0$')

    # plt.plot(d_outs['g_fk'], label=['x', 'y', 'z', 'wx', 'wy', 'wz'])
    # plt.legend()
    # plt.grid()
    # plt.title(r'$x-f(q) - \frac{\partial f(q)}{\partial q} \Delta q - \delta = 0 $')

    if save_fig is not None:
        plt.savefig(save_fig)


def plt_dist_g_err_subfig(ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, dist, dist_g, dq, title=''):
    n_col = dist.shape[-1]
    Gamma_from_grad = dist[:-1] + (dist_g @ dq[:, :, np.newaxis]).squeeze(-1)[:-1]

    err_gamma_linear = dist[1:] - Gamma_from_grad
    err_gamma_linear_norm = np.linalg.norm(err_gamma_linear, axis=-1)

    ax1.plot(dist, label=[f'$\Gamma_{i}$' for i in range(min(2, n_col))])

    ax1.plot(
        1 + np.arange(len(Gamma_from_grad)),
        Gamma_from_grad,
        '--',
        # label=[(r'+ $\nabla\Gamma $dq') for i in range(min(2, n_col))],
        label=[r'$\Gamma_1$ + $\nabla\Gamma $dq', r'$\Gamma_2$ + $\nabla\Gamma $dq'],
    )
    ax1.grid()
    ax1.legend()
    ax1.set_title(f"{title}" + "$\Gamma_{t}(pt_{t+1})$")

    ax2.plot(err_gamma_linear)
    ax2.grid()
    ax2.set_title(f"{title}" + r"GradErr = $\Gamma1$-($\Gamma$+$\nabla\Gamma$@dq)")


def plt_dist_g_min_err_subfig(ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, dist, dist_g, dq, title=''):
    n_step = dist.shape[0]
    idxs= np.arange(n_step)
    idx_dist_min = dist.argmin(axis=-1)
    dist_raw = dist
    dist = dist[idxs,idx_dist_min]
    dist_g = dist_g[idxs,idx_dist_min]

    dist1 = dist_raw[idxs[1:], idx_dist_min[:-1]]
    
    # dists1 = 
    Gamma_from_grad = dist[:-1] + (dist_g[:,None,:] @ dq[:,:,None]).squeeze(-1).squeeze(-1)[:-1]

    err_gamma_linear = dist1 - Gamma_from_grad
    err_gamma_linear_norm = np.linalg.norm(err_gamma_linear, axis=-1)

    ax1.plot(dist, label=r'$\Gamma_{min}$')

    ax1.plot(
        1 + np.arange(len(Gamma_from_grad)),
        Gamma_from_grad,
        '--',
        # label=[(r'+ $\nabla\Gamma $dq') for i in range(min(2, n_col))],
        label=r'$\Gamma_{min}$ + $\nabla\Gamma $dq',
    )
    ax1.grid()
    ax1.legend()
    ax1.set_title(f"{title}" + "$\Gamma_{t}(pt_{t+1})$")

    ax2.plot(err_gamma_linear)
    ax2.grid()
    ax2.set_title(f"{title}" + r"GradErr = $\Gamma1$-($\Gamma$+$\nabla\Gamma$@dq)")

def plt_dist_min_err(d_outs, save_fig=None):
    """
    Plot linear approximation error for MMLP/JSDF at each step      Gamma wrt pt_next
    Plot predicated error for MMLP/JSDF vs Bullet at each step      Gamma wrt pt_cur
    """

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6.4 * 2, 4.8 * 3))

    # dist = d_outs['Gamma_nsdf2nextpt']
    # dist_g = d_outs['DGamma_nsdf2nextpt']
    # dq=d_outs['dq']
    # title='MMLP'
    # ax1, ax2=axes[0, :]

    # Gamma wrt pt_next ------------
    plt_dist_g_min_err_subfig(
        *axes[0, :], d_outs['Gamma_nsdf2nextpt'], d_outs['DGamma_nsdf2nextpt'], d_outs['dq'], title='MMLP '
    )
    plt_dist_g_min_err_subfig(
        *axes[1, :], d_outs['Gamma_jsdf2nextpt'], d_outs['DGamma_jsdf2nextpt'], d_outs['dq'], title='JSDF '
    )

    # Gamma wrt pt_cur ------------
    Gamma_bl_min = d_outs['Gamma_bl'].min(axis=-1)
    ax = axes[2, 0]
    # plt.figure()
    # ax.plot(d_outs['Gamma_bl'], label=['$\Gamma_1$_bl', '$\Gamma_2$_bl'])
    ax.plot(Gamma_bl_min, '--', label=r'$\Gamma_{min}$_bl')

    Gamma_nsdf_min = d_outs['Gamma_nsdf'].min(axis=-1)
    ax.plot(Gamma_nsdf_min, '-', label=r'$\Gamma_{min}$_nsdf')

    Gamma_jsdf_min = d_outs['Gamma_jsdf'].min(axis=-1)
    ax.plot(Gamma_jsdf_min, '-', label=r'$\Gamma_{min}$_jsdf')

    Gamma_rdf_min = d_outs['Gamma_rdf'].min(axis=-1)
    ax.plot(Gamma_rdf_min, '-', label=r'$\Gamma_{min}$_rdf')

    # ax.hlines(0.1, 0, len(d_outs['Gamma_jsdf']), colors='black', linestyles='dashed')
    ax.legend()
    ax.grid()
    ax.set_title("$\Gamma_{t}(pt_{t})$ (Pred vs Bullet)")

    ax = axes[2, 1]
    # ax._get_lines._idx = (ax._get_lines._idx + 1) % len(ax._get_lines._cycler_items)
    # next(ax._get_lines.prop_cycler)
    ax.plot(Gamma_nsdf_min - Gamma_bl_min, '-', label='$\Gamma_{min}$_nsdf')
    ax.plot(Gamma_jsdf_min - Gamma_bl_min, '-', label='$\Gamma_{min}$_jsdf')
    ax.plot(Gamma_rdf_min - Gamma_bl_min, '-', label='$\Gamma_{min}$_rdf')
    ax.legend()
    ax.grid()
    ax.set_title("$\Gamma$ Err vs BL")
    if save_fig is not None:
        plt.savefig(save_fig)

def plt_dist_err(d_outs, save_fig=None):
    """
    Plot linear approximation error for MMLP/JSDF at each step      Gamma wrt pt_next
    Plot predicated error for MMLP/JSDF vs Bullet at each step      Gamma wrt pt_cur
    """

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6.4 * 2, 4.8 * 3))

    # Gamma wrt pt_next ------------
    plt_dist_g_err_subfig(
        *axes[0, :], d_outs['Gamma_nsdf2nextpt'], d_outs['DGamma_nsdf2nextpt'], d_outs['dq'], title='MMLP '
    )
    plt_dist_g_err_subfig(
        *axes[1, :], d_outs['Gamma_jsdf2nextpt'], d_outs['DGamma_jsdf2nextpt'], d_outs['dq'], title='JSDF '
    )

    # Gamma wrt pt_cur ------------
    ax = axes[2, 0]
    # plt.figure()
    # ax.plot(d_outs['Gamma_bl'], label=['$\Gamma_1$_bl', '$\Gamma_2$_bl'])
    ax.plot(d_outs['Gamma_nsdf'], '--', label=['$\Gamma_1$_nsdf', '$\Gamma_2$_nsdf'])
    ax.plot(d_outs['Gamma_jsdf'], '-', label=['$\Gamma_1$_jsdf', '$\Gamma_2$_jsdf'])
    ax.plot(d_outs['Gamma_ddf'], '-', label=['$\Gamma_1$_rdf', '$\Gamma_2$_ddf'])
    # ax.hlines(0.1, 0, len(d_outs['Gamma_jsdf']), colors='black', linestyles='dashed')
    ax.legend()
    ax.grid()
    ax.set_title("$\Gamma_{t}(pt_{t})$ (Pred vs Bullet)")

    # ax = axes[2, 1]
    # ax._get_lines._idx = (ax._get_lines._idx + 1) % len(ax._get_lines._cycler_items)
    # # next(ax._get_lines.prop_cycler)
    # ax.plot(d_outs['Gamma_nsdf'] - (d_outs['Gamma_bl']), '--', label=['$\Gamma_1$_nsdf', '$\Gamma_2$_nsdf'])
    # ax.plot(d_outs['Gamma_jsdf'] - (d_outs['Gamma_bl']), '-', label=['$\Gamma_1$_jsdf', '$\Gamma_2$_jsdf'])
    # ax.legend()
    # ax.grid()
    # ax.set_title("$\Gamma$ Err vs BL")
    if save_fig is not None:
        plt.savefig(save_fig)
