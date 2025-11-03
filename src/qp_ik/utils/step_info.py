import numpy as np
from qp_ik.ca_qp_ik_cpp import CAQPIK
from qp_ik.utils import robot_kin


def cal_Gamma_net_run(d_outs, ik: CAQPIK, method):
    l_q = d_outs['q']
    n_step = len(l_q)
    l_pts = d_outs['pts_cur']
    l_Gamma_net_run = []
    idx_step=0
    for idx_step in range(n_step):
        q_cur = l_q[idx_step]
        pts_cur= l_pts[idx_step]
        T_w2links = robot_kin.tor_fk_tfs(q_cur)
        Gamma, _ = ik.cal_Gamma_g_all(q_cur, pts_cur, T_w2links, method=method)
        Gamma = Gamma.reshape(-1, len(pts_cur)) # n_link, n_pt
        Gamma_min = Gamma.min(axis=0)
        l_Gamma_net_run.append(Gamma_min)
    l_Gamma_net_run = np.array(l_Gamma_net_run)
    l_Gamma_net_run -= ik.col_r
    return l_Gamma_net_run