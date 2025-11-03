from typing import List

import numpy as np
from bl_sim_env.utils.np_tf import R2rv, T2tR, T2tw, T2xyzrpy
from bl_sim_env.utils.util_log import listd_2_d
from casadi import vcat
from qp_ik.ca_qp_ik_cpp import CAQPIK, cal_err_Gamma
from qp_ik.env.col_env_base import ColEnvBase
from qp_ik.utils import robot_kin


def cal_log_info(
    robot,
    col_env: ColEnvBase,
    ik: CAQPIK,
    l_outs,
    l_qps,
    pts_traj,
    l_method: List[str] = ["nsdf", "jsdf", "rdf"],
):
    d_outs = listd_2_d(l_outs)

    l_dist = []
    for i_step in range(len(d_outs["q"])):
        pts_cur = pts_traj[min(i_step, len(pts_traj) - 1)]
        pts_next = pts_traj[min(i_step + 1, len(pts_traj) - 1)]  # used in QP

        col_env.update_obstacles(pts_cur)
        pcs_cur = col_env.get_pcs()

        d_info = {}
        for method in l_method:
            Gamma, _ = ik.cal_Gamma_g_all(d_outs["q"][i_step], pcs_cur, method=method)
            d_info[f"Gamma_{method}"] = Gamma
        
        robot.reset_q(d_outs["q"][i_step])
        if col_env.name == "mesh":
            bl_dist = np.zeros((1, 9))
        else:
            bl_dist = col_env.gel_bl_dists()
            if "ball" in col_env.name:
                col_env.update_obstacles(pts_cur)
                bl_dist += col_env.ball_r
        d_info["bl_dist"] = bl_dist

        col_env.update_obstacles(pts_next)
        pcs = col_env.get_pcs()

        # Gamma  cur_q -> pcs_next
        for method in l_method:
            Gamma2nextpt, DGamma2nextpt = ik.cal_Gamma_g_all(d_outs["q"][i_step], pcs, method=method)
            d_info[f"Gamma_{method}2nextpt"] = Gamma2nextpt
            d_info[f"DGamma_{method}2nextpt"] = DGamma2nextpt

        l_dist.append(d_info)
    l_dist = listd_2_d(l_dist)
    l_dist["Gamma_bl"] = l_dist["bl_dist"].min(axis=-1)

    d_outs.update(l_dist)
    d_outs["ret_qp"] = l_qps
    return d_outs


def cal_log_data(ik: CAQPIK, q_cur, pts_next, T_target, ret_dq, ret_delta):
    # q_cur ----------
    # p_T_cur = robot_kin.fk_tfs(q_cur)[-1]
    # p_x_cur = T2tw(p_T_cur)
    # R_cur = p_T_cur[:3, :3]
    # t_cur = p_T_cur[:3, 3]

    delta_t_norm = np.linalg.norm(ret_delta[:3])
    delta_r_norm = np.linalg.norm(ret_delta[3:])

    t_target, R_target = T2tR(T_target)

    ee_err, J, T_w2links, T_w2ee = ik.cal_vars_fk(q_cur, t_target, R_target)

    p_T_cur = T_w2ee
    p_x_cur = T2tw(p_T_cur)
    R_cur = p_T_cur[:3, :3]
    t_cur = p_T_cur[:3, 3]

    Gamma, DGamma = ik.get_Gamma_with_g(q_cur, pts_next, T_w2links)
    if ik.filter_Gamma:
        idx_Gamma_mask = Gamma < 0.5
        Gamma = Gamma[idx_Gamma_mask]
        DGamma = DGamma[idx_Gamma_mask, :]
    
    # ee_err, J, Gamma, DGamma = ik.cal_tmp_vars(q_cur, t_target, R_target, pts_next)
    g_fk = ee_err - J @ ret_dq - ret_delta
    # g_Gamma = Gamma - ik.col_r - ik.col_margin + DGamma @ ret_dq
    # g_Gamma_log = cal_err_Gamma(Gamma, ik.col_r, ik.col_margin) + DGamma @ ret_dq

    # q_next ----------
    q_next = np.array(q_cur + ret_dq).squeeze()
    # Gamma_next, _ = ik.get_Gamma_with_g(q_next, pts_next)
    # Gamma_next= Gamma_next[idx_Gamma_mask]

    T_w2links1, _ = robot_kin.fk_tf_jacos(q_next)
    t_cur1 = T_w2links1[-1][:3, 3]
    R_cur1 = T_w2links1[-1][:3, :3]

    t_err1 = t_cur1 - t_cur
    w_err1 = R_cur @ R2rv(R_cur.T @ R_cur1)
    err_Dfk = vcat([t_err1, w_err1])

    # err_DGamma = Gamma_next - (Gamma + DGamma @ ret_dq)
    if len(Gamma) == 0:
        d_min = np.nan
    else:
        d_min = Gamma.min()
    out_d = {
        # qp
        "delta": np.array(ret_delta).squeeze(),
        "delta_t_norm": np.array(delta_t_norm).squeeze(),
        "delta_r_norm": np.array(delta_r_norm).squeeze(),
        "g_fk": np.array(g_fk).squeeze(),
        # "g_Gamma_log": np.array(g_Gamma_log).squeeze(),
        # "g_Gamma": np.array(g_Gamma).squeeze(),
        # states in current step
        "x": np.array(p_x_cur),  # trans, rotvec
        "q": np.array(q_cur).squeeze(),
        "dq": np.array(ret_dq).squeeze(),
        "q_next": np.array(q_next).squeeze(),
        "pts_next": np.array(pts_next).squeeze(),
        # "Gamma_next": np.array(Gamma_next).squeeze(),
        "ee_err": np.array(ee_err).squeeze(),
        # "Gamma": np.array(Gamma).squeeze(),
        # "DGamma": np.array(DGamma).squeeze(),
        # "err_DGamma": np.array(err_DGamma).squeeze(),
        "err_Dfk": np.array(err_Dfk).squeeze(),
        "d_min": d_min,
    }
    return out_d

def cal_log_qp_step(ik: CAQPIK, q_cur, pts_next, T_target, ret_dq, ret_delta):
    # q_cur ----------
    p_T_cur = robot_kin.fk_tfs(q_cur)[-1]
    p_x_cur = T2tw(p_T_cur)
    R_cur = p_T_cur[:3, :3]
    t_cur = p_T_cur[:3, 3]

    delta_t_norm = np.linalg.norm(ret_delta[:3])
    delta_r_norm = np.linalg.norm(ret_delta[3:])

    t_target, R_target = T2tR(T_target)

    ee_err, J, T_w2links, _ = ik.cal_vars_fk(q_cur, t_target, R_target)
    Gamma, DGamma = ik.get_Gamma_with_g(q_cur, pts_next, T_w2links)
    if ik.filter_Gamma:
        idx_Gamma_mask = Gamma < 0.5
        Gamma = Gamma[idx_Gamma_mask]
        DGamma = DGamma[idx_Gamma_mask, :]
 
    if len(Gamma) == 0:
        Gamma_min = 0.5
    else:
        Gamma_min = Gamma.min()

    # ee_err, J, Gamma, DGamma = ik.cal_tmp_vars(q_cur, t_target, R_target, pts_next)
    g_fk = ee_err - J @ ret_dq - ret_delta
    # g_Gamma = Gamma - ik.col_r - ik.col_margin + DGamma @ ret_dq
    # g_Gamma_log = cal_err_Gamma(Gamma, ik.col_r, ik.col_margin) + DGamma @ ret_dq

    # q_next ----------
    q_next = np.array(q_cur + ret_dq).squeeze()
    # Gamma_next, _ = ik.get_Gamma_with_g(q_next, pts_next)
    # Gamma_next= Gamma_next[idx_Gamma_mask]

    T_w2links1, _ = robot_kin.fk_tf_jacos(q_next)
    t_cur1 = T_w2links1[-1][:3, 3]
    R_cur1 = T_w2links1[-1][:3, :3]

    t_err1 = t_cur1 - t_cur
    w_err1 = R_cur @ R2rv(R_cur.T @ R_cur1)
    err_Dfk = vcat([t_err1, w_err1])

    # err_DGamma = Gamma_next - (Gamma + DGamma @ ret_dq)

    out_d = {
        # qp
        "delta": np.array(ret_delta).squeeze(),
        "delta_t_norm": np.array(delta_t_norm).squeeze(),
        "delta_r_norm": np.array(delta_r_norm).squeeze(),
        # "g_fk": np.array(g_fk).squeeze(),
        # "g_Gamma_log": np.array(g_Gamma_log).squeeze(),
        # "g_Gamma": np.array(g_Gamma).squeeze(),
        # states in current step
        "x": np.array(p_x_cur),  # trans, rotvec
        "q": np.array(q_cur).squeeze(),
        "dq": np.array(ret_dq).squeeze(),
        # "q_next": np.array(q_next).squeeze(),
        # "pts_next": np.array(pts_next).squeeze(),
        # "Gamma_next": np.array(Gamma_next).squeeze(),
        "ee_err": np.array(ee_err).squeeze(),
        # "err_Dfk": np.array(err_Dfk).squeeze(),
        "Gamma_min": Gamma_min,
    }
    return out_d

def print_init_target_info(robot, robot_kin, q_0, T_target):
    q_target = robot.get_ik(T_target)
    T_target1 = robot_kin.fk_tfs(q_target)[-1]
    print("-" * 10)
    print(f"q_0:      {list(q_0)}")
    print(f"xyzrpy_0: {list(T2xyzrpy(robot_kin.fk_tfs(q_0)[-1]))}")

    print("-" * 10)
    print(f"q_1:      {list(q_target)}")
    print(f"xyzrpy_1: {list(T2xyzrpy(T_target1))}")

    T_diff = np.linalg.inv(T_target) @ T_target1
    err = T2tw(T_diff)
    err_t, err_r = err[:3], err[3:]
    err_t = np.linalg.norm(err_t)
    err_r = np.linalg.norm(err_r)
    # T2xyzrpy(T_diff, degrees=True)
    print(f"T_target diff[err_t, err_r] {err_t:.4f} m, {np.rad2deg(err_r):.4f} deg")
    print("-" * 10)
