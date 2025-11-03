import os
import time
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pybullet as op
from matplotlib import pyplot as plt

from bl_sim_env.env.bl_extras_base import BLPanda
from bl_sim_env.env.panda_env_base import PandaEnv
from bl_sim_env.utils import np_tf
from bl_sim_env.utils.util_file import write_pickle
from bl_sim_env.utils.util_log import get_time_str, listd_2_d
from neurg.my_utils.util_apitable import dst_bulk_update_or_create, get_dst
from neurg.my_utils.util_csv import CSVLogger
from qp_ik.ca_qp_ik_cpp import CAQPIK
from qp_ik.cfg import DEV_WS_DIR
from qp_ik.env.col_env_base import ColEnvBase
from qp_ik.filter import AverageFilter
from qp_ik.ik_utils import plt_dist_min_err, plt_ik_step_info
from qp_ik.utils import robot_kin
from qp_ik.utils.step_info import cal_Gamma_net_run
from qp_ik.utils.util_log_info import cal_log_data, cal_log_info
from qp_ik.utils.util_plt import plt_Gamma_run

LOG_DIR = DEV_WS_DIR + "/logs/qpik"


def check_setting_valid(
    ik: CAQPIK, robot: BLPanda, col_env: ColEnvBase, T_0, d_min=None, verbose=False
):
    q_0, info = robot.get_ik_checkerr(T_0, ret_info=True)
    # q_0 = cal_vaild_ik(robot, robot_kin, T_0)

    if q_0 is None:
        if verbose:
            xyzrpy = np_tf.T2xyzrpy(T_0, degrees=True)
            print(
                f"ik err for {xyzrpy}\n-- err_t, err_t: {info['err_t']}, {info['err_r']}"
            )
        return False, None

    robot.reset_q(q_0)
    # need col_env.update_obstacles(pts)
    d_bl = col_env.gel_bl_dists().min()
    if d_min is None:
        d_min = ik.col_margin + 0.02
    is_dist_valid = d_bl > d_min
    if verbose:
        print("d_bl:", d_bl)
    return is_dist_valid, q_0


def run_one_episode(
    env: PandaEnv,
    ik: CAQPIK,
    q_0,
    T_target,
    pts_traj,
    max_step=200,
    save_video=True,
    exp_ver=None,
    log_dir=LOG_DIR,
    step_time=0.01,
    pcs=None,
    warmup_qp=1,
):
    """
    q_0: 7
    pts_traj: 1+steps, n_pt, 3
    """
    p = env.p
    robot = env.robot
    col_env: ColEnvBase = env.col_env
    env_type = col_env.name
    p.configureDebugVisualizer(op.COV_ENABLE_SINGLE_STEP_RENDERING, 0)

    #  Reset env --------------------
    q_cur = deepcopy(q_0)
    robot.reset_q(q_cur)

    env.reset_init_target(T_target, robot_kin.fk_tfs(q_0)[-1])
    env.remove_txt()

    idx_step = 0
    col_origin_next = pts_traj[idx_step]
    col_env.update_obstacles(col_origin_next)
    pcs_next = col_env.get_pcs_with_offset(col_origin_next)
    n_pts = len(col_env.get_pcs())

    # warm up net
    ik.warm_up(n_pts)

    # warm up qp
    for _ in range(warmup_qp):
        _ = ik.solve_qp(q_cur, pcs_next, T_target)

    if save_video:
        env.start_record(f"{log_dir}/{exp_ver}.mp4")
        p.stepSimulation()
        p.configureDebugVisualizer(op.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    # Run --------------------------------
    l_outs = []
    l_qps = []
    reached = False
    stop_flag = False

    err_norm_buffer = AverageFilter(10, 2)
    for idx_step in range(max_step):
        col_origin_cur = deepcopy(col_origin_next)
        col_env.update_obstacles(col_origin_cur)

        col_origin_next = pts_traj[min(1 + idx_step, len(pts_traj) - 1)]
        pcs_next = col_env.get_pcs_with_offset(col_origin_next)

        dt_qp = np.nan
        try:
            t0 = time.perf_counter()
            ret_dq, ret_delta, ret_qp, _ = ik.solve_qp(q_cur, pcs_next, T_target)
            t1 = time.perf_counter()
            dt_qp = t1 - t0
            l_qps.append(ret_qp)
        except Exception as e:
            ret_dq = np.zeros(7)
            ret_delta = np.zeros(6)
            ret_qp = {}
            print("ik error" + "=" * 80 + "\n")
            print(e)
            print("\n")
            stop_flag = True
        t0 = time.perf_counter()
        info = cal_log_data(ik, q_cur, pcs_next, T_target, ret_dq, ret_delta)
        t1 = time.perf_counter()
        print(f"cal_log_data time: {(t1-t0)*1000:.3f} ms")
        info["dt_qp"] = dt_qp

        bl_dist = col_env.gel_bl_dists()
        # if env.col_obj_name == 'ball':
        #     bl_dist += env.ball_r

        info["Gamma_bl_run"] = bl_dist.min(axis=-1)
        info["Gamma_bl_run_min"] = info["Gamma_bl_run"].min()
        info["pts_cur"] = col_origin_cur
        l_outs.append(info)

        d_bl = info["Gamma_bl_run_min"]
        ee_err = info["ee_err"]

        # check if reach goal
        err_norm_buffer.update(
            np.array([np.linalg.norm(ee_err[:3]), np.linalg.norm(ee_err[3:])])
        )
        ee_err_t_norm, ee_err_r_norm = err_norm_buffer.get_maxabs()
        if ee_err_t_norm < 0.005 and ee_err_r_norm < np.deg2rad(1):
            print(
                f"Reach Goal - ee_err_t_norm: {ee_err_t_norm}, ee_err_r_norm: {np.rad2deg(ee_err_r_norm)}"
            )
            reached = True
            stop_flag = True

        env.update_txt_step_dis(idx_step, d_bl, ik.col_margin, ik.col_margin - 0.02)

        p.stepSimulation()
        if save_video:
            p.configureDebugVisualizer(op.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        else:
            pass
            time.sleep(step_time)

        if stop_flag:
            break

        dq_limited = ik.limit_vel(ret_dq, step_time)

        q_cur = np.array(q_cur + dq_limited).squeeze()
        robot.reset_q(q_cur)

    if save_video:
        env.stop_record()

    d_outs = cal_log_info(robot, col_env, ik, l_outs, l_qps, pts_traj)

    d_metrics = {
        "steps": idx_step + 1,
        # "dt_qp_all": l_dt_qp.mean(),
        # "dt_qp_gamma_J": l_dt_gamma_J.mean(),
        "dis_close_traj_min": d_outs["Gamma_bl_run_min"].min(),
        "dis_close_traj_max": d_outs["Gamma_bl_run_min"].max(),
        "err_t": ee_err_t_norm,
        "err_r": ee_err_r_norm,
        "reached": reached,
    }
    l_extra = [ret_qp["extra"] for ret_qp in d_outs["ret_qp"]]
    d_extra = listd_2_d(l_extra)
    # for k in ['dt_qp_all', 'dt_qp_gamma_J', 'qp_d_min']:
    for k in d_extra:
        d_metrics[k] = d_extra[k].mean()

    d_outs["metrics"] = d_metrics
    print("=" * 80)
    s_out = f"{env_type} - {exp_ver} |\t"
    for k in d_metrics:
        if k == "steps":
            s_out += f"{k}: {d_metrics[k]} | "
        elif k.startswith("dt"):
            s_out += f"\t{k}: {d_metrics[k]*1000:.3f}"
        elif k.startswith("dis"):
            s_out += f"\t{k}: {d_metrics[k]:.4f}"
        else:
            s_out += f"\t{k}: {d_metrics[k]:.4f}"
    print(s_out)
    print("=" * 80)
    return d_outs


def run_log_one_episode(
    robot: BLPanda,
    ik: CAQPIK,
    q_0,
    T_target,
    pts_traj,
    max_step=200,
    save_video=True,
    exp_ver_prefix=None,
    log_dir=LOG_DIR,
    pcs=None,
    quiet=False,
    args=None,
    log_csv=True,
    log_apitable=False,
    log_traj_info=True,
    b_plt_Gamma_run=True,
):
    exp_ver = exp_ver_prefix + "_" + ik.method
    os.makedirs(log_dir, exist_ok=True)
    env_name = robot.col_env.name

    exp_file_prefix = f"{log_dir}/{exp_ver}"

    d_outs = run_one_episode(
        robot,
        ik,
        q_0,
        T_target,
        pts_traj,
        max_step=max_step,
        save_video=save_video,
        exp_ver=exp_ver,
        log_dir=log_dir,
        pcs=pcs,
    )

    d_outs.update(
        {
            f"Gamma_{ik.method}_run": cal_Gamma_net_run(
                ik=ik, d_outs=d_outs, method=ik.method
            )
        }
    )

    d_log = {
        "test_ver": args.test_ver,
        "exp_ver": args.exp_ver,
        "controller": args.controller,
        "exp_ver_w_method": exp_ver,
        "env_name": env_name,
        "dist_method": ik.method,
        "ik.q_step_limit": ik.q_step_limit,
        "log_time": get_time_str(),
    }
    d_log.update(d_outs["metrics"])

    if log_traj_info:
        traj_info = dict(q_0=q_0, T_target=T_target, pts_traj=pts_traj)
        traj_info["metrics"] = d_outs["metrics"]
        traj_info["q"] = d_outs["q"]
        traj_info["d_log"] = d_log
        traj_info["cli_args"] = asdict(args)
        if "ball" in env_name:
            traj_info.update(d_outs)

        if hasattr(robot.col_env, "to_dict"):
            traj_info["col_env_args"] = robot.col_env.to_dict()
            print("col_env_args", traj_info["col_env_args"])

        write_pickle(f"{log_dir}/{exp_ver}_traj_info.pkl", traj_info)

    if log_csv:
        csv_file = args.csv_file or (DEV_WS_DIR + f"/logs/debug/{env_name}.csv")
        csv_logger = CSVLogger(csv_file, exp_ver)
        csv_logger.log_dict(d_log)

    if log_apitable:
        try:
            dst = get_dst(args.dst_name)
            df = pd.DataFrame([d_log])
            df.set_index("exp_ver_w_method", inplace=True)
            dst_bulk_update_or_create(dst, df)
        except Exception as e:
            print("=" * 80)
            print("Warning: log_apitable failed!")
            print(e)
            print("=" * 80)

    if b_plt_Gamma_run:
        plt_Gamma_run(
            d_outs["Gamma_bl_run"],
            d_outs[f"Gamma_{ik.method}_run"],
            ik.method,
            ik.col_margin,
            ik.col_margin-0.02,
            save_fig=f"{exp_file_prefix}_Gamma_run.png",
        )

    x_target = np_tf.T2tw(T_target)
    if save_video:
        os.makedirs(log_dir, exist_ok=True)
        plt.close()
        plt_ik_step_info(
            d_outs,
            x_target,
            save_fig=f"{exp_file_prefix}_step_info.png",
            is_spheres=False,
        )
        plt.close()
        plt_dist_min_err(d_outs, save_fig=f"{exp_file_prefix}_dist_info.png")
    else:
        plt_ik_step_info(d_outs, x_target)
        plt.pause(1)

    return d_outs


if __name__ == "__main__":
    pass
