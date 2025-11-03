import time
from dataclasses import dataclass
from typing import Final

import casadi as ca
import numpy as np
import torch
from casadi import DM

from neurg.my_utils import np_tf
from neurg.my_utils.np_tf import R2rv, T2tR
from qp_ik.dist_fun import QPBaseNet, QPJSDFNet, QPNSDFNet, QPRDFNet
from qp_ik.utils import robot_kin


def cal_err_Gamma(Gamma, col_r, col_margin):
    return np.log(np.maximum((Gamma - col_r - col_margin) + 1, 1e-7))


@dataclass
class PandaConfig:
    n_joint: Final[int] = 7
    l_q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0]
    l_q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04]

    q_min = np.array(l_q_min[:n_joint])
    q_max = np.array(l_q_max[:n_joint])

    q_step_limit = np.deg2rad(1)  # deg
    q_vel_limit = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0])  # ros fr3
    # q_vel_limit = np.array([2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 2.6100]) # ros panda


@dataclass
class QPConfig:
    vel_scale: float = 0.5

    n_joint = PandaConfig.n_joint
    q_min = PandaConfig.q_min
    q_max = PandaConfig.q_max
    q_step_limit: float = PandaConfig.q_step_limit
    q_vel_limit: np.ndarray = PandaConfig.q_vel_limit * vel_scale

    dim_delta = 6
    dim_dq = n_joint
    dim_x = dim_dq + dim_delta

    p_slack: float = 5
    p_daming: float = 1

    qp_k = 8  # n_col_links used in qp
    col_margin: float = 0.1
    col_r: float = 0.0

    ignore_vel_limit: bool = True
    check_col: bool = True
    filter_Gamma: bool = False


class CAQPIK(QPConfig):
    def __init__(self, device="cuda", col_type="ball", qp_cfg=None) -> None:
        super().__init__()
        self.col_type = col_type

        # --------------------------
        self.device = device
        self.dtype = torch.float32
        self.tensor_args = {"device": self.device, "dtype": self.dtype}
        self.aot = False
        self.remove_base = True
        self.n_col_links = 9

        # self.nsdf_net.aot_dists_g_with_tf = aot_ts_compile(self.nsdf_net.dists_g_with_tf)

        # self.nsdf_net = RobotSDFNet()
        # self.nsdf_net.prepare_eval(self.device, self.dtype)
        # self.nsdf_net.update_aot()
        self.qp_nsdf_net = QPNSDFNet()
        self.qp_jsdf_net = QPJSDFNet()
        self.qp_rdf_net = QPRDFNet()

        self.init_QP()

        self.method = "nsdf"
        self.get_Gamma_with_g = self.qp_nsdf_net.cal_Gamma_g_all

    def check_q_bound_ratio(self, q):
        return (q - self.q_min) / (self.q_max - self.q_min)

    def update_cfg(self, qp_cfg):
        for k in qp_cfg:
            setattr(self, k, qp_cfg[k])

    def get_Gamma_net(self, method) -> QPBaseNet:
        if method == "nsdf":
            return self.qp_nsdf_net
        elif method == "jsdf":
            return self.qp_jsdf_net
        elif method == "rdf":
            return self.qp_rdf_net
        else:
            raise ValueError(f"method {method} not supported")

    def change_method(self, method):
        self.method = method
        qp_net = self.get_Gamma_net(method)
        self.get_Gamma_with_g = qp_net.cal_Gamma_g_all

    def cal_Gamma_g_all(self, q, pts, T_w2links=None, method="nsdf"):
        qp_net = self.get_Gamma_net(method)
        return qp_net.cal_Gamma_g_all(q, pts, T_w2links)

    def cal_Gamma_g_min(self, q, pts, T_w2links=None, method="nsdf"):
        qp_net = self.get_Gamma_net(method)
        return qp_net.cal_Gamma_g_min(q, pts, T_w2links)

    def warm_up(self, n_pts=10000):
        n_repeat = max(min(1000, 2000 * 1000 // n_pts), 5)
        if self.method == "rdf":
            n_repeat = max(min(500, 2000 * 200 // n_pts), 5)
        for _i in range(n_repeat):
            q = np.random.rand(7) * 2 * np.pi - np.pi
            pts = np.random.rand(n_pts, 3) * 0.1
            Gammp, DGamma = self.get_Gamma_with_g(q, pts)
            torch.cuda.synchronize()

    def set_goal(self, T_target):
        t_target, R_target = T2tR(T_target)
        self.t_target = t_target
        self.R_target = R_target

    def cal_vars_fk(self, q, t_target, R_target):
        T_w2links, J_ee = robot_kin.tor_fk_tf_jacos(q)

        T_w2ee = T_w2links[-1].cpu().numpy()
        J_ee = J_ee.cpu().numpy()

        t_cur = T_w2ee[:3, 3]
        R_cur = T_w2ee[:3, :3]

        t_err = t_target[:3] - t_cur
        w_err = R_cur @ R2rv(R_cur.T @ R_target)  # in current frame, Rcur ->Rtarget
        ee_err = np.r_[t_err, w_err]  # T_cur

        return ee_err, J_ee, T_w2links[:-1], T_w2ee

    def cal_Gamma_g(self, q, pts, T_w2links):
        Gamma, DGamma = self.get_Gamma_with_g(q, pts, T_w2links)

        if self.filter_Gamma:
            idx_Gamma_mask = Gamma < 0.5
            Gamma = Gamma[idx_Gamma_mask]
            DGamma = DGamma[idx_Gamma_mask, :]

        return Gamma, DGamma

    def cal_tmp_vars(self, q, t_target, R_target, pts):
        T_w2links, J_ee = robot_kin.tor_fk_tf_jacos(q)

        T_w2ee = T_w2links[-1].cpu().numpy()
        J_ee = J_ee.cpu().numpy()
        T_w2links = T_w2links[:-1]

        t_cur = T_w2ee[:3, 3]
        R_cur = T_w2ee[:3, :3]

        t_err = t_target[:3] - t_cur
        w_err = R_cur @ R2rv(R_cur.T @ R_target)  # in current frame, Rcur ->Rtarget
        ee_err = np.r_[t_err, w_err]  # T_cur

        Gamma, DGamma = self.get_Gamma_with_g(q, pts, T_w2links)
        return ee_err, J_ee, Gamma, DGamma

    def init_QP(self):
        dim_dq = self.dim_dq
        dim_delta = self.dim_delta
        dim_x = self.dim_x

        # qp H
        qp_H = DM.eye(dim_x)
        qp_H[:dim_dq, :dim_dq] = DM.eye(dim_dq) * self.p_daming
        qp_H[dim_dq:, dim_dq:] = DM.eye(dim_delta) * self.p_slack

        self.qp_H = qp_H
        self.qp_g = DM.zeros(dim_x)

        # q_cur = (self.q_min+self.q_max)/2
        self.lbx = DM.zeros(dim_x)
        # self.lbx[:dim_dq] = np.maximum(-self.q_step_limit, self.q_min - q_cur) # need update when q_cur changed
        self.lbx[:dim_dq] = self.q_min
        self.lbx[dim_delta:] = -ca.inf
        self.ubx = DM.zeros(dim_x)
        # self.ubx[:dim_dq] = np.minimum(self.q_step_limit, self.q_max - q_cur) # need update when q_cur changed
        self.ubx[:dim_dq] = self.q_max
        self.ubx[dim_delta:] = ca.inf

        self.qp_options = {
            "printLevel": "none",
            # "jit":True,
            # "sparse": True
        }

    def solve_qp(self, q, pts, T_target):
        """
        q:          current q pos
        T_target:   ee goal
        pts:        collision pts
        """
        t00 = time.perf_counter()

        self.set_goal(T_target)
        t_target, R_target = self.t_target, self.R_target

        t_fk0 = time.perf_counter()
        ee_err, J_ee, T_w2links, T_w2ee = self.cal_vars_fk(q, t_target, R_target)
        t_fk1 = time.perf_counter()

        Gamma, J_Gamma = self.cal_Gamma_g(q, pts, T_w2links)
        t_gamma_J1 = time.perf_counter()

        # QP ----------------------------------------------
        # scalars
        dim_delta = self.dim_delta
        dim_dq = self.dim_dq
        n_col_pair = len(Gamma)

        # qp lb,ub
        self.lbx[:dim_dq] = np.maximum(self.q_min - q, -self.q_step_limit)
        self.ubx[:dim_dq] = np.minimum(self.q_max - q, self.q_step_limit)

        # qp A
        qp_A = DM.zeros(dim_delta + n_col_pair, dim_dq + dim_delta)
        qp_A[:dim_delta, :dim_dq] = J_ee
        qp_A[:dim_delta, dim_dq:] = DM.eye(dim_delta)

        if n_col_pair > 0:
            qp_A[dim_delta : dim_delta + n_col_pair, :dim_dq] = J_Gamma

        # lbA, ubA
        qp_lbA = DM.ones(dim_delta + n_col_pair) * -ca.inf
        qp_ubA = DM.ones(dim_delta + n_col_pair) * ca.inf
        qp_lbA[:dim_delta] = ee_err
        qp_ubA[:dim_delta] = ee_err
        if n_col_pair > 0:
            log_Gamma_err = -np.log(
                np.maximum((Gamma - self.col_r - self.col_margin) + 1, 1e-7)
            )
            qp_lbA[dim_delta:] = log_Gamma_err

        # build qp
        t_qp_init_vars0 = time.perf_counter()
        qp = {}
        qp["h"] = self.qp_H.sparsity()
        qp["a"] = qp_A.sparsity()
        t_qp_init_vars1 = time.perf_counter()

        # opt --------------------------------------------------
        S = ca.conic("S", "qpoases", qp, self.qp_options)
        t_qp_init1 = time.perf_counter()

        r = S(
            h=self.qp_H,
            g=self.qp_g,
            a=qp_A,
            lba=qp_lbA,
            uba=qp_ubA,
            lbx=self.lbx,
            ubx=self.ubx,
        )

        x_opt = r["x"]
        ret_dq = x_opt[:dim_dq]
        ret_delta = x_opt[dim_dq:]
        t99 = time.perf_counter()

        # step info
        if len(Gamma) == 0:
            Gamma_min = 0.55
        else:
            Gamma_min = Gamma.min()

        delta_t_norm = np.linalg.norm(ret_delta[:3])
        delta_r_norm = np.linalg.norm(ret_delta[3:])
        p_x_cur = np_tf.T2tw(T_w2ee)
        step_info = {
            # qp
            "delta": np.array(ret_delta).squeeze(),
            "delta_t_norm": np.array(delta_t_norm).squeeze(),
            "delta_r_norm": np.array(delta_r_norm).squeeze(),
            # "g_fk": np.array(g_fk).squeeze(),
            # "g_Gamma_log": np.array(g_Gamma_log).squeeze(),
            # "g_Gamma": np.array(g_Gamma).squeeze(),
            # states in current step
            "x": np.array(p_x_cur),  # trans, rotvec
            "q": np.array(q).squeeze(),
            "dq": np.array(ret_dq).squeeze(),
            # "q_next": np.array(q_next).squeeze(),
            # "pts_next": np.array(pts_next).squeeze(),
            # "Gamma_next": np.array(Gamma_next).squeeze(),
            "ee_err": np.array(ee_err).squeeze(),
            # "err_Dfk": np.array(err_Dfk).squeeze(),
            "Gamma_min": Gamma_min,
        }
        t_step_info = time.perf_counter()

        extra_info = dict(
            dt_qp_all=t99 - t00,
            dt_qp_fk=t_fk1 - t_fk0,
            dt_qp_gamma_J=t_gamma_J1 - t_fk1,
            dt_qp_init_vars=t_qp_init_vars1 - t_qp_init_vars0,
            dt_qp_init=t_qp_init1 - t_qp_init_vars1,
            dt_qp_solve=t99 - t_qp_init1,
            dt_step_info=t_step_info - t99,
        )
        extra_info["qp_d_min"] = Gamma_min

        out = ""
        for k in extra_info:
            if k.startswith("dt_"):
                out += f"{k}: {extra_info[k]*1000:.3f}, \t"
            else:
                out += f"{k}: {extra_info[k]:.3f}, \t"
        print(out)
        r["extra"] = extra_info

        return ret_dq, ret_delta, r, step_info

    def limit_vel(self, ret_dq, step_time):
        if self.ignore_vel_limit:
            dq_limited = ret_dq
        else:
            dq_lim = self.q_vel_limit
            scale = ret_dq / (dq_lim * step_time)
            scale_max = max(1, np.abs(scale).max())
            dq_limited = ret_dq / scale_max
        return dq_limited
