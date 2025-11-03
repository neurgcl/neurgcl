from typing import Final

import numpy as np
import torch
from torch import cos, sin

from neurg.my_utils.torch_tf import b_mat34mul


def DH2mat34(a, d, alpha, q):
    # arr = [[cos(q),             -sin(q),            0,              a],
    #        [sin(q)*cos(alpha),  cos(q)*cos(alpha),  -sin(alpha),    -d*sin(alpha)],
    #        [sin(q)*sin(alpha),  cos(q)*sin(alpha),  cos(alpha),     d*cos(alpha)],
    #        [0, 0, 0, 1]]
    cq = cos(q)
    sq = sin(q)
    ca = cos(alpha)
    sa = sin(alpha)

    arr = torch.zeros((3, 4), device=q.device)
    arr[0, 0] = cq
    arr[0, 1] = -sq
    # arr[0, 2] = 0
    arr[0, 3] = a

    arr[1, 0] = sq * ca
    arr[1, 1] = cq * ca
    arr[1, 2] = -sa
    arr[1, 3] = -d * sa

    arr[2, 0] = sq * sa
    arr[2, 1] = cq * sa
    arr[2, 2] = ca
    arr[2, 3] = d * ca

    # arr[3, 0] = 0
    # arr[3, 1] = 0
    # arr[3, 2] = 0
    # arr[3, 3] = 1
    return arr


def bDH2mat34(a, d, alpha, q):
    # arr = [[cos(q),             -sin(q),            0,              a],
    #        [sin(q)*cos(alpha),  cos(q)*cos(alpha),  -sin(alpha),    -d*sin(alpha)],
    #        [sin(q)*sin(alpha),  cos(q)*sin(alpha),  cos(alpha),     d*cos(alpha)],
    #        [0, 0, 0, 1]]
    cq = cos(q)
    sq = sin(q)
    ca = cos(alpha)
    sa = sin(alpha)

    arr = torch.zeros((len(q), 3, 4), device=q.device)
    arr[:, 0, 0] = cq
    arr[:, 0, 1] = -sq
    # arr[0, 2] = 0
    arr[:, 0, 3] = a

    arr[:, 1, 0] = sq * ca
    arr[:, 1, 1] = cq * ca
    arr[:, 1, 2] = -sa
    arr[:, 1, 3] = -d * sa

    arr[:, 2, 0] = sq * sa
    arr[:, 2, 1] = cq * sa
    arr[:, 2, 2] = ca
    arr[:, 2, 3] = d * ca
    return arr


def zero_invalid_J(J, minIdx):
    zero_mask = torch.arange(7, device=J.device)[None, :] >= minIdx[:, None]  # batch, 7
    J = J.transpose(0, 1)
    J[:, zero_mask] = 0
    J = J.transpose(0, 1)
    return J


def bq_cal_J_local_wrt_q(R_links2w, b_joint_rot_axis, b_joint_t, b_point_t):
    """
    batch q, calculate jacobian of point wrt joint angle, remove link0 (J_link0_wrt_q = 0)
    Inputs:
        R_links2w = T_links2w[1:, :, :3, :3]         # ch, n, 3, 3
        b_joint_rot_axis = T_w2links[1:8, :, :3, 2]  # act, n, 3
        b_joint_t = T_w2links[1:8, :, :3, 3]         # act, q, 3
        b_point_t = pts                              # b, 3
    Return:
        J:  ch, n, b, 3, act
    """
    R_links2w = R_links2w.unsqueeze(-3)  # ch, n, 1, 3, 3
    b_joint_rot_axis = b_joint_rot_axis.permute(1, 2, 0).unsqueeze(-3)  # n, 1, 3, act
    b_joint_t = b_joint_t.permute(1, 2, 0).unsqueeze(-3)  # n, 1, 3, act
    b_point_t = b_point_t.unsqueeze(-1).unsqueeze(0)  # 1, b, 3, 1

    # jacobian of point wrt joint angle
    # (ch, n, b, 3, act) = (ch, n, 1, 3, 3) @ (1, n, b, 3, act)
    J = -R_links2w @ (torch.cross(b_joint_rot_axis, (b_point_t - b_joint_t))).unsqueeze(0)

    # zero_mask (ch, 7) = (1, 7)  >= (ch, 1)       | (8, 7)  = (1, 7) >= (8, 1)
    zero_mask = torch.arange(7)[None, :] - 1 >= torch.arange(R_links2w.shape[0])[:, None]
    zero_mask = zero_mask[:, None, None, None, :].expand(-1, J.shape[1], J.shape[2], J.shape[3], -1)
    J[zero_mask] = 0
    # J = torch.concatenate([torch.zeros_like(J[[0]]), J])
    return J


class TorDHRobotKin(torch.nn.Module):
    DH_Params: torch.Tensor
    q_bounds: torch.Tensor
    q_extents: torch.Tensor
    T_link7tohand: torch.Tensor
    T_handtoee: torch.Tensor
    n_act_joint: Final[int] = 7
    n_col_link: Final[int] = n_act_joint + 2  # base_link + 7 act_joint + ee_link

    link_names = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand']

    def __init__(self):
        super().__init__()

        device = None
        dtype = torch.float32

        # r, d, alpha
        DH_Params = np.array(
            [
                [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0],
                [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107],
                [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0],
            ]
        )

        q_bounds = np.array(
            [
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04],
            ]
        )
        q_bounds = q_bounds[:, : self.n_act_joint]
        q_extents = q_bounds[1] - q_bounds[0]

        DH_Params = torch.tensor(DH_Params, dtype=dtype)
        q_bounds = torch.tensor(q_bounds, dtype=dtype)
        q_extents = torch.tensor(q_extents, dtype=dtype)

        i = 7
        T_link7tohand = DH2mat34(DH_Params[0][i], DH_Params[1][i], DH_Params[2][i], torch.tensor(-torch.pi / 4))
        T_handtoee = DH2mat34(*torch.tensor([0.0, 0.1034, 0.0, 0.0], dtype=dtype))

        T_link7tohand = T_link7tohand.unsqueeze(0)
        T_handtoee = T_handtoee.unsqueeze(0)

        for k in ['DH_Params', 'q_bounds', 'q_extents', 'T_link7tohand', 'T_handtoee']:
            v = locals()[k]
            self.register_buffer(k, v)
        # self.register_buffer('DH_Params', DH_Params)
        # self.register_buffer('q_bounds', q_bounds)
        # self.register_buffer('q_extents', q_extents)
        # self.register_buffer('T_link7toee', T_link7toee)

    def forward(self, q: torch.Tensor, link_dim: int = 0):
        """Return mat34 of: 0, link0, link1, ..., link7, hand, ee
        Input:
            q: (batch_sz, n_act_joint)

        Return:
            link_dim=0: (n_col_link, batch_sz, 3, 4)
            link_dim=1: (batch_sz, n_col_link, 3, 4)
        """
        T_outs = []

        tmp_T = torch.zeros((q.shape[0], 3, 4), device=q.device)
        tmp_T[:, :3, :3] = torch.eye(3, device=q.device)
        T_outs.append(tmp_T)
        for i in range(self.n_act_joint):
            tmp_T = b_mat34mul(
                tmp_T, bDH2mat34(self.DH_Params[0][i], self.DH_Params[1][i], self.DH_Params[2][i], q[:, i])
            )
            T_outs.append(tmp_T)

        tmp_T = b_mat34mul(tmp_T, self.T_link7tohand)
        T_outs.append(tmp_T)

        tmp_T = b_mat34mul(tmp_T, self.T_handtoee)
        T_outs.append(tmp_T)
        T_outs = torch.stack(T_outs, dim=link_dim)
        return T_outs

    @torch.jit.export
    def get_random_q(self, n: int = 1):
        return self.q_bounds[0] + torch.rand((n, len(self.q_extents)), device=self.q_extents.device) * self.q_extents

    def fk_links_T44(self, q: torch.Tensor):
        """Return mat44 of: 0, link0, link1, ..., link7, hand, ee
        Input:
            q: (batch_sz, n_act_joint)

        Return:
            (batch_sz, n_col_link, 4, 4)
        """
        T_outs = self.forward(q, link_dim=1)
        return T_outs

    def fk_col_link_T44(self, q: torch.Tensor):
        """
        only col links
        Input:
            q: (batch_sz, n_act_joint)

        Return:
            (batch_sz, n_col_link, 4, 4)
        """
        T_outs = self.forward(q, link_dim=1)[:, :8].contiguous()
        return T_outs


if __name__ == "__main__":
    from neurg.my_utils import np_tf

    robot_cfg = TorDHRobotKin()
    q = robot_cfg.get_random_q(1)
    tfs = robot_cfg(q)
    print(tfs.shape)

    # fmt: off
    q_0 = np.array(
        [-0.0965195386873646, -0.48818972252726783, -0.5262399291814241, -2.0770811368204902,
         -0.23848617636484631, 1.6444697759066003, 0.23276192924707884])
    q_target = np.array(
        [0.09651953869727141, -0.48818972254287984, 0.5262399291691557, -2.0770811368275495, 
         0.23848617638248512, 1.6444697759146427, 1.3380343975414601])
    # fmt: on
    q = torch.tensor(q_0, dtype=torch.float32).unsqueeze(0)
    tfs = robot_cfg(q)
    np_tfs = tfs.cpu().numpy()
    print(np_tfs.shape)
    print(np_tf.T2xyzrpy(tfs[-2, 0].cpu().numpy(), degrees=True).tolist())
    print(np_tf.T2xyzrpy(tfs[-1, 0].cpu().numpy(), degrees=True).tolist())
