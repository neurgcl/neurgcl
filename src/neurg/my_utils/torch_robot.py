"""
b: the first dim is batch 
"""

from typing import Final, List
import numpy as np
from torch import cos, sin
import torch

from neurg.my_utils.torch_tf import b_mat34mul, mat34mul


def b_inv_tfs(T_w2links: torch.Tensor):
    """
    T_w2links: (x,) n_ch, 3, 4
    """
    shape = T_w2links.shape
    T_w2links = T_w2links.view(-1, shape[-2], shape[-1])

    T_invs = torch.empty_like(T_w2links)
    RT = T_w2links[:, :3, :3].transpose(-1, -2)
    T_invs[:, :3, :3] = RT
    T_invs[:, :3, 3] = -(RT @ T_w2links[:, :3, [3]]).squeeze(-1)
    T_invs[:, 3, :3] = 0
    T_invs[:, 3, 3] = 1
    return T_invs.view(shape)


def b_inv_mat34(T_w2links: torch.Tensor):
    """
    T_w2links: (x,) n_ch, 3, 4
    """
    shape = T_w2links.shape
    T_w2links = T_w2links.view(-1, shape[-2], shape[-1])
    RT = T_w2links[:, :3, :3].transpose(-1, -2)
    return torch.concatenate([RT, -(RT @ T_w2links[:, :3, [3]])], dim=-1).view(shape)


def b_tfs_pts(T: torch.Tensor, pts: torch.Tensor):
    """
     R @ pts + t

     T:     n_ch, x, 3, 4
     pts:   n_pt, 3
    ---
     return: n_ch, x, n_pt, 3
    """
    shape = T.shape
    #
    T = T.view(-1, shape[-2], shape[-1])  # ch*x, 3, 4
    # ch*x, 1, 3, 3 @ 1,n_pt,3,1   -> ch*x, n_pt, 3, 1
    #                              -> ch, x, n_pt, 3

    # n_ch, x, n_pt, 3
    return (T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1) + T[:, :3, [3]].unsqueeze(1)).view(
        shape[0], shape[1], pts.shape[0], pts.shape[1]
    )

    # # n_ch, x*n_pt, 3
    # return ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).view(shape[0], -1, 3)


def batch_tf_pt(T: torch.Tensor, pts: torch.Tensor):
    """
     R @ pts + t

     T:     n_x, n_link, 3, 4
     pts:   n_x, 3
    ---
     return: n_x, n_link, 3
    """
    return (T[:, :, :3, :3] @ pts.unsqueeze(1).unsqueeze(-1) + T[:, :, :3, [3]]).squeeze(-1)


def b_tf_pts_ch_pt(T, pts):
    """
     R @ pts + t

     T:     n_ch, 3, 4
     R:     n_ch, 3, 3
     pts:   batch, 3
    ---
     return: n_ch, n_pt, 3
    """
    # T[:, :3, :3].unsqueeze(1).shape                                       # ch, 1, 3, 3
    # pts.unsqueeze(0).unsqueeze(-1).shape                                  # 1, n_pt, 3, 1
    # (T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)).shape    # ch, n_pt, 3,1

    # T[:, :3, [3]].shape                   # ch, 3, 1
    # T[:, :3, [3]].unsqueeze(1)            # ch, 1, 3, 1

    # ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).shape             # ch, n_pt, 3, 1
    return ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).squeeze(
        -1
    )  # ch, n_pt, 3


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
    J = -R_links2w @ (torch.linalg.cross(b_joint_rot_axis, (b_point_t - b_joint_t), dim=-2)).unsqueeze(0)

    # zero_mask (ch, 7) = (1, 7)  >= (ch, 1)       | (8, 7)  = (1, 7) >= (8, 1)
    zero_mask = torch.arange(7)[None, :] - 1 >= torch.arange(R_links2w.shape[0])[:, None]
    zero_mask = zero_mask[:, None, None, None, :].expand(-1, J.shape[1], J.shape[2], J.shape[3], -1)
    J[zero_mask] = 0
    return J


def cal_J_all_local_wrt_q(R_links2w, b_joint_rot_axis, b_point_t, b_joint_t):
    """
    all links, remove link0 (J_link0_wrt_q = 0)

    R_links2w:          n_joint+1,3,3     n_link-1,3,3     =T_links2w[1:, :3, :3]
    b_joint_rot_axis:   n_joint, 3          7, 3            =T_w2links[1:8, :3, 2]
    b_point_t:          n_pts, 3            batch, 3        =pts
    b_joint_t:          n_joint, 3          7, 3            =T_w2links[1:8, :3, 3]
    """
    # (1, 7, 3) x (batch , 7, 3)  = (batch,7, 3)  --transpose--> (batch, 3, 7)
    # J: ch, batch, 3,7
    J = -R_links2w.unsqueeze(1) @ (
        torch.linalg.cross(b_joint_rot_axis.unsqueeze(0), (b_point_t.unsqueeze(1) - b_joint_t), dim=-1)
    ).transpose(1, 2)

    # zero_mask (ch, 7) = (1, 7)  >= (ch, 1)       | (8, 7)  = (1, 7) >= (8, 1)
    zero_mask = torch.arange(7)[None, :] - 1 >= torch.arange(R_links2w.shape[0])[:, None]
    J = J.transpose(0, -2)  # 3, batch, ch, 7
    J[..., zero_mask] = 0
    J = J.transpose(0, -2)  # ch, batch, 3, 7
    return J


class RobotConfig(torch.nn.Module):
    DH_Params: torch.Tensor
    q_bounds: torch.Tensor
    q_extents: torch.Tensor
    T_link7tohand: torch.Tensor
    T_link7toee: torch.Tensor
    n_act_joint: Final[int] = 7
    n_col_link: Final[int] = n_act_joint + 2  # base_link + 7 act_joint + ee_link

    link_names: Final[str] = [
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
    ]

    def __init__(self):
        super(RobotConfig, self).__init__()

        dtype = torch.float32

        # r, d, alpha
        DH_Params = np.array(
            [
                [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0],
                [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107],
                [
                    0,
                    -np.pi / 2,
                    np.pi / 2,
                    np.pi / 2,
                    -np.pi / 2,
                    np.pi / 2,
                    np.pi / 2,
                    0,
                ],
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
        T_link7tohand = DH2mat34(
            DH_Params[0][i],
            DH_Params[1][i],
            DH_Params[2][i],
            torch.tensor(-torch.pi / 4),
        )
        T_link7toee = DH2mat34(
            DH_Params[0][i],
            DH_Params[1][i] + 0.065,
            DH_Params[2][i],
            torch.tensor(-torch.pi / 4),
        )

        T_link7tohand = T_link7tohand.unsqueeze(0)
        T_link7toee = T_link7toee.unsqueeze(0)

        for k in ["DH_Params", "q_bounds", "q_extents", "T_link7tohand", "T_link7toee"]:
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
                tmp_T,
                bDH2mat34(
                    self.DH_Params[0][i],
                    self.DH_Params[1][i],
                    self.DH_Params[2][i],
                    q[:, i],
                ),
            )
            T_outs.append(tmp_T)

        tmp_T = b_mat34mul(tmp_T, self.T_link7tohand)
        T_outs.append(tmp_T)

        tmp_T = b_mat34mul(tmp_T, self.T_link7toee)
        T_outs.append(tmp_T)
        T_outs = torch.stack(T_outs, dim=link_dim)
        return T_outs

    # def cal_fk_link_first(self, q: torch.tensor):
    #     """ Return mat34 of: 0, link0, link1, ..., link7, hand, ee
    #     q: (batch_sz, n_act_joint)
    #     return: (batch_sz, n_col_link, 3, 4)
    #     """
    #     T_outs = []

    #     tmp_T = torch.zeros((q.shape[0], 3, 4), device=q.device)
    #     tmp_T[:, :3, :3] = torch.eye(3, device=q.device)
    #     T_outs.append(tmp_T)
    #     for i in range(self.n_act_joint):
    #         tmp_T = b_mat34mul(tmp_T, bDH2mat34(self.DH_Params[0][i], self.DH_Params[1][i], self.DH_Params[2][i], q[:, i]))
    #         T_outs.append(tmp_T)

    #     tmp_T = b_mat34mul(tmp_T,  self.T_link7tohand)
    #     T_outs.append(tmp_T)

    #     tmp_T = b_mat34mul(tmp_T,  self.T_link7toee)
    #     T_outs.append(tmp_T)
    #     T_outs = torch.stack(T_outs, dim=0)
    #     return T_outs

    @torch.jit.export
    def get_random_q(self, n: int = 1):
        return self.q_bounds[0] + torch.rand((n, len(self.q_extents)), device=self.q_extents.device) * self.q_extents


if __name__ == "__main__":
    robot_cfg = RobotConfig()
    q = robot_cfg.get_random_q(2)
    tfs = robot_cfg(q)
    print(tfs.shape)
