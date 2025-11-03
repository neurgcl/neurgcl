import torch


@torch.jit.script
def tf_BT44_Npts(T44, pts):
    """
    Args:
        T44 (torch.Tensor): (B, 4, 4)
        pts (torch.Tensor): (N, 3)
    Returns:
        ret (torch.Tensor): (B, N, 3)
    """
    R = T44[:, :3, :3].unsqueeze(1)
    t = T44[:, :3, [3]].unsqueeze(1)
    ret = (R @ pts.unsqueeze(-1) + t).squeeze(-1)
    return ret


@torch.jit.script
def inv_tf_BT44(T44):
    """
    Args:
        T44 (torch.Tensor): (B, 4, 4)
    """
    RT = T44[:, :3, :3].transpose(-1, -2)
    return torch.cat([RT, -(RT @ T44[:, :3, [3]])], dim=-1)


@torch.jit.script
def inv_tf_BT44_Npts(T44, pts: torch.Tensor):
    """
    Args:
        T44 (torch.Tensor): (B, 4, 4)
        pts (torch.Tensor): (N, 3)
    """
    RT = T44[:, :3, :3].transpose(-1, -2)
    return (RT.unsqueeze(1) @ pts.unsqueeze(-1) - (RT @ T44[:, :3, [3]]).unsqueeze(1)).squeeze(-1)


def tf_pts__ch_3_pt(T, pts):
    """
     R @ pts + t

     T:     n_ch, 4, 4
     pts:   3, n_pt
    ---
     return: n_ch, 3, n_pt
    """
    return (T[:, :3, :3] @ pts) + T[:, :3, [3]]


def tf_pts__ch_pt_3(T, pts):
    """
     R @ pts + t

     T:     n_ch, 3, 4
     pts:   n_pt, 3
    ---
     return: n_ch, n_pt, 3
    """
    # T[:, :3, :3].unsqueeze(1).shape                                       # ch, 1, 3, 3
    # pts.unsqueeze(0).unsqueeze(-1).shape                                  # 1, n_pt, 3, 1
    # (T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)).shape    # ch, n_pt, 3,1

    # T[:, :3, [3]].shape                   # ch, 3, 1
    # T[:, :3, [3]].unsqueeze(1)            # ch, 1, 3, 1

    # ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).shape     # ch, n_pt, 3, 1

    # ((T[:, :3, :3] @ pts.unsqueeze(1).unsqueeze(-1)) + T[:, :3, [3]]).shape  # n_pt, ch, 3, 1
    # ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).shape  # ch, n_pt, 3, 1;  2,4,3,1

    # return ((T[:, :3, :3] @ pts.unsqueeze(1).unsqueeze(-1)) + T[:, :3, [3]]).squeeze(-1)  # n_pt, n_ch,  3
    return ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).squeeze(
        -1
    )  # n_ch, n_pt, 3


def b_pair_tf_pt(T: torch.Tensor, pts: torch.Tensor):
    """
     R @ pts + t

     T:     n_x, n_link, 3, 4
     pts:   n_x, 3
    ---
     return: n_x, n_link, 3
    """
    return (T[:, :, :3, :3] @ pts.unsqueeze(1).unsqueeze(-1) + T[:, :, :3, [3]]).squeeze(-1)


# @torch.jit.script
def b_inv_mat34(T_w2links: torch.Tensor):
    """
    T_w2links: (x,) n_ch, 3, 4
    """
    shape = T_w2links.shape
    T_w2links = T_w2links.view(-1, shape[-2], shape[-1])
    RT = T_w2links[:, :3, :3].transpose(-1, -2)
    return torch.concatenate([RT, -(RT @ T_w2links[:, :3, [3]])], dim=-1).view(shape)


def b_inv_mat44(T_w2links: torch.Tensor):
    """
    T_w2links: x, n_ch, 4, 4
    """
    shape = T_w2links.shape

    T_w2links = T_w2links.view(-1, shape[-2], shape[-1])
    n_tf = T_w2links.shape[0]

    RT = T_w2links[:, :3, :3].transpose(-1, -2)

    T44 = torch.zeros((n_tf, 4, 4), device=T_w2links.device, dtype=T_w2links.dtype)
    T44[:, :3, :3] = RT
    T44[:, :3, [3]] = -(RT @ T_w2links[:, :3, [3]])
    T44[:, 3, 3] = 1
    T44 = T44.view(shape)
    return T44


# @torch.jit.script
def b_mat34mul(T1, T2):
    R = T1[:, :3, :3] @ T2[:, :3, :3]
    t = T1[:, :3, :3] @ T2[:, :3, [3]] + T1[:, :3, [3]]
    return torch.concat((R, t), dim=-1)


# @torch.jit.script
def mat34mul(T1, T2):
    R = T1[:3, :3] @ T2[:3, :3]
    t = T1[:3, :3] @ T2[:3, [3]] + T1[:3, [3]]
    return torch.concat((R, t), dim=-1)

def b_tf_pts_ch_pt(T, pts):
    """ R @ pts + t
    Args:
        T:     (L, 3, 4)
        pts:   (N, 3)
    Return:
        return: (L, N, 3)
    """
    return ((T[:, :3, :3].unsqueeze(1) @ pts.unsqueeze(0).unsqueeze(-1)) + T[:, :3, [3]].unsqueeze(1)).squeeze(-1)  # ch, n_pt, 3