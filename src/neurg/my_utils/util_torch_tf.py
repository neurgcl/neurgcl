import torch



def b_invRt(R, t):
    """
    R: (b, 3, 3), t: (b, 3, 1)
    """
    Rinv = R.transpose(-2, -1)
    tinv = (-Rinv @ t.transpose(-2, -1)).transpose(-2, -1)
    return Rinv, tinv


def b_inv_mat34(T_w2links):
    RT = T_w2links[:, :3, :3].transpose(-1, -2)
    # T_invs = torch.empty_like(T_w2links)
    # T_invs[:, :3, :3] = RT
    # T_invs[:, :3, 3] = -(RT @ T_w2links[:, :3, [3]]).squeeze(-1)
    return torch.concatenate([RT, -(RT @ T_w2links[:, :3, [3]])], dim=-1)


# @torch.jit.script
def inv_mat34(T):
    RT = T[:3, :3].transpose(-1, -2)
    # T_inv = torch.empty_like(T)
    # T_inv[:3, :3] = RT
    # T_inv[:3, 3] = -(RT @ T[:3, [3]]).squeeze(-1)
    # return T_inv
    return torch.concatenate([RT, -(RT @ T[:3, [3]])], dim=-1)


def mat34dotpt(T34, pts):
    """
    T34: (3, 4)
    pts: (3, n)
    """
    return T34[:, :, :3].dot(pts) + T34[:, :, [3]]


# def mat44Inv(T):
#     assert (T.shape[1] == 4 or T.shape[1] == 4) and T.shape[2] == 4
#     R = T[:, :3, :3]
#     p = T[:, :3, [3]]
#     Rt = R.transpose(0, 2, 1)
#     mat34 = np.c_[Rt, -np.matmul(Rt, p)]
#     return np.concatenate([mat34, np.array([[[0, 0, 0, 1]]]).repeat(T.shape[0], axis=0)], axis=1)

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    # Define two rotation matrices as tensors
    R1 = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    R2 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    # Calculate the rotation error between R1 and R2
    cos_theta = (torch.trace(torch.matmul(R1, R2.T)) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)  # Ensure the value is within the range [-1, 1]
    theta = torch.acos(cos_theta)  # Calculate the angle in radians
    rotation_error = theta * 180 / torch.tensor([3.14159265358979323846])  # Convert to degrees

    print(rotation_error.item())  # Output: 180.0