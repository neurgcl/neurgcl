from math import cos, sin

import numpy as np


def Rx(q):
    c = cos(q)
    s = sin(q)
    R = np.eye(3)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def Rx_neg(q):
    c = cos(-q)
    s = sin(-q)
    R = np.eye(3)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def Ry(q):
    c = cos(q)
    s = sin(q)
    R = np.eye(3)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def Ry_neg(q):
    c = cos(-q)
    s = sin(-q)
    R = np.eye(3)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def Rz(q):
    c = cos(q)
    s = sin(q)
    R = np.eye(3)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def Rz_neg(q):
    c = cos(-q)
    s = sin(-q)
    R = np.eye(3)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def axis2idx(axis):
    if axis[0] == 1:
        return 0
    elif axis[0] == -1:
        return 1
    elif axis[1] == 1:
        return 2
    elif axis[1] == -1:
        return 3
    elif axis[2] == 1:
        return 4
    else:
        return 5


# l_R = [Rx, Rx_neg, Ry, Ry_neg, Rz, Rz_neg]


# def link2T(T_origin, q, axis):
#     # R = l_R[axis](q)
#     if axis == 4:  # z
#         c = cos(q)
#         s = sin(q)
#         # R = np.eye(3)
#         # R[0, 0] = c
#         # R[0, 1] = -s
#         # R[1, 0] = s
#         # R[1, 1] = c
#         R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
#     elif axis == 5:  # -z
#         c = cos(-q)
#         s = sin(-q)
#         # R = np.eye(3)
#         # R[0, 0] = c
#         # R[0, 1] = -s
#         # R[1, 0] = s
#         # R[1, 1] = c
#         R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
#     elif axis == 2:  # y
#         c = cos(q)
#         s = sin(q)
#         # R = np.eye(3)
#         # R[0, 0] = c
#         # R[0, 2] = s
#         # R[2, 0] = -s
#         # R[2, 2] = c
#         R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
#     elif axis == 3:  # -y
#         c = cos(-q)
#         s = sin(-q)
#         # R = np.eye(3)
#         # R[0, 0] = c
#         # R[0, 2] = s
#         # R[2, 0] = -s
#         # R[2, 2] = c
#         R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
#     elif axis == 0:  # x
#         c = cos(q)
#         s = sin(q)
#         # R = np.eye(3)
#         # R[1, 1] = c
#         # R[1, 2] = -s
#         # R[2, 1] = s
#         # R[2, 2] = c
#         R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
#     else:
#         c = cos(-q)
#         s = sin(-q)
#         # R = np.eye(3)
#         # R[1, 1] = c
#         # R[1, 2] = -s
#         # R[2, 1] = s
#         # R[2, 2] = c
#         R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

#     T1 = np.eye(4)
#     T1[:3, :3] = T_origin[:3, :3] @ R
#     T1[:3, 3] = T_origin[:3, 3]
#     return T1


def link2T(T_origin, q, axis):
    # R = l_R[axis](q)
    if axis == 4:  # z
        c = cos(q)
        s = sin(q)
        # R = np.eye(3)
        # R[0, 0] = c
        # R[0, 1] = -s
        # R[1, 0] = s
        # R[1, 1] = c
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == 5:  # -z
        c = cos(-q)
        s = sin(-q)
        # R = np.eye(3)
        # R[0, 0] = c
        # R[0, 1] = -s
        # R[1, 0] = s
        # R[1, 1] = c
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == 2:  # y
        c = cos(q)
        s = sin(q)
        # R = np.eye(3)
        # R[0, 0] = c
        # R[0, 2] = s
        # R[2, 0] = -s
        # R[2, 2] = c
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 3:  # -y
        c = cos(-q)
        s = sin(-q)
        # R = np.eye(3)
        # R[0, 0] = c
        # R[0, 2] = s
        # R[2, 0] = -s
        # R[2, 2] = c
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 0:  # x
        c = cos(q)
        s = sin(q)
        # R = np.eye(3)
        # R[1, 1] = c
        # R[1, 2] = -s
        # R[2, 1] = s
        # R[2, 2] = c
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        c = cos(-q)
        s = sin(-q)
        # R = np.eye(3)
        # R[1, 1] = c
        # R[1, 2] = -s
        # R[2, 1] = s
        # R[2, 2] = c
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    T = np.eye(4)
    T[:3, :3] = T_origin[:3, :3] @ R
    T[:3, 3] = T_origin[:3, 3]
    return T
