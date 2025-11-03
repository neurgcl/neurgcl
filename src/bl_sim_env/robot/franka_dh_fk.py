from math import cos, sin
import math

import numpy as np

from ..utils.np_tf import euler2R

# Joint 1-7, Flange: 8 frames
r = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
alpha = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]

"""
r,=,[0,0,0,0.0825,-0.0825,0,0.088,0];
d,=,[0.333,0,0.316,0,0.384,0,0,0.107];
alpha,=,[0,-pi/2,pi/2,pi/2,-pi/2,pi/2,pi/2,0];
"""

# fmt:off
def A(r, d, alpha, q):
    arr = [[math.cos(q), - math.sin(q)*math.cos(alpha), math.sin(q)*math.sin(alpha), r*math.cos(q)],
           [math.sin(q),	math.cos(q)*math.cos(alpha), - math.cos(q)*math.sin(alpha),  r*math.sin(q)],
           [0,       math.sin(alpha),          math.cos(alpha),          d],
           [0,       0,                   0,                   1]]
    return np.array(arr)


def Am(r, d, alpha, q):
    arr = [[math.cos(q), - math.sin(q), 0, r],
           [math.sin(q)*math.cos(alpha), math.cos(q)*math.cos(alpha), - math.sin(alpha), - d*math.sin(alpha)],
           [math.sin(q)*math.sin(alpha), math.cos(q)*math.sin(alpha), math.cos(alpha), d*math.cos(alpha)],
           [0,    0,   0,   1]]
    return np.array(arr)
# fmt:on


def franka_dh_fk(j_state, r=r, d=d, alpha=alpha, base=np.eye(4)):
    # % base matrix holds index 1 instead of 0 (because matlab)
    P = [None] * 11
    P[0] = base

    # kinematic chain for 7 joints
    for i in range(1, 8):
        P[i] = np.dot(P[i - 1], Am(r[i - 1], d[i - 1], alpha[i - 1], j_state[i - 1]))
        # print(f"\n{i}: {P[i]}")

    # % transformation for hand end-effector, no joint movement here
    # % -pi/4 is a mesh rotation, found manually, not specified anywhere
    P[8] = np.dot(P[7], Am(r[-1], d[-1], alpha[-1], -np.pi / 4))

    # % final two for fingers - inspired by:
    # % https: // github.com/marcocognetti/FrankaEmikaPandaDynModel/blob/master/matlab/utils/LoadFrankaSTLModel.m

    # % finger1
    T = np.eye(4)
    T[:3, 3] = np.array([0, j_state[7], 0.065])
    P[9] = np.dot(P[8], T)

    # % finger2
    T = np.eye(4)
    T[:3, :3] = euler2R(np.deg2rad([180, 0, 0]), seq="ZYX")
    T[:3, 3] = np.array([0, -j_state[7], 0.065])
    P[10] = np.dot(P[8], T)

    return P


def DH2T(a, d, alpha, q):
    arr = [
        [cos(q), -sin(q), 0, a],
        [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -d * sin(alpha)],
        [sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha), d * cos(alpha)],
        [0, 0, 0, 1],
    ]
    return arr


DH_franka = np.array([r, d, alpha])


def fk_tfs(q):
    n_link = 9

    T_outs = np.empty((n_link, 4, 4))
    tmp_T = np.eye(4)
    T_outs[0] = tmp_T
    for i in range(7):
        T_outs[1 + i] = T_outs[i] @ Am(r[i], d[i], alpha[i], q[i])

    i = 7
    T_outs[1 + i] = T_outs[i] @ Am(r[i], d[i], alpha[i], -np.pi / 4)  # panda_hand
    return T_outs


def fk_tf_jacos(q, idx_link=-1):
    T_w2links = fk_tfs(q)

    point_t = T_w2links[idx_link][:3, 3]
    J = np.empty((6, 7))
    for i in range(7):
        # J_x = joint_axis x( point_t - joint_t)
        J[:3, i] = np.cross(T_w2links[1 + i][:3, 2], point_t - T_w2links[1 + i][:3, 3])
        # J_w = joint_axis
        J[3:, i] = T_w2links[1 + i][:3, 2]
    return T_w2links, J


q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0])
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04])

if __name__ == "__main__":
    joint_state = np.random.uniform(q_min, q_max)
    print(joint_state)
    joint_state = np.array(
        [
            0.9010268205475738,
            -1.4280964528458524,
            -0.8156619494973723,
            -2.066372492711048,
            -0.6399130629921466,
            0.9087261826439509,
            -2.7760439257271434,
            0.018275886901312603,
        ]
    )
    print(list(joint_state))
    P = franka_dh_fk(joint_state, r, d, alpha)
    print(P[-1])
    pass
