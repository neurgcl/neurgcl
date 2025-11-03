from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as sR
from . import modern_robotics as mr

# 按照内旋方式，Z-Y-X旋转顺序（指先绕自身轴Z，再绕自身轴Y，最后绕自身轴X），可得旋转矩阵（内旋是右乘） R.from_euler('ZYX', [1,2,3]).as_matrix()
# 按照外旋方式，X-Y-Z旋转顺序（指先绕固定轴X，再绕固定轴Y，最后绕固定轴Z），可得旋转矩阵（外旋是左乘） R.from_euler('xyz', [1,2,3]).as_matrix()


def R2rv(R):
    return sR.from_matrix(R).as_rotvec()


def rv2R(rotvec):
    return sR.from_rotvec(np.array(rotvec).squeeze()).as_matrix()


def R2euler(R, degrees=False, seq="xyz"):
    return sR.from_matrix(R).as_euler(seq, degrees=degrees)


def euler2R(euler, degrees=False, seq="xyz"):
    return sR.from_euler(seq, euler, degrees=degrees).as_matrix()


def T2tw(T):
    # T ->> t, rotvec
    return np.r_[np.array(T[:3, 3]).squeeze(), R2rv(T[:3, :3])]


def T2tR(T):
    return T[:3, 3], T[:3, :3]


def tR2T(t, R):
    return mr.RpToTrans(R, t)


def T2xyzrpy(T, degrees=True):
    return np.r_[T[:3, 3].copy(), R2euler(T[:3, :3], degrees=degrees)]


def xyzrpy2T(xyzrpy, degrees=True):
    t = xyzrpy[:3]
    R = euler2R(xyzrpy[3:], degrees=degrees)
    return tR2T(t, R)


def xyzrpydeg2T(xyzrpy):
    t = xyzrpy[:3]
    R = euler2R(xyzrpy[3:], degrees=True)
    return tR2T(t, R)


def T2xyzquat(T):
    """T -> xyz, quat(xyzw)"""
    return T[:3, 3], sR.from_matrix(T[:3, :3]).as_quat()


def xyzquat2T(t, quat):
    """xyz, quat(xyzw) -> T"""
    R = sR.from_quat(quat).as_matrix()
    return tR2T(t, R)

def T2xyzxyzw(T):
    """T -> xyz, quat(xyzw)"""
    return np.r_[T[:3, 3], sR.from_matrix(T[:3, :3]).as_quat()]

def T2xyzqwxyz(T):
    """T -> xyz, quat(wxyz)"""
    xyzw = sR.from_matrix(T[:3, :3]).as_quat()
    return T[:3, 3], np.r_[xyzw[3], xyzw[0:3]]


def xyzqwxyz2T(t, quat):
    """xyz, quat(wxyz) -> T"""
    R = sR.from_quat(np.r_[quat[1:4], quat[0]]).as_matrix()
    return tR2T(t, R)


def mat34Inv(T):
    assert (T.shape[1] == 3 or T.shape[1] == 4) and T.shape[2] == 4
    R = T[:, :3, :3]
    p = T[:, :3, [3]]
    Rt = R.transpose(0, 2, 1)
    return np.c_[Rt, -np.matmul(Rt, p)]


def mat34dot(m1, m2):
    t = np.matmul(m1[..., :3], m2[..., [3]]) + m1[..., [3]]
    R = np.matmul(m1[..., :3], m2[..., :3])
    return np.c_[R, t]


def mat34dotpt(m1, pts):
    """
    pts: (3, n)
    """
    return m1[..., :3].dot(pts) + m1[..., [3]]


def mat44Inv(T):
    assert (T.shape[1] == 4 or T.shape[1] == 4) and T.shape[2] == 4
    R = T[:, :3, :3]
    p = T[:, :3, [3]]
    Rt = R.transpose(0, 2, 1)
    mat34 = np.c_[Rt, -np.matmul(Rt, p)]
    return np.concatenate(
        [mat34, np.array([[[0, 0, 0, 1]]]).repeat(T.shape[0], axis=0)], axis=1
    )


def transform_pts(batch_T, batch_pts):
    assert len(batch_T.shape) == 3 and batch_T.shape[1] == 4 and batch_T.shape[2] == 4
    n_link = len(batch_T)
    b_pts_fk = [None] * n_link
    for i in range(n_link):
        R = batch_T[i][:3, :3]
        T = batch_T[i][:3, 3]
        b_pts_fk[i] = np.dot(batch_pts, R.T) + T.T
    b_pts_fk = np.array(b_pts_fk)
    return b_pts_fk


def transform_meshs(batch_T, link_meshs):
    assert len(batch_T.shape) == 3 and batch_T.shape[1] == 4 and batch_T.shape[2] == 4
    n_link = len(batch_T)
    link_meshs_fk = deepcopy(link_meshs)
    for i in range(n_link):
        R = batch_T[i][:3, :3]
        T = batch_T[i][:3, 3]
        link_meshs_fk[i].vertices = np.dot(link_meshs[i].vertices, R.T) + T.T
    return link_meshs_fk


def tf_pts(T, pts):
    """
     R @ pts + t 

     T:     4, 4
     pts:   batch,3
    ---
     return: batch,3
    """
    return ((T[:3, :3] @ np.expand_dims(pts, axis=-1)) + T[:3, [3]]).squeeze(-1)