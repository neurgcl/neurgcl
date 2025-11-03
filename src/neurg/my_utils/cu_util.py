import torch
from curobo.geom import transform as cutf
from curobo.geom.transform import BatchTransformPoint
import warp as wp

# @wp.kernel
# def compute_quat_to_matrix(
#     quat: wp.array(dtype=wp.vec4),
#     out_mat: wp.array(dtype=wp.mat33),
# ):
#     # b pose_1 and b pose_2, compute pose_1 * pose_2
#     b_idx = wp.tid()
#     # read data:

#     in_quat = quat[b_idx]

#     # read point
#     # create a transform from a vector/quaternion:
#     q_1 = wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0])
#     m_1 = wp.quat_to_matrix(q_1)


#     # write pt:
#     out_mat[b_idx] = m_1


@wp.kernel
def compute_tf_Bpose_Npt(
    position: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    pt: wp.array(dtype=wp.vec3),
    n_pts: wp.int32,
    n_poses: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):  # given n,3 points and b poses, get b,n,3 transformed points
    # we tile as
    b_idx, p_idx = wp.tid()

    # read data:
    in_position = position[b_idx]
    in_quat = quat[b_idx]
    in_pt = pt[p_idx]

    # read point
    # create a transform from a vector/quaternion:
    t = wp.transform(in_position, wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0]))

    # transform a point
    p = wp.transform_point(t, in_pt)

    # write pt:
    out_pt[b_idx * n_pts + p_idx] = p


def tf_Bposquat_Npt(position, quaternion, points, out_points=None):
    # given P,3 points and B poses, get B,P,3 transformed points
    position = position.view(-1, 3)
    quaternion = quaternion.view(-1, 4)

    B, _ = position.shape
    N, _ = points.shape

    if out_points is None:
        out_points = torch.zeros((B, N, 3), device=points.device, dtype=points.dtype)

    wp.launch(
        kernel=compute_tf_Bpose_Npt,
        dim=(B, N),
        inputs=[
            wp.from_torch(position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
            wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            wp.from_torch(points.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
            N,
            B,
        ],
        outputs=[wp.from_torch(out_points.view(-1, 3).contiguous(), dtype=wp.vec3)],
        stream=wp.stream_from_torch(position.device),
    )
    return out_points


def inv_tf_Bposquat_Npt(
    link_pos, link_quat, points, out_pos_inv=None, out_quat_inv=None, out_points=None
) -> torch.Tensor:
    """
    Args:
        link_pos: [B, 3]
        link_quat: [B, 4]
        points: [N, 3]
    Returns:
        out_points: [B, N, 3]
    """
    out_pos_inv, out_quat_inv = cutf.pose_inverse(link_pos, link_quat, out_pos_inv, out_quat_inv)
    out_points = tf_Bposquat_Npt(out_pos_inv, out_quat_inv, points, out_points)
    return out_points


if __name__ == '__main__':
    pass
