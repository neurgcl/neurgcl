import torch
import warp as wp


def generate_non_adj_pairs(n_link):
    """Generate non-adjacent link index pairs."""
    l_pairs = []
    for i in range(n_link):
        for j in range(i + 2, n_link):
            l_pairs.append((i, j))
    return l_pairs


def cal_Tl1l2s(T_w2links, t_pairs):
    """Calculate link1 to link2 transformation matrices for all link pairs

    Args:
        T_w2links: (n_q, n_link, 4, 4) 
    Returns:
        l_T_l1l2: (n_q, n_pairs, 4, 4)
    """
    l_T_w2l1 = T_w2links[:, t_pairs[:, 0]]
    l_T_w2l2 = T_w2links[:, t_pairs[:, 1]]
    l_T_l1l2 = torch.linalg.inv(l_T_w2l1) @ l_T_w2l2  # n_q, n_pairs, 4, 4
    return l_T_l1l2


def q_pair_tf_pts(l_T_l1l2: torch.Tensor, l_l2_to_pts: torch.Tensor)-> torch.Tensor:
    """R @ pts + t

    Args:
        l_T_l1l2: (n_q, n_pairs, 4, 4)
        l_l2_to_pts: (n_pairs, n_pt, 3)
    Returns:
        l_l1_to_pts: (n_q, n_pairs, n_pt, 3)
    """
    #                           ((n_q, n_pairs, 1, 3, 3) @ (n_pairs, n_pt, 3, 1)).squeeze(-1)
    #                          = (n_q, n_pairs, n_pt, 3)
    # (n_q, n_pairs, n_pt, 3)  = (n_q, n_pairs, n_pt, 3) + (n_q, n_pairs, 1, 3)

    return (l_T_l1l2[:, :, :3, :3].unsqueeze(2) @ l_l2_to_pts.unsqueeze(-1)).squeeze(-1) + l_T_l1l2[:, :, :3, 3].unsqueeze(2)


def cal_l1_to_pts(T_w2links, t_pairs, l_l2_to_pts):
    """Given links' transformations, transform points from the link2 frame to the link1 frame for all link pairs."""
    l_T_w2l1 = T_w2links[:, t_pairs[:, 0]]
    l_T_w2l2 = T_w2links[:, t_pairs[:, 1]]
    l_T_l1l2 = torch.linalg.inv(l_T_w2l1) @ l_T_w2l2  # n_q, n_pairs, 4, 4
    # l_l2_to_pts: n_pairs, n_pt, 3
    l_l1_to_pts = q_pair_tf_pts(l_T_l1l2, l_l2_to_pts)
    return l_l1_to_pts


def cal_l1_to_pts_by_wp(T_w2links, t_pairs, l_l2_to_pts):
    """Given links' transformations, transform points from the link2 frame to the link1 frame for all link pairs.
    
    Args:
        T_w2links: (n_q, n_link, 4, 4)
        t_pairs: (n_pair, 2)
        l_l2_to_pts: (n_pair, n_pt, 3)
    Returns:
        l_l1_to_pts: (n_q, n_pair, n_pt, 3)
    """
    l_T_l1l2 = cal_Tl1l2s_by_wp(T_w2links, t_pairs)   # (n_q, n_pairs, 4, 4)
    l_l1_to_pts = q_pair_tf_pts(l_T_l1l2, l_l2_to_pts)
    return l_l1_to_pts


def init_warp(quiet=True):
    wp.config.quiet = quiet
    # wp.config.print_launches = True
    # wp.config.verbose = True
    # wp.config.mode = "debug"
    # wp.config.verify_cuda = True
    # wp.config.enable_backward = True
    # wp.config.verify_autograd_array_access = True
    # wp.config.cache_kernels = False
    wp.init()

    # wp.force_load(wp.device_from_torch(tensor_args.device))
    return True


@wp.kernel
def wpkernel_compute_T_l1l2(
    T_w2links: wp.array(dtype=wp.mat44),
    t_pairs: wp.array(dtype=wp.vec2i),
    n_q: wp.int32,
    n_link: wp.int32,
    n_pair: wp.int32,
    T_l1l2: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()

    q_idx = tid / n_pair
    pair_idx = tid - (q_idx * n_pair)
    pair = t_pairs[pair_idx]
    tf_stride = q_idx * n_link

    T_l1 = T_w2links[tf_stride + pair[0]]
    T_l2 = T_w2links[tf_stride + pair[1]]
    T_l1l2[tid] = wp.inverse(T_l1) @ T_l2


def cal_Tl1l2s_by_wp(T_w2links, t_pairs, out_mat=None):
    """
    Args:
        T_w2links: (n_q, n_link, 4, 4)
        t_pairs: (n_pair, 2)
    Returns:
        out_mat: (n_q, n_pair, 4, 4)
    """
    init_warp()
    n_q = T_w2links.shape[0]
    n_link = T_w2links.shape[1]
    n_pair_ret = t_pairs.shape[0]
    device = T_w2links.device

    if out_mat is None:
        out_mat = torch.zeros(
            (n_q, n_pair_ret, 4, 4), device=device, dtype=torch.float32
        )

    wp.launch(
        kernel=wpkernel_compute_T_l1l2,
        dim=(n_q * n_pair_ret),
        inputs=[
            wp.from_torch(
                T_w2links.detach().view(-1, 4, 4).contiguous(), dtype=wp.mat44
            ),
            wp.from_torch(t_pairs.detach().view(-1, 2).contiguous(), dtype=wp.vec2i),
            n_q,
            n_link,
            n_pair_ret,
        ],
        outputs=[
            wp.from_torch(out_mat.detach().view(-1, 4, 4), dtype=wp.mat44),
        ],
        stream=wp.stream_from_torch(device),
    )
    return out_mat
