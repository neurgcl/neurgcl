from typing import Dict

import torch

from neurg.kinetics.cu_kin import get_fk_model


class NeuGModel:
    def __init__(self, robot_cpkt_path: str = None, link_name: str = None):
        raise NotImplementedError(
            "The CUDA-accelerated NeuRG method is not open-source yet and will be made open-source after the paper is accepted."
        )

    def l_dist(self, x, ret_grad=False):
        """Local Distance Query ($DQ_l$): query the distances from obstacles to a single link in the link-local frame

        Args:
            x: (N, 3) points
            ret_grad: whether to return gradient
        Returns:
            y: (N, 1) sdf values
            grad: (N, 3) gradients, only if ret_grad is True
        """
        pass


class NeuRGModel:
    def __init__(self, robot_ckpt_path: str, ee_link: str = "panda_hand_tcp", **kwargs):
        raise NotImplementedError(
            "The CUDA-accelerated NeuRG method is not open-source yet and will be made open-source after the paper is accepted."
        )
        self.device = kwargs.get("device") or torch.device("cuda")
        self.dtype = torch.float

        self.d_neug_model: Dict[str, NeuGModel] = {}
        self.robot_ckpt_path = robot_ckpt_path
        self.kin = get_fk_model(ee_link=ee_link)

    def hyperparams(self):
        return {
            "hid_dim": 32,
            "hid_size": 3,
            "actv_type": "softplus",
        }

    def get_method_w_param(self):
        hyperparams = self.hyperparams()
        actv_type = hyperparams["actv_type"]

        if actv_type == "softplus":
            actv_type = "soft"
        return (
            f"neurg_{hyperparams['hid_dim']:03d}x{hyperparams['hid_size']}_{actv_type}"
        )

    # Main Exported APIs ------------------------------------------------------
    def get_link_net(self, link_name: str) -> NeuGModel:
        return self.d_neug_model[link_name]

    def l_dist(self, x, ret_grad=False):
        """Local Distance Query ($DQ_l$):
        Parallelly query the distances from points to each individual link in the link-local frame.

        Args:
            x: (L, N, 3) points in local link frame
            ret_grad: whether to return gradient
        Returns:
            y: (L, N) sdf values
            dy_dx: (L, N, 3) sdf gradients w.r.t. local points, only if ret_grad is True
        """
        pass

    def g_dist(self, pts, T_w2links, ret_grad=False):
        """Global Distance Query ($DQ_g$):
        Computes the distances from spatial points to the robot arm in a specific configuration.

        Args:
            pts: (N, 3) points in world frame
            T_w2links: (L, 4, 4) transformation matrix from world to link
            ret_grad: whether to return gradient
        Returns:
            y: (N, 1) sdf values
            dy_dq: (N, 3) sdf gradients w.r.t. joint configuration, only if ret_grad is True
        """
        pass

    def sc_dist(self, q):
        """Self-collision Query ($SC$):
        Given a joint configuration, calculate the self-collision distances, i.e., the distances between non-adjacent link pairs.

        Args:
            q: (7,) joint configuration
        Returns:
            dists: (N_pair,) self-collision distances
        """
        pass
