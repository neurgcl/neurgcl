# implementation is based on https://github.com/NVlabs/storm

import torch
from torch import nn
from torch.nn import ReLU

# from .network_macros_mod import MLPRegression, scale_to_base, scale_to_net
from jsdf.sdf.network_macros_mod import *

# from .util_file import *
from functorch.compile import aot_function

from functools import partial

if torch.__version__ < "2.0":
    from functorch import jacrev, vjp, vmap
    from functorch import make_functional_with_buffers
else:
    from torch.func import jacrev, vjp, vmap


class JSDFNet:
    """This class loads a network to predict the signed distance given a robot joint config."""

    def __init__(self, in_channels, out_channels, skips, layers):

        super().__init__()
        act_fn = ReLU
        in_channels = in_channels
        self.out_channels = out_channels
        dropout_ratio = 0
        mlp_layers = layers
        self.model = MLPRegression(in_channels, self.out_channels, mlp_layers, skips, act_fn=act_fn, nerf=True)
        self.m = torch.zeros((500, 1)).to('cuda:0')
        self.m[:, 0] = 1
        self.order = list(range(out_channels))
        self.hid_dim = layers[1]
        self.hid_sz = len(layers) - 1

    def get_method_w_param(self):
        return f'jsdf_{self.hid_dim}x{self.hid_sz}'

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def cal_dis(self, x):
        return self.model.forward(x)

    def dists_q_pt_pair(self, x):
        return self.model.forward(x) * 0.01

    def set_link_order(self, order):
        self.order = order

    def load_weights(self, f_name, tensor_args):
        """Loads pretrained network weights if available.

        Args:
            f_name (str): file name, this is relative to weights folder in this repo.
            tensor_args (Dict): device and dtype for pytorch tensors
        """
        try:
            chk = torch.load(f_name)
            self.model.load_state_dict(chk["model_state_dict"])
            self.norm_dict = chk["norm"]
            for k in self.norm_dict.keys():
                self.norm_dict[k]['mean'] = self.norm_dict[k]['mean'].to(**tensor_args)
                self.norm_dict[k]['std'] = self.norm_dict[k]['std'].to(**tensor_args)
            print('Weights loaded!')
        except Exception as E:
            print('WARNING: Weights not loaded')
            print(E)
        self.model = self.model.to(**tensor_args)
        self.tensor_args = tensor_args
        self.model.eval()

    def dists_qxpt(self, qs, pts):
        """
        qs: (B, 7)
        pts: (N, 3)

        Returns:
            dists: (B, N, n_link)
        """

        B = qs.shape[0]
        N = pts.shape[0]
        # x = [qs, pts] (B, N, 10)
        x = torch.cat([qs[:, None].repeat(1, N, 1), pts[None, :].repeat(B, 1, 1)], dim=-1)
        x = x.view(-1, 10)
        return (self.model.forward(x) * 0.01).view(B, N, -1)

    def pts2robot_dists_g(self, q, pts):
        """
        Used for qp

        q: n_joint
        pts: n_pt,3

        return:
            dists: n_link, n_pt
            grads: n_link, n_pt, n_joint
        """
        net_x = torch.concat([q.unsqueeze(0).repeat([pts.shape[0], 1]), pts], axis=1)
        dists, vjp_fn = vjp(self.model.forward, net_x)  # batch, n_link
        device = dists.device

        def cal_min_n_g(vjp_fn, dists, idx):
            grad_v = torch.zeros_like(dists, requires_grad=False)
            grad_v = torch.scatter(grad_v, 1, idx.unsqueeze(-1).to(device), 1)
            return vjp_fn(grad_v)[0]

        # [[0, 1, 2, 3, 4, 5, 6, 7, 8],
        #  [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        idxs = torch.arange(dists.shape[-1]).unsqueeze(0).expand(pts.shape[0], -1).T

        grads = vmap(partial(cal_min_n_g, vjp_fn, dists))(idxs).detach()[..., :7]
        dists = dists.detach().T

        scale = 0.01
        return dists * scale, grads * scale

    def mpc_dists(self, qs: torch.Tensor, col_pts: torch.Tensor):
        """
        Args:
            qs: (B, H, 7)
            pts: (P, 3)

            first change the shape of link_pos and link_quat to (LxBxH, 3) and (LxBxH, 4)
        Returns:
            dists: (L, B, H, P)
        """
        B, H, _ = qs.shape
        P, _ = col_pts.shape
        dists = self.dists_qxpt(qs.view(-1, 7), col_pts)  # (B*H, P, L)
        dists = dists.view(B, H, P, -1)[:, :, :, 1:]
        return dists
