import copy
from typing import Dict

import torch
from torch import vmap
from torch.autograd.functional import vjp
from torch.func import functional_call, stack_module_state

from neurg.kinetics.cu_kin import get_fk_model
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.torch_robot import cal_J_all_local_wrt_q
from neurg.my_utils.torch_tf import b_inv_mat34, b_pair_tf_pt
from neurg.my_utils.util_file import load_pickle
from neurg.net.lsdf_model import LinkSdfModel
from neurg.net.multi_linear import MultiMlp
from neurg.utils.scol_util import cal_l1_to_pts_by_wp, generate_non_adj_pairs
from neurg.utils_sdf import b_tf_pts_ch_pt


class NSDFLModel(LinkSdfModel):
    """ NeuG Model for a single link (PyTorch version)"""
    def __init__(self, robot_cpkt_path: str = None, link_name: str = None, device: str = "cuda"):
        super().__init__()
        self.robot_cpkt_path = robot_cpkt_path
        if robot_cpkt_path is not None:
            self.load_weights(robot_cpkt_path, link_name, device=device)

    def l_dist(self, x, ret_grad=False):
        """Local Distance Query ($DQ_l$): query the distances from obstacles to a single link in the link-local frame

        Args:
            x: (N, 3) points
            ret_grad: whether to return gradient
        Returns:
            y: (N, 1) sdf values
            grad: (N, 3) gradients, only if ret_grad is True
        """
        if ret_grad:
            y, grad = self.forward_backward(x)
            return y, grad
        else:
            y = self.forward(x)
            return y
        
class NSDFSCModel:
    def __init__(self, neurg_model: "NSDFModel", n_sample=1024, surface_normal_dir=PATH_ROOT + "/logs/nsdf/dataset/surface_normal"):
        self.rnet = neurg_model
        self.prepare(n_sample=n_sample, surface_normal_dir=surface_normal_dir)

    def prepare(self, n_sample, surface_normal_dir):
        rnet = self.rnet
        device = rnet.device
        mesh_names = rnet.link_names

        d_normal = load_pickle(surface_normal_dir + f"/{n_sample}.pkl")
        l_link_pcd = [
            torch.tensor(d_normal[mesh_name]["pts"]) for mesh_name in mesh_names
        ]
        l_link_pcd = torch.stack(l_link_pcd, dim=0).to(device, dtype=torch.float32)

        l_pairs = generate_non_adj_pairs(len(mesh_names))
        t_pairs = torch.tensor(l_pairs, device=device, dtype=torch.int32)

        self.l_link_pcd = l_link_pcd
        self.n_pairs = t_pairs.shape[0]
        self.t_pairs = t_pairs
        self.l_l1idx = self.t_pairs[:, 0].tolist()

        # points in link2 frame (n_pairs, n_pt, 3)
        self.l_l2_to_pts = (self.l_link_pcd[self.t_pairs[:, 1]].to(dtype=torch.float32).contiguous())

        # setup batch model
        models = []
        for i in range(len(l_pairs)):
            pair0 = l_pairs[i][0]
            mesh_name = mesh_names[pair0]
            models.append(rnet.d_neug_model[mesh_name])
        models = [model.to(device) for model in models]

        params, buffers = stack_module_state(models)
        base_model = copy.deepcopy(models[0])
        base_model = base_model.to("meta")

        # self._sc_models = models
        self._params = params
        self._buffers = buffers
        self._base_model = base_model

        # update batch size (adjust according to GPU memory)
        hid_dim = self.rnet.hid_dim
        if self.n_pairs <= 21:
            if hid_dim == 64:
                self.batch_size = 1000
            else:
                self.batch_size = 2000
        else:
            if hid_dim == 64:
                self.batch_size = 768
            else:
                self.batch_size = 1024

    def forward(self, qs):
        """Self-collision distance query

        Args:
            qs: (n_q, n_dof) joint configurations
        Returns:
            dists: (n_q, n_pairs) self-collision distances for all link pairs
        """

        def fmodel(params, buffers, x):
            return functional_call(self._base_model, (params, buffers), (x,))

        T_w2links = self.rnet.kin.fk_T_w2links(qs)
        l_l1_to_pts = cal_l1_to_pts_by_wp(T_w2links, self.t_pairs, self.l_l2_to_pts)

        preds = vmap(fmodel, in_dims=(0, 0, 1), out_dims=1)(
            self._params, self._buffers, l_l1_to_pts
        )
        preds = preds.squeeze(-1)  # (n_q, n_pairs, n_pt)
        min_d = torch.min(preds, dim=-1).values
        return min_d


class NSDFModel(torch.nn.Module):
    """ NeuRG Model for the whole robot (PyTorch version) """
    def __init__(self, robot_ckpt_path: str, ee_link: str = "panda_hand_tcp", **kwargs):
        super().__init__()
        self.device = (
            kwargs.get("device") or torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.dtype = torch.float

        self.d_neug_model: Dict[str, NSDFLModel] = {}
        self.robot_ckpt_path = robot_ckpt_path
        self.load_weights(robot_ckpt_path)
        self.kin = get_fk_model(ee_link=ee_link)
        self.prepare_eval(device=self.device)
        self.sc_model = NSDFSCModel(self, n_sample=1024)

    def hyperparams(self):
        return {
            "hid_dim": self.hid_dim,
            "hid_size": self.hid_size,
            "actv_type": self.actv_type,
        }

    def get_method_w_param(self):
        hyperparams = self.hyperparams()
        actv_type = hyperparams["actv_type"]

        if actv_type == "softplus":
            actv_type = "soft"
        return f"nsdf_{hyperparams['hid_dim']:03d}x{hyperparams['hid_size']}_{actv_type}"

    def load_weights(self, path, device="cuda"):
        link_infos = torch.load(path)
        for link_name in link_infos.keys():
            for k in link_infos[link_name].keys():
                if isinstance(link_infos[link_name][k], torch.Tensor):
                    link_infos[link_name][k] = link_infos[link_name][k].to(device)

        d_neug_model: Dict[str, NSDFLModel] = {}
        for link_name in link_infos.keys():
            neug_model = NSDFLModel()
            neug_model.load_from_link_info(link_infos[link_name])
            neug_model.eval()
            neug_model.requires_grad_(False)
            neug_model.to(device)
            d_neug_model[link_name] = neug_model
        self.d_neug_model = d_neug_model

    def _extract_mlp_hyperparams(self, mlp_model):
        l_sizes = [mlp_model[0].in_features]
        for i in range(len(mlp_model)):
            if isinstance(mlp_model[i], torch.nn.Linear):
                l_sizes.append(mlp_model[i].out_features)
        hid_dim = mlp_model[0].out_features
        hid_size = len(mlp_model) // 2 - 1
        actv_type = "relu" if isinstance(mlp_model[1], torch.nn.ReLU) else "softplus"
        return {"hid_dim": hid_dim, "hid_size": hid_size, "actv_type": actv_type}

    def _set_link_names(self, names):
        # fmt: off
        bbox_centers = torch.stack([self.d_neug_model[lname].bbox_center for lname in names])  # (n_link, 3)
        bbox_centers = bbox_centers.to(dtype=torch.float32, device=self.device).unsqueeze(1)
        self.register_buffer("bbox_centers", bbox_centers)  # (ch, 1, dim_in) (9, 1, 3)

        bboxs = torch.stack([self.d_neug_model[lname].bbox_largest for lname in names])  # (n_link, 2, 3)
        self.register_buffer("bboxs", bboxs)

        bbox_no_base = bboxs[1:].clone()
        self.register_buffer("bbox_no_base", bbox_no_base)
        # fmt: on

        d_mlp_models = {}
        l_mlps = []
        for k in names:
            mlp = self.d_neug_model[k].mlp
            mlp.requires_grad_(False)
            mlp.eval()
            mlp.to(self.device)
            l_mlps.append(mlp)
            d_mlp_models[k] = mlp
        self.l_mlps = l_mlps
        self.d_mlp_models = d_mlp_models

        hyper_params = self._extract_mlp_hyperparams(l_mlps[0])
        hid_dim, hid_size, actv_type = (
            hyper_params["hid_dim"],
            hyper_params["hid_size"],
            hyper_params["actv_type"],
        )

        # setup MultiMlp model
        mlp_model = l_mlps[0]
        n_ch = len(l_mlps)
        mmlp = MultiMlp(
            [3] + [hid_dim] * hid_size + [1],
            n_ch,
            activation=mlp_model[1],
        )
        mmlp.reset_from_mlps(l_mlps)
        mmlp.eval()
        mmlp.requires_grad_(False)
        mmlp.to(self.device)
        self.mmlp = mmlp

        self.link_names = names
        self.hid_dim = hid_dim
        self.hid_size = hid_size
        self.actv_type = actv_type

    def prepare_eval(self, mesh_names=None, device="cuda"):
        # fmt: off
        if mesh_names is None:
            mesh_names = [
                "link0", 
                "link1", "link2", "link3", 
                "link4", "link5", "link6", "link7", 
                "hand"
            ]
        # fmt: on
        self.device = device
        self._set_link_names(mesh_names)
        self.eval()
        self.requires_grad_(False)
        self.to(device)

    # Basic Internal Functions ------------------------------------------------
    def _norm_x(self, x):
        return x - self.bbox_centers

    def _unnorm_x(self, x):
        return x + self.bbox_centers

    def forward(self, pts_local: torch.Tensor):
        """
        Args:
            pts_local (L, N, 3): pts in local;
            dists: (L, N)
        """
        return self.mmlp(self._norm_x(pts_local)).squeeze(-1)

    def _fwd_ldist(self, pts_local: torch.Tensor):
        """
        Args:
            pts_local: (L, N, 3) points in local link frame
        Returns:
            dists: (L, N) sdf values
        """
        dists = self.forward(pts_local)
        return dists

    # Main Exported APIs ------------------------------------------------------
    def get_link_net(self, link_name: str) -> NSDFLModel:
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
        if ret_grad:
            y, dy_dx = vjp(
                self._fwd_ldist, x, torch.ones((x.shape[:2]), device=x.device)
            )
            return y, dy_dx
        else:
            y = self._fwd_ldist(x)
            return y

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
        if ret_grad:
            T_links2w = b_inv_mat34(T_w2links)  # (L, 3, 4)
            b_pts_local = b_tf_pts_ch_pt(T_links2w, pts)  # (L, N, 3)
            dists, dists_g = self.l_dist(b_pts_local, ret_grad=True)

            T_w2joints = T_w2links[1:8, :3, :]

            # J_pt_local_wrt_q (L, N, 3, 7)
            J = cal_J_all_local_wrt_q(T_links2w[1:, :3, :3], T_w2joints[:, :, 2], pts, T_w2joints[:, :, 3])

            # J_dists_wrt_q
            J = (dists_g[1:].unsqueeze(-2) @ J).squeeze(-2)
            J = torch.concatenate([torch.zeros_like(J[[0]]), J])
            return dists, J
        else:
            T_links2w = b_inv_mat34(T_w2links)  # (L, 3, 4)
            b_pts_local = b_tf_pts_ch_pt(T_links2w, pts)  # (L, N, 3)
            dists = self.l_dist(b_pts_local, ret_grad=False)
            return dists

    def sc_dist(self, q):
        """Self-collision Query ($SC$):
        Given a joint configuration, calculate the self-collision distances, i.e., the distances between non-adjacent link pairs.

        Args:
            q: (7,) joint configuration
        Returns:
            dists: (N_pair,) self-collision distances
        """
        return self.sc_model.forward(q.unsqueeze(0)).squeeze(0)

    # Evaluation APIs ----------------------------------------------------
    def dists_q_pt_pair(self, x):
        """To evaluate distance error on JSDF dataset

        Args:
            x: (batch, 10) , first 7 dim are q, last 3 dim are pts
        """
        qs = x[:, :7]  # (batch, 7)
        pts = x[:, 7:]  # (batch, 3)
        T_w2links = self.kin.fk_link_T44(qs).detach()[:, :9, :3].contiguous()
        if self.dtype == torch.float16:
            pts = pts.half()
            T_w2links = T_w2links.half()
        T_links2w = b_inv_mat34(T_w2links)  # (N, L, 3, 4)
        b_pts_links = b_pair_tf_pt(T_links2w, pts)
        pts_net = b_pts_links.transpose(0, 1)
        dists = self.l_dist(pts_net).T  # (B, N, L)
        return dists


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("test NeuGModel" + '-'*40)
    neug = NSDFLModel("logs/link_infos/softplus_coll_32x3.pth", "hand")
    pts = torch.randn(10, 3).to(device)
    sdf = neug.l_dist(pts)
    sdf, grad = neug.l_dist(pts, ret_grad=True)
    print("Test NeuGModel done!\n")

    print("Test NeuRGModel" + '-'*40)
    neurg_model = NSDFModel("logs/link_infos/softplus_coll_32x3.pth")
    print(neurg_model.get_method_w_param())

    print("Test Local Distance Query")
    pts_local = torch.randn(len(neurg_model.link_names), 4, 3).to(device)
    dists = neurg_model.l_dist(pts_local)
    dists, dists_g = neurg_model.l_dist(pts_local, ret_grad=True)
    print("dists.shape:", dists.shape, "dists_g.shape:", dists_g.shape)
    print("dists:", dists.cpu())

    print("Test Global Distance Query")
    q = torch.randn(7).to(device)
    T_w2links = neurg_model.kin.qp_fk_T44s(q).detach()[:9, :3, :].contiguous()
    pts = torch.randn(4, 3).to(device)
    dists = neurg_model.g_dist(pts, T_w2links)
    dists, dists_g = neurg_model.g_dist(pts, T_w2links, ret_grad=True)
    print("dists.shape:", dists.shape, "dists_g.shape:", dists_g.shape)
    print("dists:", dists.cpu())
    print("Test NeuRGModel done!\n")

    print("Test self-collision" + '-'*40)
    q = torch.tensor([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]).to(device)
    dists = neurg_model.sc_dist(q)
    print("dists.shape:", dists.shape)
    print("dists:", dists.cpu())
    """[
    0.1374, 0.3128, 0.4309, 0.5028, 0.5202, 0.4975, 0.4492, 0.1393, 0.2163,
    0.2553, 0.3207, 0.3218, 0.2916, 0.0697, 0.1463, 0.3112, 0.3223, 0.2942,
    0.0739, 0.2897, 0.3698, 0.3805, 0.2122, 0.3023, 0.3213, 0.0226, 0.0670,
    0.0303]
    """
    print("Test self-collision done!\n")
