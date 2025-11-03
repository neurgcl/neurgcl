import argparse
import math
import os
from typing import Dict

import numpy as np
import torch
from rdf.panda_layer.panda_layer import PandaLayer
from rdf.rdf_ds import RdfDataSet
from rdf.rdf_utils import transform_points
from torch.utils.data import Dataset
from tqdm import tqdm

from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.torch_tf import b_inv_mat44, b_pair_tf_pt
from neurg.my_utils.util_file import load_pickle

np.set_printoptions(threshold=np.inf)
OUR_DIR = PATH_ROOT + '/logs/rdf'


class RdfLinkDataset(Dataset):
    def __init__(
        self,
        data_dir=os.path.join(PATH_ROOT, "data/dataset/splited/collision_links_240304_164742"),
        link_name="link0",
        stage="train",
        device="cpu",
        precision=32,
        normalize=True,
    ) -> None:
        super().__init__()

        sub_dir = os.path.join(data_dir, stage)
        data = load_pickle(sub_dir + f"/{link_name}.pkl")

        self.data = data
        self.bbox = data["bbox_largest"]
        self.extents = self.bbox[1] - self.bbox[0]
        self.bbox_center = self.bbox[0] + self.extents / 2.0
        self.scale = float(np.linalg.norm(self.extents / 2.0))

        pts_keys = [
            "pts_inside",
            "pts_outside",
            "pts_on_sur",
            "pts_near_sur",
            # "pts_bbox_pad_fixed",
            "pts_bbox_largest",
        ]
        data_all = []
        for k in pts_keys:
            data_all.append(data[k])
        data_all = np.concatenate(data_all)

        if precision == 16:
            DTYPE = torch.float16
        else:
            DTYPE = torch.float32
        if normalize:
            self.x = torch.tensor(self.norm_x(data_all[:, :3]), device=device, dtype=DTYPE)
        else:
            self.x = torch.tensor(data_all[:, :3], device=device, dtype=DTYPE)
        self.y = torch.tensor(data_all[:, [3]], device=device, dtype=DTYPE)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret = (self.x[idx], self.y[idx])
        return ret

    def __getitems__(self, idx):
        return self.__getitem__(idx)

    def norm_x(self, x):
        return (x - self.bbox_center) / self.scale

    def unnorm_x(self, x):
        return (x * self.scale) + self.bbox_center

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)


class RDFLinkNet:
    def __init__(self, n_func, domain_min, domain_max, device, weights, scale, offset, mesh_name):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        self.weights = weights
        self.scale = scale
        self.offset = offset
        self.mesh_name = mesh_name

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            return (x - self.offset.cpu().numpy()) / self.scale
        return (x - self.offset) / self.scale

    def unnormalize(self, x):
        if isinstance(x, np.ndarray):
            return x * self.scale + self.offset.cpu().numpy()
        return x * self.scale + self.offset

    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self, t, use_derivative=False):
        # t is normalized to [0,1]
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (
                1 - t
            ).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def build_basis_function_from_points(self, p, use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self.build_bernstein_t(p, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func**3)
        if use_derivative == False:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]
            d_phi_x = torch.einsum(
                "ij,ik->ijk", torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func**2), phi_z
            ).view(-1, self.n_func**3)
            d_phi_y = torch.einsum(
                "ij,ik->ijk", torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func**2), phi_z
            ).view(-1, self.n_func**3)
            d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1D).view(-1, self.n_func**3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
            return phi_xyz, d_phi_xyz

    def dist_link2pt(self, pts):
        pts = (pts - self.offset) / self.scale
        phi_p, _ = self.build_basis_function_from_points(pts, use_derivative=False)
        sdfs = torch.matmul(phi_p, self.weights) * self.scale
        return sdfs.unsqueeze(-1)

    def __call__(self, pts):
        return self.dist_link2pt(pts)

    def forward(self, pts):
        """
        Args:
            pts (unit: m): [N, 3]

        Returns:
            sdf (unit:m): [N, 1]
        """
        return self.dist_link2pt(pts)

    def forward_backward(self, pts):
        """
        Args:
            pts (unit: m): [N, 3]
        Returns:
            sdf (unit:m): [N, 1]
            gradient (unit:m): [N, 3]
        """
        pts_normed = (pts - self.offset) / self.scale

        x_bounded = torch.where(pts_normed > 1.0 - 1e-2, 1.0 - 1e-2, pts_normed)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = pts_normed - x_bounded

        phi, dphi = self.build_basis_function_from_points(pts_normed, use_derivative=True)
        phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
        output = torch.einsum('nbk,b->nk', phi_cat, self.weights)
        sdf = output[:, [0]]
        gradient = output[:, 1:]
        sdf = sdf + res_x.norm(dim=-1, keepdim=True)
        sdf = sdf * (self.scale)

        gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
        gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
        return sdf, gradient

    def forward_backward1(self, pts):
        """
        Args:
            pts (unit: m): [N, 3]
        Returns:
            sdf (unit:m): [N, 1]
            gradient (unit:m): [N, 3]
        """
        pts_normed = (pts - self.offset) / self.scale
        phi, dphi = self.build_basis_function_from_points(pts_normed, use_derivative=True)
        phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
        output = torch.einsum('nbk,b->nk', phi_cat, self.weights) * self.scale
        sdf = output[:, [0]]
        gradient = output[:, 1:]
        return sdf, gradient

    def to(self, device):
        self.weights = self.weights.to(device)
        self.offset = self.offset.to(device)


class RDFRobotNet:
    def __init__(self, device='cuda', ckpt_path='logs/rdf/models/coll_BP_8.pt') -> None:
        domain_min, domain_max = -1, 1
        model = torch.load(ckpt_path)
        n_func = round(math.pow(model[0]['weights'].shape[0], 1 / 3))
        self.model = model
        self.n_func = n_func
        self.device = device
        linknets = {}
        for i in range(len(model)):
            mesh_name = model[i]['mesh_name']
            if isinstance(model[i]['offset'], np.ndarray):
                model[i]['offset'] = torch.tensor(model[i]['offset'])

            linknets[mesh_name] = RDFLinkNet(
                n_func,
                domain_min,
                domain_max,
                device,
                model[i]['weights'].to(device),
                model[i]['scale'],
                model[i]['offset'].to(device, dtype=torch.float32),
                mesh_name,
            )
        self.linknets: Dict[str, RDFLinkNet] = linknets
        self.bp_sdf = BPSDF(n_func, domain_min, domain_max, device)

    def get_method_w_param(self):
        method = 'rdf'
        return f"{method}_{self.n_func:02d}"

    def dists_q_pt_pair(self, x):
        B = len(x)
        use_derivative = False
        used_links = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        batch_sz_max = 20000
        if B > batch_sz_max:
            res = []
            for i in range(0, len(x), batch_sz_max):
                x1 = x[i : i + batch_sz_max]
                qs = x1[:, :7]  # (batch, 7)
                pts = x1[:, 7:]  # (batch, 3)
                d = self.bp_sdf.dists_q_pt_pair(pts, qs, self.model, use_derivative, used_links)
                res.append(d)
            res = torch.cat(res, dim=0)
            return res
        qs = x[:, :7]  # (batch, 7)
        pts = x[:, 7:]  # (batch, 3)
        return self.bp_sdf.dists_q_pt_pair(pts, qs, self.model, use_derivative, used_links)

    def dists_qxpt(self, qs, pts, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8], batch_sz_max=20000):
        """
        qs: (B, 7)
        pts: (N, 3)

        Returns:
            dists: (B, N, n_link)
        """

        B = len(pts)
        if B > batch_sz_max:
            res = []
            for i in range(0, len(pts), batch_sz_max):
                pts1 = pts[i : i + batch_sz_max]  # (batch, 3)
                d = self.bp_sdf.cal_qxpt(qs, pts1, self.model)
                res.append(d)
            res = torch.cat(res, dim=0)
            return res
        return self.bp_sdf.cal_qxpt(qs, pts, self.model, used_links=used_links)

    def dists_g_via_diff(self, theta, x, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        """
        calculate sdf and gradient with respect to joints (use finite difference)
        Args:
            theta: (B,7)
            x: (N,3), object points xyz
        Returns:
            dists: [B, n_col_link(-1), n_pt]
            grads: [B, n_col_link(-1), n_pt, 7]
        """
        delta = 0.001
        N = x.shape[0]
        B = theta.shape[0]
        theta = theta.unsqueeze(1)  # (B,1,7)
        d_theta = (
            theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7, 7) * delta
        ).reshape(B, -1, 7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        sdf = self.bp_sdf.cal_qxpt(theta, x, self.model, use_derivative=False, used_links=used_links)
        sdf = sdf.reshape(B, 8, N, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta  # (B,7,N,Link)
        sdf_val = sdf[:, 0, :]  #  B, N, Link
        sdf_val = sdf_val.transpose(1, 2)  # B, Link, N
        grad_val = d_sdf.permute(0, 3, 2, 1)  # (B, Link, N, 7)
        return sdf_val, grad_val


class BPSDF:
    def __init__(self, n_func, domain_min, domain_max, device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        self.robot = PandaLayer(device)

    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self, t, use_derivative=False):
        # t is normalized to [0,1]
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (
                1 - t
            ).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def build_basis_function_from_points(self, p, use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self.build_bernstein_t(p, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func**3)
        if use_derivative == False:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]
            d_phi_x = torch.einsum(
                "ij,ik->ijk", torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func**2), phi_z
            ).view(-1, self.n_func**3)
            d_phi_y = torch.einsum(
                "ij,ik->ijk", torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func**2), phi_z
            ).view(-1, self.n_func**3)
            d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1D).view(-1, self.n_func**3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
            return phi_xyz, d_phi_xyz

    def train_ours_data(self, epoches=200):
        mesh_dict = {}
        l_link_name = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand']
        for i, mesh_name in enumerate(l_link_name):
            mesh_dict[i] = {}
            mesh_dict[i]['mesh_name'] = mesh_name
            # load data
            ds_train = RdfLinkDataset(link_name=l_link_name[i], stage="train", device=self.device)
            print(f"ds_size: {len(ds_train)}")
            dl_train = SliceBatchLoader(ds_train, batch_size=225000 // 8, shuffle=True)

            wb = torch.zeros(self.n_func**3).float().to(self.device)
            B = (torch.eye(self.n_func**3) / 1e-4).float().to(self.device)
            # loss_list = []
            for iter in tqdm(range(epoches)):
                for p, sdf in dl_train:
                    p, sdf = p.to(self.device), sdf.to(self.device).squeeze(-1)
                    phi_xyz, _ = self.build_basis_function_from_points(p, use_derivative=False)

                    K = torch.matmul(B, phi_xyz.T).matmul(
                        torch.linalg.inv(
                            (
                                torch.eye(len(p)).float().to(self.device)
                                + torch.matmul(torch.matmul(phi_xyz, B), phi_xyz.T)
                            )
                        )
                    )
                    B -= torch.matmul(K, phi_xyz).matmul(B)
                    err = sdf - torch.matmul(phi_xyz, wb)
                    print(f"err_mean: {err.abs().mean()}")
                    delta_wb = torch.matmul(K, err.squeeze())

                    wb += delta_wb

            print(f'mesh name {mesh_name} finished!')
            mesh_dict[i] = {
                'mesh_name': mesh_name,
                'weights': wb,
                'offset': ds_train.bbox_center,
                'scale': float(ds_train.scale),
            }

        out_path = f'{OUR_DIR}/models/coll_BP_{self.n_func}.pt'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(mesh_dict, out_path)  # save the robot sdf model
        print(f'model saved to {out_path}')

    def get_whole_body_sdf_batch(
        self, x, pose, theta, model, use_derivative=True, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ):
        B = len(theta)
        N = len(x)
        K = len(used_links)
        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()
        trans_list = self.robot.get_transformations_each_link(pose, theta)

        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(
            -1, 4, 4
        )  # B,K,4,4
        x_robot_frame_batch = transform_points(
            x.float(), torch.linalg.inv(fk_trans).float(), device=self.device
        )  # B*K,N,3

        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)  # B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=False)
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)  # K,B*N,-1
            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
            # sdf
            sdf = (
                torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B, N).transpose(0, 1).reshape(B * K, N)
            )  # B,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            return sdf_value, None
        else:
            phi, dphi = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)  # K,B*N,-1,4

            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)

            output = (
                torch.einsum('ijkl,ik->ijl', phi_cat, weights_near)
                .reshape(K, B, N, 4)
                .transpose(0, 1)
                .reshape(B * K, N, 4)
            )
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * (scale.reshape(B, K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = (
                torch.einsum('ijk,ikl->ijl', fk_rotation, gradient.transpose(1, 2)).transpose(1, 2).reshape(B, K, N, 3)
            )
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx)[:, 0, :, :]
            # gradient_value = None
            return sdf_value, gradient_value

    def dists_q_pt_pair(self, pts, q, model, use_derivative=False, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        B = len(q)
        N = len(pts)
        K = len(used_links)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(self.device)
        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()
        trans_list = self.robot.get_transformations_each_link(pose, q)

        # fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(
        #     -1, 4, 4
        # )  # B*K,4,4
        # x_robot_frame_batch = transform_points(
        #     pts.float(), torch.linalg.inv(fk_trans).float(), device=self.device
        # )  # B*K,3
        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].contiguous()
        T_links2w = b_inv_mat44(fk_trans)  # (n_pt, n_link, 3, 4)
        x_robot_frame_batch = b_pair_tf_pt(T_links2w, pts).reshape(-1, 3)

        x_robot_frame_batch_scaled = x_robot_frame_batch - offset
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1)  # B*K,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self.build_basis_function_from_points(x_bounded.reshape(B * K, 3), use_derivative=False)
            phi = phi.reshape(B, K, -1).transpose(0, 1).reshape(K, B, -1)  # K,B,-1
            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B).transpose(0, 1).reshape(B * K)  # B,K
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K)
            sdf = sdf * scale.reshape(B, K)
            # sdf_value, idx = sdf.min(dim=1)
            # return sdf_value, None
            return sdf
        else:
            phi, dphi = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)  # K,B*N,-1,4

            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)

            output = (
                torch.einsum('ijkl,ik->ijl', phi_cat, weights_near)
                .reshape(K, B, N, 4)
                .transpose(0, 1)
                .reshape(B * K, N, 4)
            )
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * (scale.reshape(B, K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = (
                torch.einsum('ijk,ikl->ijl', fk_rotation, gradient.transpose(1, 2)).transpose(1, 2).reshape(B, K, N, 3)
            )
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx)[:, 0, :, :]
            # gradient_value = None
            # return sdf, gradient_value
            return sdf

    def cal_qxpt(self, qs, pts, model, use_derivative=False, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        B = len(qs)
        N = len(pts)
        K = len(used_links)

        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(self.device)

        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()
        trans_list = self.robot.get_transformations_each_link(pose, qs)

        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(
            -1, 4, 4
        )  # B,K,4,4
        x_robot_frame_batch = transform_points(
            pts.float(), torch.linalg.inv(fk_trans).float(), device=self.device
        )  # B*K,N,3

        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)  # B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=False)
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)  # K,B*N,-1
            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
            # sdf
            sdf = (
                torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B, N).transpose(0, 1).reshape(B * K, N)
            )  # B,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf = sdf.transpose(1, 2)  # B, N, K
            return sdf
        else:
            phi, dphi = self.build_basis_function_from_points(x_bounded.reshape(B * K * N, 3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1, 4)  # K,B*N,-1,4

            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links], dim=0).to(self.device)

            output = (
                torch.einsum('ijkl,ik->ijl', phi_cat, weights_near)
                .reshape(K, B, N, 4)
                .transpose(0, 1)
                .reshape(B * K, N, 4)
            )
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * (scale.reshape(B, K).unsqueeze(-1))
            sdf_value = sdf

            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)  # (K, N, 3)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = (
                torch.einsum('ijk,ikl->ijl', fk_rotation, gradient.transpose(1, 2)).transpose(1, 2).reshape(B, K, N, 3)
            )
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx)[:, 0, :, :]
            # gradient_value = None
            return sdf_value, gradient_value

    def get_whole_body_sdf_with_joints_grad_batch(self, x, pose, theta, model, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        """
        calculate sdf and gradient with respect to joints (use finite difference)
        Args:
            x: (N,3), object points xyz
            pose: (B,4,4)
            theta: (B,7)
            model:
        """
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)  # (B,1,7)
        d_theta = (
            theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7, 7) * delta
        ).reshape(B, -1, 7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.unsqueeze(1).expand(B, 8, 4, 4).reshape(B * 8, 4, 4)
        sdf, _ = self.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=False, used_links=used_links)
        sdf = sdf.reshape(B, 8, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)

    def get_whole_body_normal_with_joints_grad_batch(
        self, x, pose, theta, model, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    ):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (
            theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7, 7) * delta
        ).reshape(B, -1, 7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.unsqueeze(1).expand(B, 8, 4, 4).reshape(B * 8, 4, 4)
        sdf, normal = self.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=True, used_links=used_links)
        normal = normal.reshape(B, 8, -1, 3).transpose(1, 2)
        return normal  # normal size: (B,N,8,3) normal[:,:,0,:] origin normal vector normal[:,:,1:,:] derivatives with respect to joints


def test_rdf_link_net():
    n_func = 24
    link_name = 'link0'
    ckpt_path: str = PATH_ROOT + f'/data/baselines/rdf/models/raw_coll/BP_{n_func}.pt'
    rnet = RDFRobotNet(device='device', ckpt_path=ckpt_path)
    lnet = rnet.linknets[link_name]
    exit()
    pass


if __name__ == '__main__':
    test_rdf_link_net()

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=24, type=int)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    bp_sdf = BPSDF(args.n_func, args.domain_min, args.domain_max, args.device)

    # args.train = True
    # #  train Bernstein Polynomial model
    if args.train:
        bp_sdf.train_ours_data()

    # load trained model
    # model_path = f'{OUR_DIR}/models/BP_{args.n_func}.pt'
    model_path = PATH_ROOT + f'/data/baselines/rdf/models/raw_coll/BP_{args.n_func}.pt'
    model = torch.load(model_path)

    x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3], device=args.device).unsqueeze(0)
    q = x[:, :7]
    pt = x[:, 7:]
    # [0.2328, 0.1837, 0.1081, 0.2829, 0.3573, 0.5066, 0.6961, 0.6422, 0.5790]
    # ret = rnet.dists_q_pt_pair(x)
    ret = bp_sdf.dists_q_pt_pair(pt, q, model)
    print(ret)

    pts = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], device=args.device)
    ret = bp_sdf.cal_qxpt(q, pts, model)
    print(ret)

    rnet = RDFRobotNet(args.device, model_path)
    ret1 = rnet.dists_g_via_diff(q, pts)
    print(ret1)
