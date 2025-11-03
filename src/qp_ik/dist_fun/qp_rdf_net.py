import torch
from rdf.rdf_model import RDFRobotNet

from neurg.utils.model_helper import get_rdf_rnet
from qp_ik.dist_fun.qp_base_net import QPBaseNet
from qp_ik.utils import robot_kin

class QPRDFNet(QPBaseNet):
    def __init__(self, *, device="cuda", precision=32, remove_base:bool=False, aot:bool=False):
        self.device = device

        if precision == 32:
            self.dtype = torch.float32
        elif precision == 16:
            self.dtype = torch.float16
        else:
            raise ValueError("precision should be 32 or 16")
        
        self.tensor_args = {'device': self.device, 'dtype': self.dtype}
        self.remove_base = remove_base
        self.aot = aot
        
        self.n_col_links = 9
        self.rnet:RDFRobotNet = get_rdf_rnet(device=device)

    def convert2mmlp_x(self, q, pts, T_w2links=None):
        if T_w2links is None:
            tensor_q = torch.tensor(q, **self.tensor_args)
            if self.aot:
                tensor_T_w2links=self.rnet.aot_fk_tfs(tensor_q)
            else:
                tensor_T_w2links = robot_kin.tor_fk_tfs(tensor_q)[:,:3,].contiguous()
            tensor_T_w2links = tensor_T_w2links.view(-1, 3, 4)[: self.n_col_links]
        else:
            tensor_T_w2links = T_w2links[:self.n_col_links, :3].clone().detach().to(**self.tensor_args)

        tensor_pts = torch.tensor(pts, **self.tensor_args)
        return tensor_pts, tensor_T_w2links
    
    def cal_Gamma_g_all(self, q, pts, T_w2links=None):
        """
        q:                  (n_act_joint)
        tensor_pts:         (n_pt, 3)
        tensor_T_w2links:   (n_col_link, 3, 4) , include base_link
        """
        tensor_q =torch.tensor(q, **self.tensor_args).unsqueeze(0)
        tensor_pts = torch.tensor(pts, **self.tensor_args)
        if self.remove_base:
            used_links = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            used_links = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        batch_sz = 2000
        n_pts = tensor_pts.shape[0]
        n_batch = n_pts // batch_sz
        if n_pts % batch_sz != 0:
            n_batch += 1
        dists = []
        grads = []
        for i in range(n_batch):
            _tensor_pts = tensor_pts[i*batch_sz:(i+1)*batch_sz]
            _dists, _grads = self.rnet.dists_g_via_diff(tensor_q, _tensor_pts, used_links=used_links)
            dists.append(_dists)
            grads.append(_grads)
        dists = torch.concat(dists, dim=-1)
        grads = torch.concat(grads, dim=-2)
        # dists, grads=self.rnet.dists_g_via_diff(tensor_q, tensor_pts, used_links=used_links)
        dists = dists.reshape(-1)
        grads = grads.reshape(-1, 7)
        return dists.cpu().numpy(), grads.cpu().numpy()

    def cal_Gamma_g_min(self, q, pts, T_w2links=None):
        pose = torch.eye(4).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        theta = torch.tensor(q).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        pts = torch.tensor(pts).to(device=self.device, dtype=self.dtype)
        if self.remove_base:
            used_links = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            used_links = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        batch_sz = 2000
        n_pts = pts.shape[0]
        n_batch = n_pts // batch_sz
        if n_pts % batch_sz != 0:
            n_batch += 1
        dists = []
        grads = []
        for i in range(n_batch):
            _pts = pts[i*batch_sz:(i+1)*batch_sz]
            _dists, _grads = self.rnet.bp_sdf.get_whole_body_sdf_with_joints_grad_batch(_pts, pose, theta, self.rnet.model,used_links=used_links)
            dists.append(_dists)
            grads.append(_grads)
        dists = torch.concat(dists, dim=-1)
        grads = torch.concat(grads, dim=-2)
        # dists, grads = self.rnet.bp_sdf.get_whole_body_sdf_with_joints_grad_batch(pts, pose, theta, self.rnet.model,used_links=used_links)
        return dists.squeeze(0).cpu().numpy(), grads.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    import numpy as np
    q = np.random.rand(7) * 2 * np.pi - np.pi
    pts = np.random.rand(10000, 3) * 1
    qp_net = QPRDFNet()
    bp_sdf =qp_net.rnet.bp_sdf
    dists, grads = qp_net.cal_Gamma_g_all(q, pts)
    print(dists.shape, grads.shape)
    dists, grads = qp_net.cal_Gamma_g_min(q, pts)
    print(dists.shape, grads.shape)

    pass