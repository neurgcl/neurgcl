
import torch
from jsdf.jsdf_net import JSDFNet
from jsdf.qp_jsdf_net import get_jsdf_rnet

from qp_ik.dist_fun.qp_base_net import QPBaseNet


class QPJSDFNet(QPBaseNet):
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
        self.rnet:JSDFNet = get_jsdf_rnet(self.device)

    def convert2jsdf_x(self, q, pts):
        tensor_q = torch.tensor(q, device=self.device, dtype=self.dtype)
        tensor_pts = torch.tensor(pts, device=self.device, dtype=self.dtype)
        return tensor_q, tensor_pts

    def cal_Gamma_g_all(self, q, pts, T_w2links=None):
        """return: all dists (not sorted)"""
        tensor_q, tensor_pts = self.convert2jsdf_x(q, pts)
        if self.aot:
            dists, grads = self.rnet.aot_dists_g(tensor_q, tensor_pts)
        else:
            dists, grads = self.rnet.pts2robot_dists_g(tensor_q, tensor_pts)

        if self.remove_base:
            dists = dists[1:]  # ignore base_link (8, n_pt)
            grads = grads[1:]  # (8, n_pt, n_act_joint)

        return dists.reshape(-1).cpu().numpy(), grads.reshape(-1, 7).cpu().numpy()

    def cal_Gamma_g_min(self, q, pts, T_w2links=None):
        tensor_q, tensor_pts = self.convert2jsdf_x(q, pts)

        # net_x = torch.concat([tensor_q.unsqueeze(0).repeat([tensor_pts.shape[0], 1]), tensor_pts], axis=1)
        # dist, grad, minidxMask = self.jsdf_net.dist_grad_closest_aot(net_x)
        # dist_min = dist[list(range(pts.shape[0])), minidxMask]
        # scale = 0.01
        # return dist_min.cpu().numpy()*scale, grad.cpu().numpy()*scale

        if self.aot:
            dists, grads = self.rnet.aot_dists_g(tensor_q, tensor_pts)
        else:
            dists, grads = self.rnet.pts2robot_dists_g(tensor_q, tensor_pts)

        if self.remove_base:
            dists = dists[1:]  # ignore base_link (8, n_pt)
            grads = grads[1:]  # (8, n_pt, n_act_joint)

        d_sorted, idx_sorted = torch.sort(dists, dim=0)
        grads_sorted = torch.gather(grads, 0, idx_sorted.unsqueeze(-1).expand(-1, -1, 7))

        n_min = 1
        dist_min = d_sorted[:n_min].reshape(-1)
        grad_min = grads_sorted[:n_min].reshape(-1, 7)
        return dist_min.cpu().numpy(), grad_min.cpu().numpy()