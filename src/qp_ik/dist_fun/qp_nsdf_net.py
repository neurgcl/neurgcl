import torch

from neurg.api.nsdf_model import NSDFModel
from neurg.cfg import ROBOT_CKPT_PATH
from qp_ik.dist_fun.qp_base_net import QPBaseNet
from qp_ik.utils import robot_kin


def convert_to_tensor(x, *, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        return torch.tensor(x, device=device, dtype=dtype)
    
class QPNSDFNet(QPBaseNet):
    def __init__(self, *, robot_ckpt_path=ROBOT_CKPT_PATH, device="cuda", precision=32, remove_base:bool=False):
        self.device = device

        if precision == 32:
            self.dtype = torch.float32
        elif precision == 16:
            self.dtype = torch.float16
        else:
            raise ValueError("precision should be 32 or 16")
        
        self.tensor_args = {'device': self.device, 'dtype': self.dtype}
        self.remove_base = remove_base
        
        self.n_col_links = 9
        self.idx_used_link = torch.arange(9)
        self.rnet:NSDFModel = NSDFModel(robot_ckpt_path=robot_ckpt_path, device=device)

    def convert_x(self, q, pts, T_w2links=None):
        if T_w2links is None:
            q = convert_to_tensor(q, **self.tensor_args)
            T_w2links = robot_kin.tor_fk_tfs(q)[:self.n_col_links,:3,].contiguous()
        else:
            T_w2links = T_w2links[:self.n_col_links, :3].detach().to(**self.tensor_args).contiguous()

        pts = convert_to_tensor(pts, **self.tensor_args)
        return pts, T_w2links

    def cal_Gamma_g_all(self, q, pts, T_w2links=None, keep_dim=False, cuda_fmt=False):
        """
        Args:
            q:           (n_act_joint)
            pts:         (n_pt, 3)
            T_w2links:   (n_col_link, 3, 4) , include base_link

        Returns:
            dists:      [n_col_link(-1) * n_pt]
            dists_g:    [n_col_link(-1) * n_pt, n_act_joint]
        """
        tensor_pts, tensor_T_w2links = self.convert_x(q, pts, T_w2links)
        dists, grads = self.rnet.g_dist(tensor_pts, tensor_T_w2links, ret_grad=True)

        if self.remove_base:
            dists = dists[1:]
            grads = grads[1:]

        if not keep_dim:
            dists =  dists.reshape(-1)
            grads = grads.reshape(-1, 7)

        if cuda_fmt:
            return dists, grads
        else:
            return dists.cpu().numpy(), grads.cpu().numpy()

    def cal_Gamma_g_min(self, q, pts, T_w2links=None):
        """

        Args:
            q:
            pts:
            T_w2links:

        Returns:
            dists:      [n_pt]
            dists_g:    [n_pt, n_act_joint]
        """
        tensor_pts, tensor_T_w2links = self.convert_x(q, pts, T_w2links)
        dists, grads = self.rnet.g_dist(tensor_pts, tensor_T_w2links, ret_grad=True)

        if self.remove_base:
            dists = dists[1:]   # (n_link, n_pt)
            grads = grads[1:]   # (n_link, n_pt, n_act_joint)

        dists_min, idx_min = torch.min(dists,dim=0)

        N_JOINT = grads.shape[-1]
        grads_min = grads.gather(0, idx_min.unsqueeze(-1).unsqueeze(0).expand(-1, -1, N_JOINT)).squeeze(0)

        return dists_min.cpu().numpy(), grads_min.cpu().numpy()

def test_val(qp_net:QPNSDFNet):
    from neurg.my_utils import np_tf

    device = 'cuda'

    q = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], device=device)
    pts = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.5, 0.7]], device=device)

    dists, grads = qp_net.cal_Gamma_g_all(q, pts, keep_dim=True)

    print("cal_ik_vals " + '=' * 20)
    print("q:", q.shape, q)
    print("pts:", pts.shape, pts)
    # T_w2links, J_ee_wrt_q = qp_net.rnet.kin.qp_fk_T44s_J(q)

    # print("T_w2links" + '-' * 20)
    # print("T_w2links:", T_w2links.shape, T_w2links)
    # print(np_tf.T2xyzrpy(T_w2links[-2].cpu().numpy(), degrees=True).tolist())
    # print(np_tf.T2xyzrpy(T_w2links[-1].cpu().numpy(), degrees=True).tolist())

    print("dists:", dists.shape, dists)
    # print("J_dists_wrt_q:", grads.shape, grads)
    # print(grads[-4:, 0])
    # print(grads[-4:, 1])


if __name__ == "__main__":
    import numpy as np
    q = np.random.rand(7) * 2 * np.pi - np.pi
    pts = np.random.rand(100, 3) * 1
    qp_net = QPNSDFNet()
    # dists, grads = qp_net.cal_Gamma_g_all(q, pts)
    # print(dists.shape, grads.shape)
    # dists, grads = qp_net.cal_Gamma_g_min(q, pts)
    # print(dists.shape, grads.shape)

    test_val(qp_net)