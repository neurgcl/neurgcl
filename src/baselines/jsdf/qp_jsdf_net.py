import unittest

import torch

from jsdf.jsdf_net import JSDFNet
from neurg.my_utils.config import PATH_ROOT
from neurg.kinetics.cu_kin import get_fk_model

device = "cuda"


def get_jsdf_rnet(device="cuda", dtype=torch.float32):
    s = 256
    n_layers = 5
    skips = []
    fname = 'sdf_%dx%d_mesh_roscol.pt' % (s, n_layers)
    fname = PATH_ROOT + '/data/baselines/jsdf/' + fname
    if skips == []:
        n_layers -= 1
    tensor_args = {'device': device, 'dtype': dtype}
    nn_jsdf = JSDFNet(in_channels=10, out_channels=9, layers=[s] * n_layers, skips=skips)
    # print(repr(nn_jsdf.model))
    nn_jsdf.load_weights(fname, tensor_args)
    return nn_jsdf


class TestJSDF(unittest.TestCase):
    def setUp(self):
        # custom_torch_repr()
        pass
        # self.nn_jsdf = get_jsdf_net()

    def test_jsdf(self):
        rnet = get_jsdf_rnet()

        q = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], device=device)
        pts = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.5, 0.7]], device=device)
        kin_model = get_fk_model()

        T_w2links, J_ee_wrt_q = kin_model.fk1q_col_ee(q)
        dists, J_dists_wrt_q = rnet.pts2robot_dists_g(q, pts)

        print("dists:", dists.shape, dists)
        print("J_dists_wrt_q:", J_dists_wrt_q.shape, J_dists_wrt_q)
        print(J_dists_wrt_q[-4:, 0].cpu())
        print(J_dists_wrt_q[-4:, 1].cpu())

    def test_mpc(self):
        device = "cuda"
        dtype = torch.float32
        dtype = torch.float16

        rnet = get_jsdf_rnet(device, dtype)
        # qs = torch.rand((400 * 30, 7), device=device)
        qs = torch.rand((400, 30, 7), device=device, dtype=dtype)
        pts = torch.rand((300, 3), device=device, dtype=dtype)  # F32: 100,6.6G; 300,19G; F16: 300,9.7G
        dists = rnet.mpc_dists(qs, pts)
        print("dists:", dists.shape)


if __name__ == "__main__":

    unittest.main()
