import torch
import torch.nn.functional as F
from jsdf.l_jsdf import JData, JsdfDataset
from jsdf.sdf.robot_sdf import RobotSdfCollisionNet

from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.my_utils.config import PATH_ROOT


def get_dl_test(device="cuda", batch_size=500000):
    mat_path = PATH_ROOT + f'/data/baselines/jsdf/data_mesh_roscol.mat'
    jdata = JData(device, data_dir=mat_path)
    ds_test = JsdfDataset(jdata, 'test')
    # ds_test = JsdfDataset(jdata, 'val')
    # ds_test = JsdfDataset(jdata, 'train')
    print(f"len(ds_test): {len(ds_test)}")
    dl = SliceBatchLoader(ds_test, batch_size=batch_size)
    return dl


def get_jsdf_net(device="cuda"):
    s = 256
    n_layers = 5
    skips = []
    fname = 'sdf_%dx%d_mesh_roscol.pt' % (s, n_layers)
    fname = PATH_ROOT + '/data/baselines/jsdf/' + fname
    if skips == []:
        n_layers -= 1
    tensor_args = {'device': device, 'dtype': torch.float32}
    nn_jsdf = RobotSdfCollisionNet(in_channels=10, out_channels=9, layers=[s] * n_layers, skips=skips)
    nn_jsdf.load_weights(fname, tensor_args)
    return nn_jsdf


if __name__ == "__main__":
    device = "cuda"
    nn_jsdf = get_jsdf_net()

    batch_size = 500000
    l_y_pred = []
    l_y_gt = []
    l_loss = []

    dl = get_dl_test()
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            y_pred = nn_jsdf.dists_q_pt_pair(x)
            l_y_pred.append(y_pred)
            l_y_gt.append(y)
            loss = F.l1_loss(y_pred, y, reduction='none')
            l_loss.append(loss)
    l_y_pred = torch.cat(l_y_pred, dim=0)
    l_y_gt = torch.cat(l_y_gt, dim=0)
    l_loss = torch.cat(l_loss, dim=0)

    flag_close = l_y_gt <= 0.03
    flag_far_in_bbox = (l_y_gt > 0.03) & (l_y_gt <= 0.2)
    flag_far_out_bbox = (~flag_close) & (~flag_far_in_bbox)

    tmp_flag = flag_close
    tmp_pred = l_y_pred[tmp_flag]
    tmp_gt = l_y_gt[tmp_flag]
    n_data = len(tmp_pred)

    # distance(pt, links)
    err_l1 = torch.abs(tmp_pred - tmp_gt)
    err_l1_mean = err_l1.mean().item()
    err_l1_std = err_l1.std().item()
    err_l1_max = err_l1.max().item()
    err_l1_min = err_l1.min().item()

    # min distance

    print(f"loss: {l_loss.mean()}")
