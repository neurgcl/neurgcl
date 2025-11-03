import pandas as pd
import torch
from jsdf.eval import get_jsdf_net


from neurg.eval.eval_helper import analysis_link_sdf_grad_err
from neurg.utils.model_helper import get_link_net, get_robot_net
from neurg.kinetics.cu_kin import get_fk_model
from neurg.l_utils.torch.to_tensor import to_tensor
from neurg.l_utils.torch.torch_helper import try_cuda
from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.l_utils.torch.xy_dataset import batch_predict
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.torch_tf import b_inv_mat44, b_pair_tf_pt
from neurg.my_utils.util_file import load_pickle
from neurg.my_utils.util_log import get_time_str


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='nsdf')
    # parser.add_argument('--method', type=str, default='jsdf')
    # parser.add_argument('--method', type=str, default='rdf')
    args = parser.parse_args()

    cfg = {
        'time_str': get_time_str(),
        'device': str(try_cuda()),
        'scale': 1000,
    }
    cfg.update(vars(args))

    device = cfg['device']
    batch_size = 500000
    
    net_param = {}
    if cfg["method"] == "nsdf":
        net_param = {
            "hid_dim": 32,
            "hid_sz": 3,
            "actv_type": "softplus",
            "eikonal": True,
        }
    elif cfg["method"] == "rdf":
        net_param = {'n_func': 24}
        batch_size = min(batch_size, 1000)
    elif cfg['method'] == 'jsdf':
        net_param = {}
    else:
        raise ValueError(f"Unknown method: {cfg['method']}")

    rnet = get_robot_net(cfg["method"], device=device, **net_param)
    cfg["method_w_param"] = rnet.get_method_w_param()

    kin_model = get_fk_model()
    d_grad = load_pickle(PATH_ROOT + '/data/baselines/jsdf/d_test_grad.pkl')
    d_grad = to_tensor(d_grad, dtype=torch.float32)
    x = d_grad['x']
    y = d_grad['y']
    grad = d_grad['grad']
    qs = x[:, :7]
    pts = x[:, 7:]

    y_gts = torch.cat([y.reshape(-1, 1), grad.reshape(-1, 3)], dim=1)

    # eval distance -----------------------------------------
    dists = batch_predict(rnet.dists_q_pt_pair, SliceBatchLoader(x, batch_size), device).cpu()

    # eval gradients -----------------------------------------
    # prepare local points
    T_w2links = kin_model.fk_link_T44(qs.cuda())
    T_w2links = T_w2links[:, :-1].contiguous()

    T_links2w = b_inv_mat44(T_w2links)
    pts_local = b_pair_tf_pt(T_links2w, pts.cuda())
    pts_local = pts_local.transpose(0, 1).contiguous() # (L, N, 3) points in local link frame

    # compute gradients
    if cfg["method"] == "nsdf":
        _, dists_g = rnet.l_dist(pts_local, ret_grad=True)
        dists_g = dists_g.transpose(0, 1).cpu()  # (N, L, 3)

    elif cfg['method'] == 'jsdf':
        rnet = get_jsdf_net()
        _, grads, minidxMask = rnet.compute_signed_distance_wgrad(x.cuda())
        grads = grads[:, 7:] * 0.01
        grads = grads.transpose(-1, -2).cpu()

        dists_g = []
        for i in range(9):
            grad = b_pair_tf_pt(T_links2w[:, [i]][:, :, :3].contiguous().cpu(), grads[:, [i]].reshape(-1, 3).cpu())
            dists_g.append(grad)
        dists_g = torch.concat(dists_g, dim=1)

    elif cfg['method'] == 'rdf':
        l_link_name = [f'link{i}' for i in range(8)] + ['hand']

        dists_g = []
        for idx_l in range(len(l_link_name)):
            link_name = l_link_name[idx_l]
            lnet = get_link_net(rnet, link_name)
            pt1 = pts_local[idx_l]
            dl = SliceBatchLoader(pt1, batch_size=1000)
            sdfs_pred, grads_pred = batch_predict(lnet.forward_backward, dl, device)
            dists_g.append(grads_pred)
        dists_g = torch.stack(dists_g, dim=1).cpu()

    y_preds = torch.cat([dists.reshape(-1, 1), dists_g.reshape(-1, 3)], dim=1)
    l_infos = analysis_link_sdf_grad_err(y_preds, y_gts, scale=cfg['scale'])
    l_logdata = []
    for info in l_infos:
        logdata = dict(
            **cfg,
            **net_param,
            **info,
        )
        l_logdata.append(logdata)

    # show results -----------------------------------------
    df = pd.DataFrame(l_logdata)
    df = df[df["pt_type"].isin(['close', 'far_in_bbox', 'all_in_bbox'])].copy()
    pt_type_map = {
        'close': 'Close',
        'far_in_bbox': 'Far',
        'all_in_bbox': 'Average',
    }
    df['pt_type'] = df['pt_type'].map(pt_type_map)

    val_cols = ["Dis/l1/mean", "Dis/l1/std", "Grad/mean", "Grad/std"]
    df[val_cols] = df[val_cols].round(2)
    df["MDE"] = df["Dis/l1/mean"].astype(str) +' ± '+ df["Dis/l1/std"].astype(str)
    df["MGE"] = df["Grad/mean"].astype(str) +' ± '+ df["Grad/std"].astype(str)

    show_cols = ["method", "pt_type", "MDE", "MGE"]
    df_show = df[show_cols]
    df_show.set_index(["pt_type"], inplace=True)
    df_show = df_show.loc[["Average", "Close", "Far"]].reset_index()
    df_show = df_show[show_cols]
    print(df_show)
