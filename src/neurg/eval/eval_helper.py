import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import AUROC
from torchmetrics.classification import BinaryRecall

from neurg.l_utils.helper import dict2float


def cal_statics(vals, prefix: str = ""):
    """ Calculate statics of a list of values. """

    if isinstance(vals, torch.Tensor):
        vals = vals.cpu().numpy()
    else:
        vals = np.array(vals)

    dic = {
        prefix + "mean": vals.mean(),
        prefix + "std": vals.std(),
        prefix + "max": vals.max(),
        prefix + "min": vals.min(),
        prefix + "median": np.median(vals),
    }
    return dic


def metrics_reg_cls(tmp_pred, tmp_gt, scale=1000):
    d_out = {}
    cls_gt = tmp_gt <= 0.0
    n_data = len(tmp_pred)
    d_out["n_data"] = n_data
    d_out["n_{d<=0}"] = torch.sum(cls_gt)
    d_out["n_{d>0}"] = n_data - d_out["n_{d<=0}"]

    # distance(pt, links)
    err_l1 = torch.abs(tmp_pred - tmp_gt) * scale

    d_err_l1 = cal_statics(err_l1, prefix="l1/")
    d_out.update(d_err_l1)

    err_sorted = torch.sort(err_l1)[0]
    err_r95 = err_sorted[int(len(err_sorted) * 0.95)]
    err_r99 = err_sorted[int(len(err_sorted) * 0.99)]
    err_r999 = err_sorted[int(len(err_sorted) * 0.999)]
    d_out["err_r25"] = err_sorted[int(len(err_sorted) * 0.25)]
    d_out["err_r75"] = err_sorted[int(len(err_sorted) * 0.75)]
    d_out["err_r95"] = err_r95
    d_out["err_r99"] = err_r99
    d_out["err_r999"] = err_r999

    if sum(cls_gt == 1) > 0:
        safe_thr = tmp_pred[cls_gt == 1].max()
        FP = ((tmp_pred <= safe_thr) & (cls_gt == 0)).type(torch.int).sum()
        TN = ((tmp_pred > safe_thr) & (cls_gt == 0)).type(torch.int).sum()
        safe_FPR = FP / (FP + TN)
        d_out["safe_FPR"] = safe_FPR
    else:
        d_out["safe_FPR"] = -99

    auroc = AUROC(task="binary")
    auroc_val = auroc(-F.sigmoid(tmp_pred), cls_gt)
    d_out["AUROC"] = auroc_val

    classify_metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary"),
            torchmetrics.F1Score(task="binary"),
        ]
    ).to(tmp_pred.device)
    bin_cls_threshold = 0.0
    cls_pred = tmp_pred <= bin_cls_threshold
    cls_metrics = classify_metrics(cls_pred, cls_gt)
    d_out.update(cls_metrics)

    bin_cls_threshold = 0.01
    recall_at_10mm = BinaryRecall().to(tmp_pred.device)((tmp_pred <= bin_cls_threshold), cls_gt)  # TP/(TP+FN)
    d_out["recall@10mm"] = recall_at_10mm.item()

    d_out = dict2float(d_out)
    return d_out


def analysis_link_error(
    l_y_pred, l_y_gt, scale=1000, thresh_close=0.1, thresh_bbox=0.2, partition=True, **kwargs
):
    if not partition:
        d_flag = {'all':torch.ones_like(l_y_gt, dtype=torch.bool)}
    else:
        flag_close_in_bbox = l_y_gt <= thresh_close
        flag_far = ~flag_close_in_bbox
        flag_far_in_bbox = (l_y_gt > thresh_close) & (l_y_gt <= thresh_bbox)
        flag_all_in_bbox = l_y_gt <= thresh_bbox
        flag_far_out_bbox = ~flag_all_in_bbox
        flag_all = torch.ones_like(l_y_gt, dtype=torch.bool)
        d_flag = {
            "close": flag_close_in_bbox,
            "far": flag_far,
            "all": flag_all,
            "far_in_bbox": flag_far_in_bbox,
            "all_in_bbox": flag_all_in_bbox,
            "far_out_bbox": flag_far_out_bbox,
        }

    l_d_out = []
    for pt_type in d_flag.keys():
        tmp_flag = d_flag[pt_type]
        tmp_pred = l_y_pred[tmp_flag]
        tmp_gt = l_y_gt[tmp_flag]

        if len(tmp_pred) == 0:
            continue

        d_out = {}
        d_out["pt_type"] = pt_type
        d_out["thresh_close"] = thresh_close * scale
        d_out["thresh_bbox"] = thresh_bbox * scale
        d_out["scale"] = scale
        metrics = metrics_reg_cls(tmp_pred, tmp_gt, scale)
        d_out.update(metrics)
        d_out.update(kwargs)
        l_d_out.append(d_out)

    if not partition:
        l_d_out = l_d_out[0]
    return l_d_out


def cal_grad_err(g_pred, g_gt, grad_scale=1):
    if isinstance(g_pred, np.ndarray):
        g_pred = torch.from_numpy(g_pred)
    if isinstance(g_gt, np.ndarray):
        g_gt = torch.from_numpy(g_gt)

    g_gt = g_gt.to(torch.float64)
    g_pred = g_pred.to(torch.float64)

    eikonal_mse_loss = torch.square(torch.linalg.norm(g_pred, dim=-1) - 1)
    eikonal_l1_loss = torch.abs(torch.linalg.norm(g_pred, dim=-1) - 1)

    g_pred_normed = g_pred / torch.linalg.norm(g_pred, axis=-1).unsqueeze(-1)
    g_gt_normed = g_gt / torch.linalg.norm(g_gt, axis=-1).unsqueeze(-1)

    err_grad = torch.linalg.norm(g_pred - g_gt_normed, dim=-1)
    err_grad_normed = torch.linalg.norm(g_pred_normed - g_gt_normed, dim=-1)

    err_grad_angle = torch.rad2deg(torch.acos((g_gt_normed * g_pred_normed).sum(dim=-1)))
    out = {
        **cal_statics(err_grad * grad_scale, prefix='Grad/'),  # L2 err
        **cal_statics(err_grad_normed * grad_scale, prefix='GradN/'),  # normed L2 err
        **cal_statics(err_grad_angle, prefix='GradAng/'),  # angle err (degree)
        **cal_statics(eikonal_mse_loss, prefix='GradEikMSE/'),
        **cal_statics(eikonal_l1_loss, prefix='GradEikL1/'),
        'grad_scale': grad_scale,
    }
    return out

def analysis_link_sdf_grad_err(
    y_preds, y_gts, scale=1000, grad_scale=1, thresh_close=0.1, thresh_bbox=0.2, partition=True, **kwargs
):
    """
    Args:
        y_gts (torch.Tensor) [Nx4]: Ground truth SDF and gradients
        y_preds (torch.Tensor) [Nx4]: Predicted SDF and gradients
        scale (float): 1000 to convert the error to mm
    """

    dist_gt = y_gts[:, 0]
    grad_gt = y_gts[:, 1:]

    dist_pred = y_preds[:, 0]
    grad_pred = y_preds[:, 1:]

    if not partition:
        d_flag = {'all':torch.ones_like(dist_gt, dtype=torch.bool)}
    else:
        flag_close_in_bbox = dist_gt <= thresh_close
        flag_far = ~flag_close_in_bbox
        flag_far_in_bbox = (dist_gt > thresh_close) & (dist_gt <= thresh_bbox)
        flag_all_in_bbox = dist_gt <= thresh_bbox
        flag_far_out_bbox = ~flag_all_in_bbox
        flag_all = torch.ones_like(dist_gt, dtype=torch.bool)
        d_flag = {
            'close': flag_close_in_bbox,
            'far': flag_far,
            'all': flag_all,
            'far_in_bbox': flag_far_in_bbox,
            'all_in_bbox': flag_all_in_bbox,
            'far_out_bbox': flag_far_out_bbox,
        }
    l_d_out = []
    for pt_type in d_flag.keys():
        tmp_flag = d_flag[pt_type]
        tmp_dist_pred = dist_pred[tmp_flag]
        tmp_dist_gt = dist_gt[tmp_flag]

        if len(tmp_dist_pred) == 0:
            continue

        d_out = {}
        d_out['pt_type'] = pt_type
        d_out['thresh_close'] = thresh_close * scale
        d_out['thresh_bbox'] = thresh_bbox * scale

        dis_err = (tmp_dist_gt - tmp_dist_pred).abs() * scale
        dis_metrics = cal_statics(dis_err, prefix='Dis/l1/')
        grad_metrics = cal_grad_err(grad_pred[tmp_flag], grad_gt[tmp_flag], grad_scale=grad_scale)

        d_out.update(dis_metrics)
        d_out.update(grad_metrics)
        d_out.update(kwargs)
        l_d_out.append(d_out)
    
    if not partition:
        l_d_out = l_d_out[0]
    return l_d_out

def df_print(dic, show=True):
    if isinstance(dic, dict):
        df = pd.DataFrame([dic])
    else:
        df = pd.DataFrame(dic)
    if show:
        print(df)
    return df
