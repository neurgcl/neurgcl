from neurg.kinetics.cu_kin import get_fk_model
import numpy as np
import torch

kin_model = get_fk_model()
dtype=torch.float32


def tor_fk_tfs(q):
    if isinstance(q, np.ndarray):
        q = torch.tensor(q, device='cuda', dtype=dtype)
    return kin_model.qp_fk_T44s(q)

def tor_fk_tf_jacos(q):
    if isinstance(q, np.ndarray):
        q = torch.tensor(q, device='cuda', dtype=dtype)
    return kin_model.qp_fk_T44s_J(q)

def fk_tfs(q)->np.ndarray:
    ret = tor_fk_tfs(q)
    return ret.cpu().numpy()

def fk_tf_jacos(q)->np.ndarray:
    ret = tor_fk_tf_jacos(q)
    return ret[0].cpu().numpy(), ret[1].cpu().numpy()