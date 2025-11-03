from typing import Union

import torch
from jsdf.jsdf_net import JSDFNet
from jsdf.l_jsdf import JData, JsdfDataset
from rdf.rdf_model import RDFLinkNet, RDFRobotNet

from neurg.api.nsdf_model import NSDFLModel, NSDFModel
from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.my_utils.config import PATH_ROOT


# Dataset ---------------------------------------------------------------------
def get_ds_jsdf_test(device="cuda"):
    mat_path = PATH_ROOT + f"/data/baselines/jsdf/data_mesh_roscol.mat"
    jdata = JData(device, data_dir=mat_path)
    jdata.y = jdata.y * 0.01
    ds_test = JsdfDataset(jdata, "test")
    return ds_test

def get_dl_jsdf_test(device="cuda", batch_size=500000):
    ds_test = get_ds_jsdf_test(device=device)
    print(f"len(ds_test): {len(ds_test)}")
    dl = SliceBatchLoader(ds_test, batch_size=batch_size)
    return dl


# Network ---------------------------------------------------------------------
def get_jsdf_rnet(device="cuda"):
    s = 256
    n_layers = 5
    skips = []
    fname = "sdf_%dx%d_mesh_roscol.pt" % (s, n_layers)
    fname = PATH_ROOT + "/data/baselines/jsdf/" + fname
    if skips == []:
        n_layers -= 1
    tensor_args = {"device": device, "dtype": torch.float32}
    nn_jsdf = JSDFNet(
        in_channels=10, out_channels=9, layers=[s] * n_layers, skips=skips
    )
    nn_jsdf.load_weights(fname, tensor_args)
    return nn_jsdf


def get_nsdf_rnet(
    hid_dim=32, hid_sz=3, actv_type="softplus", eikonal=True, device="cuda"
) -> NSDFModel:
    str_eikonal = "_nograd" if not eikonal else ""
    robot_sdf_ckpt = (
        PATH_ROOT
        + f"/logs/nsdf/link_infos/{actv_type}_coll{str_eikonal}_{hid_dim}x{hid_sz}.pth"
    )
    rnet = NSDFModel(robot_sdf_ckpt, device=device)
    return rnet

def get_nsdf_lnet(rnet: NSDFModel = None, link_name: str = "link0") -> NSDFLModel:
    if rnet is None:
        rnet = get_nsdf_rnet()
    model = rnet.get_link_net(link_name)
    model.to(rnet.device)
    return model


def get_rdf_rnet(n_func=None, ckpt_path=None, device="cuda") -> RDFRobotNet:
    """
    ckpt_path = PATH_ROOT + '/data/baselines/rdf/models/raw_coll/BP_24.pt'  # dataset gen by rdf
    """
    # assert n_func and ckpt_path are not given at the same time
    assert not ((n_func is not None) and (ckpt_path is not None))
    
    if ckpt_path is None:
        if n_func is None:
            n_func = 24

        ckpt_path: str = (
            PATH_ROOT + f"/data/baselines/rdf/models/raw_coll/BP_{n_func}.pt"
        )
    rnet = RDFRobotNet(device=device, ckpt_path=ckpt_path)
    return rnet


def get_rdf_lnet(rnet: RDFRobotNet = None, link_name: str = "link0") -> RDFLinkNet:
    if rnet is None:
        rnet = get_rdf_rnet()
    lnet = rnet.linknets[link_name]
    return lnet


def get_robot_net(method, **kwargs) -> Union[RDFRobotNet, JSDFNet, NSDFModel]:
    """
    Get the robot net
    """
    if method == "rdf":
        return get_rdf_rnet(**kwargs)
    elif method == "jsdf":
        return get_jsdf_rnet(**kwargs)
    elif method == "nsdf":
        return get_nsdf_rnet(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_link_net(
    rnet: Union[NSDFModel, RDFRobotNet], link_name: str
) -> Union[NSDFLModel, RDFLinkNet]:
    """
    Get the link net
    """
    if isinstance(rnet, NSDFModel):
        return get_nsdf_lnet(rnet, link_name)
    elif isinstance(rnet, RDFRobotNet):
        if link_name == "hand":
            try:
                return get_rdf_lnet(rnet, link_name=link_name)
            except KeyError:
                return get_rdf_lnet(rnet, link_name="link8")
        return get_rdf_lnet(rnet, link_name=link_name)
    else:
        raise ValueError(f"Unknown rnet type: {type(rnet)}")


