from typing import Dict

import torch
from torch import nn

from neurg.net.basic_model import mlp
from neurg.net.grad_helper import cal_grad
from neurg.net.l_model_linksdf import LinkSdfTorModel


class LinkSdfModel(nn.Module):
    def __init__(self, hid_dim=256, hid_size=3, actv_type="softplus", beta=100):
        super().__init__()
        self.input_dim = 3
        self.output_dim = 1

        self.hid_dim = hid_dim
        self.hid_size = hid_size

        if actv_type == "softplus":
            actv = nn.Softplus(beta)
        elif actv_type == "relu":
            actv = nn.ReLU
        else:
            raise NotImplementedError

        self.mlp = mlp(
            [self.input_dim] + [self.hid_dim] * self.hid_size + [self.output_dim],
            activation=actv,
            output_activation=nn.Identity,
        )
        bbox_largest = torch.zeros((2, 3))
        bbox_extents = torch.zeros(3)
        bbox_raw = torch.zeros((2, 3))
        bbox_center = torch.zeros(3)
        bbox_radius = torch.zeros(1)
        bbox_radius_padded = torch.zeros(1)

        self.register_buffer("bbox_largest", bbox_largest)  # bbox_min, bbox_max
        self.register_buffer("bbox_extents", bbox_extents)  # bbox_min, bbox_max
        self.register_buffer("bbox_raw", bbox_raw)  # bbox_min, bbox_max
        self.register_buffer("bbox_center", bbox_center)
        self.register_buffer("bbox_radius", bbox_radius)
        self.register_buffer("bbox_radius_padded", bbox_radius_padded)

    def forward(self, x):
        y = self.mlp(x - self.bbox_center)
        return y
    
    def forward_backward(self, x: torch.Tensor):
        with torch.enable_grad():
            x.grad = None
            x.requires_grad_()
            y = self.forward(x)
            grad = cal_grad(y, x)
            x.requires_grad_(False)
        return y, grad

    def load_bbox(self, bbox):
        tensor_args = {"dtype": self.bbox_raw.dtype, "device": self.bbox_raw.device}

        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().clone().to(**tensor_args)
        else:
            bbox = torch.tensor(bbox).to(**tensor_args)
        bbox_extents = bbox[1] - bbox[0]
        bbox_center = bbox[0] + bbox_extents / 2.0
        self.bbox_raw = bbox
        self.bbox_extents = bbox_extents
        self.bbox_center = bbox_center

    def load_from_link_info(self, link_info: Dict[str, torch.Tensor]):
        tensor_args = {"dtype": self.bbox_raw.dtype, "device": self.bbox_raw.device}
        l_key = ['bbox_largest', 'bbox_extents', 'bbox_raw', 'bbox_center', 'bbox_radius', 'bbox_radius_padded']
        for k in l_key:
            if k not in link_info:
                raise ValueError(f"link_info should contain key: {k}")
            setattr(self, k, link_info[k].detach().clone().to(**tensor_args))
        model:LinkSdfTorModel = link_info['model']
        
        self.hid_dim = model.hid_dim
        self.hid_size = model.hid_size
        self.mlp = model.mlp

    def load_weights(self, robot_ckpt_path: str, link_name: str, device: str = "cuda"):
        link_infos = torch.load(robot_ckpt_path)
        link_info = link_infos[link_name]
        
        self.load_from_link_info(link_info)
        self.eval()
        self.requires_grad_(False)
        self.to(device)


if __name__ == "__main__":
    path, link_name = "logs/link_infos/softplus_coll_32x3.pth", "hand"
    link_infos = torch.load(path)
    link_info = link_infos[link_name]
    model = link_info["model"]
    lsdf = LinkSdfModel()
    lsdf.load_weights(path, link_name)

    pts = torch.randn(10, 3).to("cuda")
    y = lsdf.forward(pts)
    y,grad = lsdf.forward_backward(pts)
    print("done!")