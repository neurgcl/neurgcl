import os
import numpy as np
import torch
from torch.utils.data import Dataset
from neurg.l_utils.torch.to_tensor import to_device, to_tensor
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import load_pickle


class RdfDataSet(Dataset):
    def __init__(
        self,
        data_dir=os.path.join(PATH_ROOT, "data/baselines/rdf/data/raw_coll"),
        link_name="link0",
        stage="train",
        device="cpu",
        precision=32,
        normalize=True,
        pts_keys=None,
    ) -> None:
        super().__init__()

        if precision == 16:
            DTYPE = torch.float16
        else:
            DTYPE = torch.float32

        sub_dir = os.path.join(data_dir, stage)
        d_raw = np.load(f'{sub_dir}/voxel_128_{link_name}.npy', allow_pickle=True).item()
        pt_near = d_raw['near_points']
        gt_near = d_raw['near_sdf']
        pt_random = d_raw['random_points']
        gt_random = d_raw['random_sdf']
        scale = d_raw['scale']
        center = d_raw['center']

        data_all = {
            'near': np.concatenate([pt_near, gt_near[:, None]], axis=-1),
            'random': np.concatenate([pt_random, gt_random[:, None]], axis=-1),
        }
        if pts_keys is None:
            pts_keys = ['near', 'random']
        data = {k: data_all[k] for k in pts_keys}

        if normalize:
            for k in data:
                data[k][:, :3] = (data[k][:, :3] - center) / scale
                data[k][:, 3] = data[k][:, 3] / scale

        self.scale = float(scale)
        self.center = to_tensor(center, dtype=DTYPE, device=device)
        self.data = to_tensor(data, dtype=DTYPE, device=device)

        data_concat = torch.stack(list(self.data.values()), dim=1).reshape(-1, 4)
        self.x = data_concat[:, :3].contiguous()
        self.y = data_concat[:, [3]].contiguous()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret = (self.x[idx], self.y[idx])
        return ret

    def __getitems__(self, idx):
        return self.__getitem__(idx)

    def norm_x(self, x):
        return (x - self.center) / self.scale

    def unnorm_x(self, x):
        return (x * self.scale) + self.center

    def norm_y(self, y):
        return y / self.scale

    def unnorm_y(self, y):
        return y * self.scale

    def to(self, device):
        for k in ['scale', 'center', 'data', 'x', 'y']:
            setattr(self, k, to_device(getattr(self, k), device))


if __name__ == "__main__":
    ds = RdfDataSet()
