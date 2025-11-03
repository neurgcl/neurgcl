import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from neurg.my_utils.config import PATH_ROOT


class JData:
    def __init__(
        self,
        device="cpu",
        precision=32,
        data_dir="data/baselines/jsdf/data_mesh_roscol.mat",
    ) -> None:
        # data = loadmat(PATH_ROOT + '/data/JSDF_paper/data_mesh_test_paper.mat')['dataset']
        if os.path.isabs(data_dir):
            data = loadmat(data_dir)["dataset"]
        else:
            data = loadmat(PATH_ROOT + "/" + data_dir)["dataset"]
        self.device = device

        L1 = 0
        L2 = int(0.95 * data.shape[0])
        print(L1, L2)
        n_size = L2
        train_ratio = 0.98
        test_ratio = 0.01
        val_ratio = 1 - train_ratio - test_ratio

        idx_train = np.arange(0, int(n_size * train_ratio))
        idx_val = np.arange(idx_train[-1] + 1, int(n_size * (train_ratio + test_ratio)))
        idx_test = np.arange(idx_val[-1] + 1, int(n_size))
        # idx_test = np.arange(idx_val[-1] + 1, int(data.shape[0]))

        if precision == 16:
            DTYPE = torch.float16
        else:
            DTYPE = torch.float32

        self.x = torch.tensor(data[:, 0:10], device=self.device, dtype=DTYPE)
        self.y = 100 * torch.tensor(data[:, 10:], device=self.device, dtype=DTYPE)

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)


class JsdfDataset(Dataset):
    """ """

    def __init__(self, jdata, stage) -> None:
        super().__init__()
        x = jdata.x
        y = jdata.y

        idx_train = jdata.idx_train
        idx_val = jdata.idx_val
        idx_test = jdata.idx_test

        if stage == "train":
            self.x = x[idx_train, :]
            self.y = y[idx_train, :]
        elif stage == "val":
            self.x = x[idx_val, :]
            self.y = y[idx_val, :]
        elif stage == "test":
            self.x = x[idx_test, :]
            self.y = y[idx_test, :]
        else:
            raise ValueError(f"stage: {stage} not recognized")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret = (self.x[idx], self.y[idx])
        return ret

    def __getitems__(self, idx):
        ret = (self.x[idx], self.y[idx])
        return ret
