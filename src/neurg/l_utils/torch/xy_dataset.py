import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from lightning.fabric.utilities import move_data_to_device


class XYDataset(Dataset):
    def __init__(self, x, y, device="cpu", precision=32) -> None:
        super().__init__()
        if precision == 16:
            DTYPE = torch.float16
        else:
            DTYPE = torch.float32

        if isinstance(x, torch.Tensor):
            self.x = x.to(device, dtype=DTYPE)
        else:
            self.x = torch.tensor(x, device=device, dtype=DTYPE)
        if isinstance(y, torch.Tensor):
            self.y = y.to(device, dtype=DTYPE)
        else:
            self.y = torch.tensor(y, device=device, dtype=DTYPE)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret = (self.x[idx], self.y[idx])
        return ret

    def __getitems__(self, idx):
        return self.__getitem__(idx)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)


def xy_predict(func, dl, device="cpu", pbar=None):
    xs = []
    ys = []
    preds = []
    with torch.no_grad():
        if pbar:
            dl = tqdm(dl, desc="xy_predict")
        for x, y in dl:
            x = x.to(device)
            pred = func(x)
            xs.append(x.cpu())
            ys.append(y.cpu())
            preds.append(pred.cpu())
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    preds = torch.cat(preds, dim=0)
    return xs, ys, preds


def batch_predict(func, dl, device="cpu", pbar=None):
    preds = []
    with torch.no_grad():
        if pbar:
            dl = tqdm(dl, desc="batch_predict")
        for x in dl:
            x = x.to(device)
            pred = func(x)
            preds.append(move_data_to_device(pred, "cpu"))
    if len(preds) == 1:
        return preds[0]
    elif len(preds) > 1:
        item = preds[0]
        if isinstance(item, torch.Tensor):
            return torch.cat(preds, dim=0)
        elif isinstance(item, dict):
            return {k: torch.cat([pred[k] for pred in preds], dim=0) for k in item.keys()}
        elif isinstance(item, list) or isinstance(item, tuple):
            return tuple(torch.cat([pred[i] for pred in preds], dim=0) for i in range(len(item)))
    return preds


if __name__ == '__main__':

    def func(x):
        return (x, torch.cat([x, x], dim=1))

    x = torch.rand(10, 2, 1)
    y = batch_predict(func, x)
    if isinstance(y, tuple) or isinstance(y, list):
        for i in range(len(y)):
            print(f"y[{i}]: {y[i].shape}")
    if isinstance(y, torch.Tensor):
        print(f"y: {y.shape}")

    def func(x):
        return x

    y = batch_predict(func, x)
    if isinstance(y, tuple) or isinstance(y, list):
        for i in range(len(y)):
            print(f"y[{i}]: {y[i].shape}")
    if isinstance(y, torch.Tensor):
        print(f"y: {y.shape}")
    print('done')
