import os
from typing import Optional

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import _METRIC
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from neurg.l_utils.torch.util_dataset import SliceBatchLoader
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import load_pickle
from neurg.net.basic_model import mlp
from neurg.net.norm_point_sampler import NormPointSampler

# np.set_printoptions(suppress=True)

"""
class EmptyClass:
    pass


self = EmptyClass()
"""

"""
--------------------------------------------------
Data
-------------------------
"""


class LinkSdfDataset(Dataset):
    def __init__(
        self,
        data_dir=os.path.join(PATH_ROOT, "data/dataset/splited/links_231111_194431"),
        link_name="link0",
        stage="train",
        device="cpu",
        precision=32,
        normalize=True,
    ) -> None:
        super().__init__()

        sub_dir = os.path.join(data_dir, stage)
        data = load_pickle(sub_dir + f"/{link_name}.pkl")

        self.data = data
        self.bbox = data["bbox_largest"]
        self.extents = self.bbox[1] - self.bbox[0]
        self.bbox_center = self.bbox[0] + self.extents / 2.0

        pts_keys = [
            "pts_inside",
            "pts_outside",
            "pts_on_sur",
            "pts_near_sur",
            "pts_bbox_largest",
        ]
        data_all = []
        for k in pts_keys:
            data_all.append(data[k])
    
        k = "pts_bbox_pad_fixed"
        if k in data:
            data_all.append(data[k])

        data_all = np.concatenate(data_all)
        self.data_all = data_all

        if precision == 16:
            DTYPE = torch.float16
        else:
            DTYPE = torch.float32
        if normalize:
            self.x = torch.tensor(self.norm_x(data_all[:, :3]), device=device, dtype=DTYPE)
        else:
            self.x = torch.tensor(data_all[:, :3], device=device, dtype=DTYPE)
        self.y = torch.tensor(data_all[:, [3]], device=device, dtype=DTYPE)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        ret = (self.x[idx], self.y[idx])
        return ret

    def __getitems__(self, idx):
        return self.__getitem__(idx)

    def norm_x(self, x):
        return x - self.bbox_center

    def unnorm_x(self, x):
        return x + self.bbox_center

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)


class LinkSdfDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        link_name,
        shuffle=True,
        batch_size: int = 256,
        num_workers: int = 0,
        device="cpu",
        precision=32,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.link_name = link_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.precision = precision
        self.device = device

        self.load = False

    def setup(self, stage=None):
        if not self.load:
            if self.trainer:
                self.device = self.trainer.strategy.root_device

            self.ds_train = LinkSdfDataset(self.data_dir, self.link_name, "train", self.device, self.precision)
            self.ds_val = LinkSdfDataset(self.data_dir, self.link_name, "val", self.device, self.precision)
            self.ds_test = LinkSdfDataset(self.data_dir, self.link_name, "test", self.device, self.precision)
            self.load = True

    def train_dataloader(self):
        self.setup()
        return SliceBatchLoader(self.ds_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        self.setup()
        return SliceBatchLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        self.setup()
        return SliceBatchLoader(self.ds_test, batch_size=self.batch_size)


"""
--------------------------------------------------
Net Model
-------------------------
"""


class LinkSdfTorModel(nn.Module):
    def __init__(self, hid_dim=256, hid_size=3, actv_type="softplus", beta=100, bias=True):
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
            bias=bias,
        )

    def forward(self, x):
        y = self.mlp(x)
        return y

    def compute_gradient(self, x, y, retain_graph=True):
        d_points = torch.ones_like(y, requires_grad=False, device=x.device)
        grads = grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_points,
            create_graph=retain_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        return grads


class LinkSdfLModel(L.LightningModule):
    def __init__(
        self,
        hid_dim=32,
        hid_size=3,
        actv_type="softplus",
        bias=True,
        lr=2e-3,
        grad_max_norm=1.0,
        loss_type="mse",
        eikonal_weight=0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LinkSdfTorModel(hid_dim, hid_size, actv_type, bias=bias)

        # Need optimizer.zero_grad(), self.manual_backward(loss), optimizer.step()
        self.automatic_optimization = False
        # self.automatic_optimization = True
        self.lr = lr
        self.loss_type = loss_type
        self.eikonal_weight = eikonal_weight

        self.grad_max_norm = grad_max_norm

        self.test_stage_name = "test"

        self.sampler = NormPointSampler(1, 0.02)

    def update_test_stage_name(self, name):
        self.test_stage_name = name

    def forward(self, *args, **kwargs):
        y = self.model(*args, **kwargs)
        return y

    # def on_train_start(self):
    #     pass

    def log_static(
        self,
        name: str,
        value: _METRIC,
        prog_bar: bool = False,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        static=True,
    ) -> None:
        """log mean, min, max, std"""
        if static:
            if isinstance(value, torch.Tensor):
                if not len(value.shape) > 0:
                    static = False
            else:
                static = False
                print("warn:!!!! can not log static")

        if static:
            sep = "/"
            out_dict = {
                name + sep + "mean": value.mean(),
                name + sep + "std": value.std(),
                name + sep + "min": value.min(),
                name + sep + "max": value.max(),
            }

            for k, v in out_dict.items():
                reduce_fx = "mean"
                if k.split(sep)[-1] == "max":
                    reduce_fx = "max"
                elif k.split(sep)[-1] == "min":
                    reduce_fx = "min"
                self.log(f"{k}", v, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch, reduce_fx=reduce_fx)

        else:
            self.log(name, value, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch, reduce_fx=reduce_fx)

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = batch

        on_step = None
        if stage == "train":
            prog_bar = True
            retain_graph = True
        else:
            prog_bar = None
            retain_graph = False

        sampled_pts = self.sampler.get_points(x)

        with torch.enable_grad():
            x.grad = None
            x.requires_grad_()
            y_pred = self.model.forward(x)
            g = self.model.compute_gradient(x, y_pred, retain_graph=retain_graph)
            x.requires_grad_(False)

            sampled_pts.requires_grad_()
            sampled_pred = self.model.forward(sampled_pts)
            sampled_grad = self.model.compute_gradient(sampled_pts, sampled_pred, retain_graph=retain_graph)
            sampled_pts.requires_grad_(False)

            all_grad = torch.concat([g, sampled_grad], dim=0)

        if self.loss_type == "mse":
            sdf_mse_loss = F.mse_loss(y, y_pred, reduction="mean")

            eikonal_diff = torch.linalg.norm(all_grad, dim=-1) - 1
            eikonal_mse_loss = torch.square(eikonal_diff).mean()

            loss = sdf_mse_loss + eikonal_mse_loss * self.eikonal_weight

            if stage != "train" or (stage == "train" and self.trainer._logger_connector.should_update_logs):
                with torch.no_grad():
                    l1_loss = F.l1_loss(y, y_pred, reduction="none")
                    eikonal_l1_loss = torch.abs(eikonal_diff).mean()

                    self.log(f"{stage}/mse_loss", sdf_mse_loss, on_step=on_step, prog_bar=False)
                    self.log_static(f"{stage}/l1_loss", l1_loss, on_step=on_step, prog_bar=False)
                    self.log(f"{stage}/eikonal/mse_loss", eikonal_mse_loss, on_step=on_step, prog_bar=False)
                    self.log(f"{stage}/eikonal/l1_loss", eikonal_l1_loss, on_step=on_step, prog_bar=False)

        elif self.loss_type == "l1":
            l1_loss = F.l1_loss(y, y_pred, reduction="none")
            loss = torch.mean(l1_loss)
            eikonal_diff = torch.linalg.norm(all_grad, dim=-1) - 1
            eikonal_l1_loss = torch.abs(eikonal_diff).mean()
            if self.eikonal_weight != 0:
                loss += eikonal_l1_loss * self.eikonal_weight
            with torch.no_grad():
                self.log_static(f"{stage}/l1_loss", l1_loss, on_step=on_step, prog_bar=False)
                self.log(f"{stage}/eikonal/l1_loss", eikonal_l1_loss, on_step=on_step, prog_bar=False)
        else:
            print(f"loss_type error: loss_type={self.loss_type}")
            exit(-1)

        self.log(f"{stage}/loss", loss, on_step=on_step, prog_bar=prog_bar)

        return x, y, loss

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()

        # ----------
        _, _, loss = self._common_step(batch, batch_idx, "train")
        # ----------

        if not self.automatic_optimization:
            self.manual_backward(loss)

            # clip gradients
            if self.grad_max_norm is not None:
                self.clip_gradients(opt, gradient_clip_val=self.grad_max_norm, gradient_clip_algorithm="norm")
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max_norm)

            opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, self.test_stage_name)

    def predict_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pred = self(x)
        return y_pred

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, ReduceLROnPlateau):
            if not self.automatic_optimization:
                sch.step(self.trainer.callback_metrics["train/loss"])

    # def validation_epoch_end(self, outputs):
    #     if not self.automatic_optimization:
    #         sch = self.lr_schedulers()
    #         sch.step(self.trainer.callback_metrics["train/loss"])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = ReduceLROnPlateau(opt, patience=100 * 5, factor=0.5, min_lr=1e-6)
        ret = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                # "monitor": "train/loss",
                # "frequency": 1
            },
        }
        return ret


if __name__ == "__main__":
    from neurg.l_main import cli_main

    cli_main()
    # model = LinkSdfModel.load_from_checkpoint()
    # base_config = '--config=config/soft_ssdf_cls_auto.yaml'
    # if len(sys.argv) == 1:
    #     sys.argv += [base_config]
    # if inDebug():
    #     sys.argv += ['--logger.name=debug', '--max_epochs=100']
    # print(f"sys.argv: {sys.argv}")
    # cli_fit_test(test=True)
