from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from .components import televit
import wandb
import einops


class plUNET(pl.LightningModule):
    def __init__(
            self,
            input_vars: list = None,
            positional_vars: list = None,
            patch_size: list = None,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss='dice',
            encoder='efficientnet-b5',
            use_indices=False,
            use_global_input=False,
            sea_masked=False,
            vit_patch_size=16,
            decoder=False,
            global_patch_size=None,
    ):
        super().__init__()
        self.sea_masked = sea_masked
        if patch_size is None:
            patch_size = [1, 80, 80]
        self.save_hyperparameters(logger=False)

        if global_patch_size is None:
            print('Global patch size is None')
            exit(2)
        
        self.net = televit.TeleViT(patch_size[0] * len(input_vars) + len(positional_vars), use_indices=use_indices, use_global_input=use_global_input,local_input_size=patch_size[1], global_patch_size=global_patch_size)

        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            if self.sea_masked:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=2)
            else:
                self.criterion = torch.nn.CrossEntropyLoss()

        if self.sea_masked:
            self.val_auprc = AveragePrecision(num_classes=1, pos_label=1, task="binary", ignore_index=2)
            self.test_auprc = AveragePrecision(num_classes=1, pos_label=1, task="binary", ignore_index=2)
        else:
            self.val_auprc = AveragePrecision(
                pos_label=1, num_classes=1)
            self.test_auprc = AveragePrecision(
                pos_label=1, num_classes=1)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x_local = batch['x_local']
        x_local_mask = batch['x_local_mask']
        x_oci = batch['x_oci']
        x_global = batch['x_global']
        y_local = batch['y_local']
        y_global = batch['y_global']
        x = x_local
        y = y_local
        # if this is the first batch
        if self.global_step == 0:
            # print the shapes of the inputs and outputs
            print(f'x_local shape: {x_local.shape}')
            print(f'x_oci shape: {x_oci.shape}')
            print(f'x_global shape: {x_global.shape}')
            print(f'y_local shape: {y_local.shape}')
            print(f'y_global shape: {y_global.shape}')

        y = y.long()
        # make x, x_t, x_global into torch.cuda.FloatTensor
        x = x.float()
        x_oci = x_oci.float()
        x_global = x_global.float()

        logits = self.net(x, x_indices=x_oci, x_global=x_global)
        if self.global_step == 0:
            print('logits shape: ', logits.shape)
        if self.sea_masked:
            y[x_local_mask == 1] = 2

        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        preds[x_local_mask] = 0
        if self.global_step == 0:
            print('preds shape: ', preds.shape)
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics
        self.val_auprc.update(preds.flatten(), targets.flatten())
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        self.test_auprc.update(preds, targets)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}
