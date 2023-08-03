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
            decoder=False,
            vit_patch_size=16,
    ):
        super().__init__()
        if patch_size is None:
            patch_size = [1, 80, 80]
        self.save_hyperparameters(logger=False)        
        self.use_indices = use_indices

        self.net = televit.TeleViT(patch_size[0] * len(input_vars) + len(positional_vars),local_input_size=patch_size[1],patch_size=vit_patch_size,use_decoder=decoder,use_indices=use_indices)
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.val_auc = AUROC(pos_label=1, num_classes=2, compute_on_cpu=True)
        self.val_f1 = F1Score(compute_on_cpu=True)
        self.val_auprc = AveragePrecision(pos_label=1, num_classes=1, compute_on_cpu=True)
        self.test_auc = AUROC(pos_label=1, num_classes=2, compute_on_cpu=True)
        self.test_auprc = AveragePrecision(pos_label=1, num_classes=1, compute_on_cpu=True)
        self.test_f1 = F1Score()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x = batch['x_local']
        x_t = batch['x_oci']
        y = batch['y_local']
        x = x
        x_t = x_t
        y = y
        y = y.long()
        logits = self.net(x, x_indices=x_t)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)            
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics
        self.val_auc.update(preds, targets)
        self.val_auprc.update(preds.flatten(), targets.flatten())
        self.val_f1.update(preds.flatten(), targets.flatten())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)
        self.test_auc.update(preds, targets)
        self.test_auprc.update(preds.flatten(), targets.flatten())
        self.test_f1.update(preds.flatten(), targets.flatten())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/auroc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
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
