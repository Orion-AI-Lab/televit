import glob
import os
from typing import List

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import f1_score, precision_score, recall_score
import gc
import numpy as np
import pytorch_lightning as pl


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class LogValPredictionsSegmentation(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, every_n_epochs: int = 10):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.preds = []
        self.targets = []
        self.inputs = []
        self.counts_true = []
        self.counts_pred = []
        self.every_n_epochs = every_n_epochs
        if every_n_epochs <= 0:
            raise ValueError('every_n_epochs must be > 0')

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready and len(self.preds) < self.num_samples:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])
            self.inputs.append(outputs["inputs"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            preds = torch.cat(self.preds[:self.num_samples])
            targets = torch.cat(self.targets[:self.num_samples])
            inputs = torch.cat(self.inputs[:self.num_samples])

            imgs = []
            input_vars = pl_module.hparams.input_vars
            for i in range(self.num_samples):
                fig, axs = plt.subplots(1, len(input_vars) + 2, figsize=(20, 4))
                for var_idx in range(len(input_vars)):
                    im = axs[var_idx].imshow(inputs[i][var_idx].detach().cpu().numpy().squeeze())
                    axs[var_idx].set_title(input_vars[var_idx])
                im = axs[-2].imshow(preds[i].detach().cpu().numpy().squeeze())
                axs[-2].set_title('Prediction')
                im = axs[-1].imshow(targets[i].detach().cpu().numpy().squeeze())
                axs[-1].set_title('Target')
                imgs.append(wandb.Image(fig, caption=f'Val {i}'))

            # log the images as wandb Image
            experiment.log(
                {
                    "val/target-prediction maps": imgs
                }
            )


            fig, ax = plt.subplots(figsize=(10, 10))
            sns.scatterplot(x=targets.flatten().detach().cpu().numpy(),
                            y=preds.flatten().detach().cpu().numpy(),
                            ax=ax, alpha=0.1)
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            experiment.log(
                {
                    "val/target-prediction scatter": wandb.Image(fig)
                }
            )

            # clear the lists
            self.preds = []
            self.targets = []
            self.inputs = []
            self.preds.clear()
            self.targets.clear()
            self.inputs.clear()


class AddMetricAggs(Callback):
    """Defines aggregation for pre-defined metrics
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, metrics, aggs):
        super().__init__()
        if len(metrics) != len(aggs):
            raise ValueError('Lists of metrics and aggregations should have the same length.')
        self.metrics = metrics
        self.aggs = aggs

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger = get_wandb_logger(trainer=trainer)
        for metric, agg in zip(self.metrics, self.aggs):
            # define a metric we are interested in the minimum of
            logger.experiment.define_metric(metric, summary=agg)
