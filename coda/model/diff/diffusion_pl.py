from typing import Any, Dict
import numpy as np
from pathlib import Path
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from coda.utils.pylogger import Log
from einops import rearrange, einsum
from coda.configs import MainStore, builds


class DiffusionPL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        ignored_weights_prefix=["pipeline.smplx", "pipeline.endecoder", "pipeline.clip", "pipeline.vae"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

    def training_step(self, batch, batch_idx):
        # Forward and get loss
        batch["epoch_num"] = self.current_epoch
        outputs = self.pipeline.forward_train(batch)

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "sync_dist": True,
        }
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"train/{k}", v, **log_kwargs)

        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check

        # ROPE inference
        outputs = self.pipeline.forward_sample(batch)

        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        if not isinstance(ckpt_path, str):
            state_dict = self.pipeline.load_pretrained_model(ckpt_path)

        else:
            state_dict = torch.load(ckpt_path, "cpu")["state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


diffusion_pl = builds(
    DiffusionPL,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="diffusion_pl", node=diffusion_pl, group="model/diffusion")
