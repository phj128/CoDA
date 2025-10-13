import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import inspect
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from coda.utils.pylogger import Log
from coda.utils.net_utils import gaussian_smooth
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from pytorch_lightning.utilities import rank_zero_only

from coda.model.diff.utils.endecoder import EnDecoder

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from coda.model.diff.pipeline.mini_pipeline import Pipeline as MiniPipeline
from coda.model.diff.pipeline.mini_pipeline import randomly_set_null_condition
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from coda.utils.smplx_utils import make_smplx


class Pipeline(MiniPipeline):
    def __init__(self, args, args_denoiser3d, args_clip, **kwargs):
        super().__init__(args, args_denoiser3d, **kwargs)
        self.clip = instantiate(args_clip, _recursive_=False)

        # ----- Freeze ----- #
        self.freeze_clip()

    def freeze_clip(self):
        self.clip.eval()
        self.clip.requires_grad_(False)

    # ========== Training ========== #
    def forward_train(self, inputs):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample
        B = length.size(0)
        scheduler = self.train_scheduler

        condition_dict = self.endecoder.encode_condition(inputs)
        x = self.endecoder.encode(inputs)

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # *. Conditions
        f_condition = condition_dict
        f_condition = randomly_set_null_condition(f_condition, 0.1)

        # Encode CLIP embedding
        assert self.training
        text = inputs["caption"]
        text = randomly_set_null_text(text, 0.1)
        clip_text = self.clip.encode_text(text, enable_cfg=False, with_projection=True)  # (B, D)
        f_condition["f_text"] = clip_text.f_text

        model_kwargs = self.build_model_kwargs(noisy_x, t, inputs, f_condition, enable_cfg=False)

        # Forward & output
        model_output = self.denoiser3d(**model_kwargs)
        model_pred = model_output["sample"]  # (B, L, C)
        if "mask" in model_output:
            seq_mask = model_output["mask"]  # (B, L)
        else:
            seq_mask = None

        # ========== Compute Loss ========== #
        prediction_type = scheduler.config.prediction_type
        if prediction_type == "sample":
            target = x
        else:
            assert prediction_type == "epsilon"
            target = noise

        if seq_mask is not None:
            mask = seq_mask[..., None]  # (B, L, C)
            model_pred = model_pred * mask
            target = target * mask
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        outputs["loss"] = loss
        return outputs

    # ========== Sample ========== #
    def forward_sample(self, inputs):
        # return {}
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        max_length = inputs["length"].max()
        try:
            max_length = max(inputs["transl"].shape[1], max_length)
        except:
            pass
        B = inputs["length"].shape[0]
        gt_x = self.endecoder._encode(inputs)
        gt_output = self.endecoder._decode(gt_x, inputs=inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        x = torch.randn((B, max_length, self.denoiser3d.output_dim), device=gt_x.device)

        # 2. Conditions
        condition_dict = self.endecoder.encode_condition(inputs)
        f_condition = condition_dict

        # Encode CLIP embedding
        text = inputs["caption"]

        if not enable_cfg:
            text = ["" for _ in range(len(text))]

        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg, with_projection=True)  # (B, D)
        f_condition["f_text"] = clip_text.f_text

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler)  # for scheduler.step()
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                inputs=inputs,
                f_condition=f_condition,
                enable_cfg=enable_cfg,
            )
            x0_, _ = self.cfg_denoise_func(self.denoiser3d, model_kwargs, scheduler, enable_cfg)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        outputs = self.endecoder.decode(x, inputs=inputs)
        outputs["gt_output"] = gt_output
        return outputs


def randomly_set_null_text(text, uncond_prob=0.1):
    """
    Args:
        text: List of str
    """
    # To support classifier-free guidance, randomly set-to-unconditioned
    B = len(text)
    # text
    text_mask = torch.rand(B) < uncond_prob
    text_ = ["" if m else t for m, t in zip(text_mask, text)]

    return text_
