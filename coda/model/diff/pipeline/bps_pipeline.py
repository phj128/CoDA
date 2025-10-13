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
from diffusers.schedulers import DDPMScheduler
from pytorch_lightning.utilities import rank_zero_only

from coda.model.diff.utils.endecoder import EnDecoder
from coda.model.diff.pipeline.mini_pipeline import randomly_set_null_condition
from coda.model.diff.pipeline.text_pipeline import Pipeline as TextPipeline, randomly_set_null_text

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from coda.utils.smplx_utils import make_smplx


class Pipeline(TextPipeline):
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
        seq_mask = model_output["mask"]  # (B, L)

        # ========== Compute Loss ========== #
        prediction_type = scheduler.config.prediction_type
        if prediction_type == "sample":
            target = x
        else:
            assert prediction_type == "epsilon"
            target = noise

        mask = inputs["mask"] * seq_mask[..., None]  # (B, L, C)

        M = inputs["basis_finger_point"].shape[-2]
        L = inputs["bps"].shape[-2]

        dist_pred = model_pred[..., :-20]
        contact_pred = model_pred[..., -20:-10]
        vert_pred = model_pred[..., -10:]
        dist_target = target[..., :-20]
        contact_target = target[..., -20:-10]
        vert_target = target[..., -10:]
        dist_mask = mask[..., :-20]
        contact_mask = mask[..., -20:-10]
        vert_mask = mask[..., -10:]

        dist_pred = dist_pred.reshape(B, L, M, -1) * dist_mask[..., None, :]
        dist_target = dist_target.reshape(B, L, M, -1) * dist_mask[..., None, :]
        contact_pred = contact_pred * contact_mask
        contact_target = contact_target * contact_mask
        vert_pred = vert_pred * vert_mask
        vert_target = vert_target * vert_mask

        loss_sum = 0.0
        wrist_dist_loss = F.l1_loss(dist_pred[..., :2].float(), dist_target[..., :2].float(), reduction="mean")
        loss_sum += wrist_dist_loss
        figner_dist_loss = F.l1_loss(dist_pred[..., 2:].float(), dist_target[..., 2:].float(), reduction="mean")
        loss_sum += figner_dist_loss
        contact_loss = F.l1_loss(contact_pred.float(), contact_target.float(), reduction="mean")
        loss_sum += contact_loss
        vert_loss = F.l1_loss(vert_pred.float(), vert_target.float(), reduction="mean")
        loss_sum += vert_loss

        outputs["loss"] = loss_sum
        outputs["wrist_dist_loss"] = wrist_dist_loss
        outputs["finger_dist_loss"] = figner_dist_loss
        outputs["contact_loss"] = contact_loss
        outputs["vert_loss"] = vert_loss
        return outputs

    # ========== Sample ========== #
    def forward_sample(self, inputs):
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        length = inputs["length"].max()
        B = inputs["length"].shape[0]
        gt_x = self.endecoder.encode(inputs)
        gt_output = self.endecoder.decode(gt_x, inputs=inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        x = torch.randn((B, length, self.denoiser3d.output_dim), device=gt_x.device)

        # 2. Conditions
        condition_dict = self.endecoder.encode_condition(inputs)
        f_condition = condition_dict

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
