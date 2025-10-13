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

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from coda.utils.smplx_utils import make_smplx
import coda.utils.matrix as matrix


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args

        self.train_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.test_scheduler = instantiate(args.scheduler_opt_sample)
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps

        # Networks
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        Log.info(self.denoiser3d)
        self.build_endecoder(args)
        self.load_vae(args)
        self.endecoder.set_vae(self.vae)

    def build_endecoder(self, args):
        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)

    def load_vae(self, args):
        if "vae_args" in args:
            self.vae = instantiate(args.vae_args, _recursive_=False)
            vae_path = args.get("vae_path", None)
            if vae_path is None:
                return
            vae_state_dict = torch.load(args.vae_path, map_location="cpu")["state_dict"]
            print(f"Load VAE from {args.vae_path}")
            filter_state_dict = {}
            prefix = "pipeline.denoiser3d."
            for k, v in vae_state_dict.items():
                filter_state_dict[k[len(prefix) :]] = v
            self.vae.load_state_dict(filter_state_dict, strict=True)
            self.vae.eval()
            self.vae.requires_grad_(False)
        else:
            self.vae = None

    @staticmethod
    def build_model_kwargs(x, timesteps, inputs, f_condition, enable_cfg, **kwargs):
        """override this if you want to add more kwargs"""
        # supermotion/decoder_multiple_crossattn
        length = torch.cat([inputs["length"]] * 2) if enable_cfg else inputs["length"]
        if enable_cfg:
            new_f_condition = {}
            for k in f_condition.keys():
                if k == "f_text":
                    new_f_condition[k] = f_condition[k].clone()
                elif "trajmat" in k:
                    identity_mat = matrix.identity_mat(f_condition[k], device=x.device)
                    new_f_condition[k] = torch.cat([identity_mat, f_condition[k]], dim=0)
                else:
                    new_f_condition[k] = torch.cat([torch.zeros_like(f_condition[k]), f_condition[k]], dim=0)
            f_condition = new_f_condition

        model_kwargs = dict(
            x=x,
            timesteps=timesteps,
            length=length,
        )
        model_kwargs.update(f_condition)
        return model_kwargs

    # ========== Training ========== #
    def forward_train(self, inputs):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample
        B = length.size(0)
        scheduler = self.train_scheduler

        condition_dict = self.endecoder.encode_condition(inputs)
        x = self.endecoder.encode(inputs)

        ####### debug vae output #######
        # out_x = self.endecoder.decode(x, inputs=inputs)
        # pos = out_x["tip_global_pos"]

        # for i in range(16):
        #     wis3d = make_wis3d(name=f"debug_endecoder_{i:03d}")
        #     add_motion_as_lines(pos[i].float(), wis3d, name=f"output_hand_tip", skeleton_type="handtip", radius=0.005)
        # raise NotImplementedError
        #################################

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # *. Conditions
        f_condition = condition_dict
        f_condition = randomly_set_null_condition(f_condition, 0.1)

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

        if "mask" in inputs:
            mask = inputs["mask"]
            if seq_mask is not None:
                mask = mask * seq_mask[..., None]  # (B, L, C)
        else:
            if seq_mask is not None:
                mask = seq_mask[..., None]  # (B, L, C)
            else:
                mask = None

        if mask is not None:
            model_pred = model_pred * mask
            target = target * mask
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        outputs["loss"] = loss
        return outputs

    def cfg_denoise_func(self, denoiser, model_kwargs, scheduler, enable_cfg):
        x = model_kwargs.pop("x")
        t = model_kwargs.pop("timesteps")

        # expand the x if we are doing classifier free guidance
        x_model_input = torch.cat([x] * 2) if enable_cfg else x
        x_model_input = scheduler.scale_model_input(x_model_input, t)

        # predict
        denoiser_out = denoiser(x_model_input, t, **model_kwargs)
        noise_pred = denoiser_out["sample"]

        # classifier-free guidance
        if enable_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # a special case since our motion prior is predicting x0 # TODO: extra check may be needed
        x0_ = noise_pred
        return x0_, denoiser_out

    # ========== Sample ========== #
    @staticmethod
    def prepare_extra_step_kwargs(scheduler, eta=0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # # check if the scheduler accepts generator
        # accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        # if accepts_generator:
        #     extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_prog_bar(self, total, **kwargs):
        return ProgBarWrapper(total=total, leave=False, bar_format="{l_bar}{bar:10}{r_bar}", **kwargs)

    # ========== Sample ========== #
    def forward_sample(self, inputs):
        # return {}
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        length = inputs["length"].max()
        B = inputs["length"].shape[0]
        gt_x = self.endecoder._encode(inputs)
        gt_output = self.endecoder._decode(gt_x, inputs=inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        if self.vae is None:
            x = torch.randn((B, length, self.denoiser3d.output_dim), device=gt_x.device)
        else:
            x = torch.randn((B, self.denoiser3d.latent_size, self.denoiser3d.latent_dim), device=gt_x.device)

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


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2]) < uncond_prob
        if "trajmat" in k:
            identity_mat = matrix.identity_mat(f_condition[k], device=f_condition[k].device)
            f_condition[k][mask] = identity_mat[mask]
        else:
            f_condition[k][mask] = 0.0
    return f_condition


class ProgBarWrapper:
    def __init__(self, **kwargs):
        self.prog_bar = self.get_prog_bar(**kwargs)

    @rank_zero_only
    def get_prog_bar(self, **kwargs):
        return tqdm(**kwargs)

    @rank_zero_only
    def update(self, **kwargs):
        self.prog_bar.update(**kwargs)
