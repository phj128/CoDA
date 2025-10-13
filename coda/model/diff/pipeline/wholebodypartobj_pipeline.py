import os
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
from diffusers import StableDiffusionPipeline
from coda.utils.grab.object_model import ObjectModel
from coda.utils.grab.mesh import Mesh
from pytorch_lightning.utilities import rank_zero_only
from coda.dataset.arctic.utils import get_wrist_trajectory, get_finger_trajectory

from coda.model.diff.utils.endecoder import EnDecoder
from coda.utils.bps import bps_gen_ball_inside, calculate_bps, get_pc_center, get_pc_scale

from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from coda.model.diff.pipeline.mini_pipeline import Pipeline as MiniPipeline
from coda.model.diff.pipeline.text_pipeline import Pipeline as TextPipeline
from coda.model.diff.pipeline.mini_pipeline import randomly_set_null_condition
from coda.model.diff.utils.optimization import (
    WholeBodyPartCondKeyLocationsLoss,
    warmup_scheduler,
    cosine_decay_scheduler,
    noise_regularize_1d,
)
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from coda.utils.smplx_utils import make_smplx
from coda.utils.arctic.object_tensors import ObjectTensors
import coda.utils.matrix as matrix


class Pipeline(TextPipeline):
    def __init__(
        self,
        args,
        args_denoiser3d,
        args_denoiser3d_lefthand,
        args_denoiser3d_righthand,
        args_denoiser3d_invtraj,
        args_denoiser3d_obj,
        args_clip,
        args_diffusion_optimization,
        args_traj_optimization,
        **kwargs,
    ):
        super().__init__(args, args_denoiser3d, args_clip, **kwargs)
        self.args_diffusion_optimization = args_diffusion_optimization
        self.args_traj_optimization = args_traj_optimization
        self.num_optim_ddim_steps = args_diffusion_optimization.num_ddim_steps
        self.args_denoiser3d_obj = args_denoiser3d_obj
        self.inverse_scheduler = instantiate(args.scheduler_opt_inverse)
        # inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
        self.endecoder.is_to_ayfz = False

        # Networks
        self.denoiser3d_lefthand = instantiate(args_denoiser3d_lefthand, _recursive_=False)
        self.denoiser3d_righthand = instantiate(args_denoiser3d_righthand, _recursive_=False)
        self.denoiser3d_invtraj = instantiate(args_denoiser3d_invtraj, _recursive_=False)
        self.denoiser3d_obj = instantiate(args_denoiser3d_obj, _recursive_=False)

        # Normalizer
        self.endecoder_lefthand: EnDecoder = instantiate(args.lefthand_endecoder_opt, _recursive_=False)
        self.endecoder_righthand: EnDecoder = instantiate(args.righthand_endecoder_opt, _recursive_=False)
        self.endecoder_invtraj: EnDecoder = instantiate(args.invtraj_endecoder_opt, _recursive_=False)
        self.endecoder_obj: EnDecoder = instantiate(args.obj_endecoder_opt, _recursive_=False)

        self.object_tensor = ObjectTensors().cuda()
        self.smplx = make_smplx(type="wholebody")

        self.vis_i = 0

    def set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr

    @torch.enable_grad()
    def diffusion_optim(
        self, inputs, invtraj_outputs, obj_verts, obj_normals, is_vis=False, is_moving_obj=False, moving_obj_kwargs=None
    ):
        self.optim_noise_body = inputs["init_noise_body"].clone().detach().requires_grad_()
        self.optim_noise_lefthand = inputs["init_noise_lefthand"].clone().detach().requires_grad_()
        self.optim_noise_righthand = inputs["init_noise_righthand"].clone().detach().requires_grad_()
        self.start_noise_body = self.optim_noise_body.clone().detach()
        self.start_noise_lefthand = self.optim_noise_lefthand.clone().detach()
        self.start_noise_righthand = self.optim_noise_righthand.clone().detach()

        B, _, _ = self.optim_noise_body.shape
        motion_length = inputs["length"]
        L = motion_length.max()
        J = 62
        target = torch.zeros((B, L, J, 3), dtype=torch.float32, device=self.optim_noise_body.device)
        select_ind = [20, 21]
        left_hand_ind = [25, 29, 33, 37, 41]
        right_hand_ind = [45, 49, 53, 57, 61]
        hand_localind = [4, 8, 12, 16, 20]
        target_mask = torch.zeros_like(target, dtype=torch.float32, device=self.optim_noise_body.device)
        wrist_traj = invtraj_outputs["wrist_traj_global"]
        finger_traj = invtraj_outputs["finger_traj_global"]
        left_finger_traj = finger_traj[..., :5, :]
        right_finger_traj = finger_traj[..., 5:, :]
        wrist_mask = invtraj_outputs["wrist_mask"]
        contact_mask = invtraj_outputs["contact_mask"]
        left_finger_mask = contact_mask[..., :5]
        right_finger_mask = contact_mask[..., 5:]

        target[..., select_ind, :] = wrist_traj
        target[..., left_hand_ind, :] = left_finger_traj
        target[..., right_hand_ind, :] = right_finger_traj
        target_mask[..., select_ind, :] = wrist_mask[..., None].float()
        target_mask[..., left_hand_ind, :] = left_finger_mask[..., None].float()
        target_mask[..., right_hand_ind, :] = right_finger_mask[..., None].float()

        # first frame
        target[:, 0, :22, :] = inputs["gt_body_global_pos"][:, 0]
        target[:, 0, 22:42] = inputs["gt_lefthand_global_pos"][:, 0, 1:]
        target[:, 0, 42:] = inputs["gt_righthand_global_pos"][:, 0, 1:]
        target_mask[:, 0] = 1.0
        target[:, :, 0, :] = target[:, 0, 0, :]

        body_endecoder = self.endecoder
        lefthand_endecoder = self.endecoder_lefthand
        righthand_endecoder = self.endecoder_righthand
        loss_fn = WholeBodyPartCondKeyLocationsLoss(
            target=target,
            target_mask=target_mask,
            motion_length=motion_length,
            body_inv_transform=body_endecoder.decode,
            lefthand_inv_transform=lefthand_endecoder.decode,
            righthand_inv_transform=righthand_endecoder.decode,
            use_mse_loss=False,
            only_body_steps=self.args_diffusion_optimization.only_body_steps,
            no_body_steps=self.args_diffusion_optimization.no_body_steps,
            no_pene_steps=self.args_diffusion_optimization.no_pene_steps,
            body_w=self.args_diffusion_optimization.body_w,
            hand_w=self.args_diffusion_optimization.hand_w,
            detach_hand_w=self.args_diffusion_optimization.detach_hand_w,
            delta_w=self.args_diffusion_optimization.delta_w,
            sdf_w=self.args_diffusion_optimization.sdf_w,
            root_height_reg_w=self.args_diffusion_optimization.root_height_reg_w,
            foot_sliding_reg_w=self.args_diffusion_optimization.foot_sliding_reg_w,
            foot_floating_reg_w=self.args_diffusion_optimization.foot_floating_reg_w,
            is_vis=is_vis,
            is_moving_obj=is_moving_obj,
            moving_obj_kwargs=moving_obj_kwargs,
        )
        criterion = lambda body_x, lefthand_x, righthand_x, iter: loss_fn(
            body_x,
            lefthand_x,
            righthand_x,
            left_hand_ind,
            right_hand_ind,
            obj_verts,
            obj_normals,
            self.smplx,
            inputs,
            iter,
            self.vis_i,
        )

        num_steps = self.args_diffusion_optimization.num_opt_steps
        self.optimizer = torch.optim.Adam(
            [self.optim_noise_body, self.optim_noise_lefthand, self.optim_noise_righthand],
            lr=self.args_diffusion_optimization.lr,
        )
        self.lr_scheduler = []
        if self.args_diffusion_optimization.lr_warm_up_steps > 0:
            self.lr_scheduler.append(
                lambda step: warmup_scheduler(step, self.args_diffusion_optimization.lr_warm_up_steps)
            )
        self.lr_scheduler.append(
            lambda step: cosine_decay_scheduler(
                step, self.args_diffusion_optimization.lr_decay_steps, num_steps, decay_first=False
            )
        )
        self.step_count = 0

        batch_size = self.optim_noise_body.shape[0]
        prog_bar = self.get_prog_bar(num_steps, desc="Diffusion Optimization")
        for i in range(num_steps):
            info = {"step": [i] * batch_size}

            # learning rate scheduler
            lr_frac = 1
            if len(self.lr_scheduler) > 0:
                for scheduler in self.lr_scheduler:
                    lr_frac *= scheduler(self.step_count)
                self.set_lr(self.args_diffusion_optimization.lr * lr_frac)
            info["lr"] = [self.args_diffusion_optimization.lr * lr_frac] * batch_size

            # criterion
            pred_sample_body = body_ddim_loop_with_gradient(self.optim_noise_body, self, inputs)
            pred_sample_lefthand = hand_ddim_loop_with_gradient(self.optim_noise_lefthand, self, inputs, is_right=False)
            pred_sample_righthand = hand_ddim_loop_with_gradient(
                self.optim_noise_righthand, self, inputs, is_right=True
            )
            loss, loss_dict = criterion(pred_sample_body, pred_sample_lefthand, pred_sample_righthand, i)
            info["loss"] = loss.detach().cpu()
            loss = loss.sum()
            info.update(loss_dict)

            # diff penalty
            if self.args_diffusion_optimization.diff_penalty_scale > 0:
                # [batch_size,]
                body_loss_diff = (self.optim_noise_body - self.start_noise_body).norm(p=2, dim=[1, 2])
                lefthand_loss_diff = (self.optim_noise_lefthand - self.start_noise_lefthand).norm(p=2, dim=[1, 2])
                righthand_loss_diff = (self.optim_noise_righthand - self.start_noise_righthand).norm(p=2, dim=[1, 2])
                loss += (
                    self.args_diffusion_optimization.diff_penalty_scale
                    * (body_loss_diff + lefthand_loss_diff + righthand_loss_diff).sum()
                )
                info["loss_diff"] = (body_loss_diff + lefthand_loss_diff + righthand_loss_diff).detach().cpu()
            else:
                info["loss_diff"] = [0] * batch_size

            # decorrelate
            body_loss_decorrelate = noise_regularize_1d(
                self.optim_noise_body.permute(0, 2, 1)[..., None, :],
            )
            lefthand_loss_decorrelate = noise_regularize_1d(
                self.optim_noise_lefthand.permute(0, 2, 1)[..., None, :],
            )
            righthand_loss_decorrelate = noise_regularize_1d(
                self.optim_noise_righthand.permute(0, 2, 1)[..., None, :],
            )

            if i < self.args_diffusion_optimization.only_body_steps:
                lefthand_loss_decorrelate *= 0.0
                righthand_loss_decorrelate *= 0.0

            decorrelate_loss = (
                self.args_diffusion_optimization.decorrelate_scale
                * (body_loss_decorrelate + lefthand_loss_decorrelate + righthand_loss_decorrelate).sum()
            )
            if self.args_diffusion_optimization.decorrelate_scale > 0:
                loss += decorrelate_loss
                info["loss_decorrelate"] = decorrelate_loss.detach().cpu()
            else:
                loss += decorrelate_loss * 0.0
                info["loss_decorrelate"] = decorrelate_loss.detach().cpu() * 0.0

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # grad mode
            self.optim_noise_body.grad.data /= self.optim_noise_body.grad.norm(p=2, dim=[1, 2], keepdim=True) + 1e-8
            self.optim_noise_lefthand.grad.data /= (
                self.optim_noise_lefthand.grad.norm(p=2, dim=[1, 2], keepdim=True) + 1e-8
            )
            self.optim_noise_righthand.grad.data /= (
                self.optim_noise_righthand.grad.norm(p=2, dim=[1, 2], keepdim=True) + 1e-8
            )

            # optimize z
            self.optimizer.step()

            # noise perturbation
            # match the noise fraction to the learning rate fraction
            noise_frac = lr_frac
            info["perturb_scale"] = [self.args_diffusion_optimization.perturb_scale * noise_frac] * batch_size

            noise_body = torch.randn_like(self.optim_noise_body)
            noise_lefthand = torch.randn_like(self.optim_noise_lefthand)
            noise_righthand = torch.randn_like(self.optim_noise_righthand)
            self.optim_noise_body.data += noise_body * self.args_diffusion_optimization.perturb_scale * noise_frac
            self.optim_noise_lefthand.data += (
                noise_lefthand * self.args_diffusion_optimization.perturb_scale * noise_frac
            )
            self.optim_noise_righthand.data += (
                noise_righthand * self.args_diffusion_optimization.perturb_scale * noise_frac
            )

            # log the norm(z - start_z)
            info["body_diff_norm"] = (
                (self.optim_noise_body - self.start_noise_body).norm(p=2, dim=[1, 2]).detach().cpu()
            )
            info["lefthand_diff_norm"] = (
                (self.optim_noise_lefthand - self.start_noise_lefthand).norm(p=2, dim=[1, 2]).detach().cpu()
            )
            info["righthand_diff_norm"] = (
                (self.optim_noise_righthand - self.start_noise_righthand).norm(p=2, dim=[1, 2]).detach().cpu()
            )

            # log current z
            info["body_z"] = self.optim_noise_body.detach().cpu()
            info["lefthand_z"] = self.optim_noise_lefthand.detach().cpu()
            info["righthand_z"] = self.optim_noise_righthand.detach().cpu()
            info["pred_sample_body"] = pred_sample_body.detach().cpu()
            info["pred_sample_lefthand"] = pred_sample_lefthand.detach().cpu()
            info["pred_sample_righthand"] = pred_sample_righthand.detach().cpu()

            self.step_count += 1
            prog_bar.update()
            postfix_dict = {
                "loss": info["loss"].mean().item(),
                "decorrelate": info["loss_decorrelate"].mean().item(),
                "target_pos": info["target_pos_loss"].mean().item(),
                "delta": info["delta_loss"].mean().item(),
                "r_h_reg": info["root_height_reg_loss"].mean().item(),
                "f_s_reg": info["foot_sliding_reg_loss"].mean().item(),
                "f_f_reg": info["foot_floating_reg_loss"].mean().item(),
                "lr": self.args_diffusion_optimization.lr * lr_frac,
            }
            if "sdf_loss" in info.keys():
                postfix_dict["loss_sdf"] = info["sdf_loss"].mean().item()
            prog_bar.prog_bar.set_postfix(postfix_dict)

        body_outputs = self.endecoder.decode(pred_sample_body, inputs=inputs)
        gt_body_outputs = self.endecoder.decode(self.endecoder.encode(inputs), inputs=inputs)

        lefthand_inputs = {}
        for k, v in inputs.items():
            if k in [
                "left_base_rotmat",
                "left_base_pos",
                "left_handpose",
                "left_hand_localpos",
                "left_handtip_localpos",
            ]:
                lefthand_inputs[k.replace("left_", "")] = v
                continue
            lefthand_inputs[k] = v
        gt_lefthand_outputs = self.endecoder_lefthand.decode(
            self.endecoder_lefthand.encode(lefthand_inputs), inputs=lefthand_inputs
        )
        lefthand_inputs["base_pos"] = body_outputs["global_pos"][..., 20, :]
        lefthand_inputs["base_rotmat"] = matrix.get_rotation(body_outputs["global_mat"])[..., 20, :, :]
        lefthand_outputs = self.endecoder_lefthand.decode(pred_sample_lefthand, inputs=lefthand_inputs)

        righthand_inputs = {}
        for k, v in inputs.items():
            if k in [
                "right_base_rotmat",
                "right_base_pos",
                "right_handpose",
                "right_hand_localpos",
                "right_handtip_localpos",
            ]:
                righthand_inputs[k.replace("right_", "")] = v
                continue
            righthand_inputs[k] = v
        gt_righthand_outputs = self.endecoder_righthand.decode(
            self.endecoder_righthand.encode(righthand_inputs), inputs=righthand_inputs
        )
        righthand_inputs["base_pos"] = body_outputs["global_pos"][..., 21, :]
        righthand_inputs["base_rotmat"] = matrix.get_rotation(body_outputs["global_mat"])[..., 21, :, :]
        righthand_outputs = self.endecoder_righthand.decode(pred_sample_righthand, inputs=righthand_inputs)

        outputs = {
            # last step's z
            "body_z": self.optim_noise_body.detach(),
            "lefthand_z": self.optim_noise_lefthand.detach(),
            "righthand_z": self.optim_noise_righthand.detach(),
            # previous steps' x
            "body_x": pred_sample_body.detach(),
            "lefthand_x": pred_sample_lefthand.detach(),
            "righthand_x": pred_sample_righthand.detach(),
        }

        outputs["global_pos"] = body_outputs["global_pos"]
        outputs["lefthand_global_pos"] = lefthand_outputs["tip_global_pos"]
        outputs["righthand_global_pos"] = righthand_outputs["tip_global_pos"]

        outputs["gt_output"] = {
            "global_pos": inputs["gt_body_global_pos"],
            "lefthand_global_pos": inputs["gt_lefthand_global_pos"],
            "righthand_global_pos": inputs["gt_righthand_global_pos"],
        }

        outputs["transl"] = body_outputs["transl"]
        outputs["global_orient"] = body_outputs["global_orient"]
        outputs["body_pose"] = body_outputs["body_pose"]
        outputs["left_hand_pose"] = lefthand_outputs["hand_pose"]
        outputs["right_hand_pose"] = righthand_outputs["hand_pose"]
        outputs["gt_output"]["transl"] = inputs["transl"]
        outputs["gt_output"]["global_orient"] = inputs["global_orient"]
        outputs["gt_output"]["body_pose"] = inputs["body_pose"]
        outputs["gt_output"]["left_hand_pose"] = inputs["left_hand_pose"]
        outputs["gt_output"]["right_hand_pose"] = inputs["right_hand_pose"]

        if is_vis:
            for i in range(B):
                wis3d = make_wis3d(name=f"debug_wholebody_{self.vis_i:03d}")
                beta = inputs["beta"][i]
                l = inputs["length"][i]
                transl = body_outputs["transl"][i]
                body_pose = body_outputs["body_pose"][i]
                global_orient = body_outputs["global_orient"][i]
                left_hand_pose = lefthand_outputs["hand_pose"][i]
                right_hand_pose = righthand_outputs["hand_pose"][i]
                smplx_out = self.smplx(
                    transl=transl,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    betas=beta,
                )

                gt_transl = inputs["transl"][i]
                gt_global_orient = inputs["global_orient"][i]
                gt_body_pose = inputs["body_pose"][i]
                gt_left_hand_pose = inputs["left_hand_pose"][i]
                gt_right_hand_pose = inputs["right_hand_pose"][i]
                gt_smplx_out = self.smplx(
                    transl=gt_transl,
                    global_orient=gt_global_orient,
                    body_pose=gt_body_pose,
                    left_hand_pose=gt_left_hand_pose,
                    right_hand_pose=gt_right_hand_pose,
                    betas=beta,
                )

                for j in range(transl.shape[0]):
                    wis3d.set_scene_id(j)
                    wis3d.add_mesh(smplx_out.vertices[j], self.smplx.bm.faces, name="pred_smplx")
                    wis3d.add_mesh(gt_smplx_out.vertices[j], self.smplx.bm.faces, name="gt_smplx")

        return outputs

    def sample_obj_instance(self, inputs, is_vis=False, GRAB_PRE_N=40, ARCTIC_PRE_N=20):
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        max_length = inputs["length"].max()
        B = inputs["length"].shape[0]

        gt_x = self.endecoder_obj._encode(inputs)
        gt_output = self.endecoder_obj._decode(gt_x, inputs=inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        x = torch.randn((B, max_length, self.denoiser3d_obj.output_dim), device=gt_x.device)

        # 2. Conditions
        condition_dict = self.endecoder_obj.encode_condition(inputs)
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
        prog_bar = self.get_prog_bar(self.num_inference_steps, desc="DDIM Sampling Obj")
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
            x0_, _ = self.cfg_denoise_func(self.denoiser3d_obj, model_kwargs, scheduler, enable_cfg)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        outputs = self.endecoder_obj.decode(x, inputs=inputs)
        contact = outputs["contact_mask"] > 0.95  # (B, L, 1)
        contact = contact[..., 0].float()  # (B, L)
        first_con_index = torch.argmax(contact.float(), dim=-1).item()
        if "GRAB" in inputs["meta"][0]["dataset_id"]:
            PRE_N = GRAB_PRE_N
            if first_con_index <= PRE_N:
                if first_con_index > 0:
                    new_transl = torch.zeros_like(outputs["transl"])
                    L = outputs["transl"].shape[1] - PRE_N
                    new_transl[:, PRE_N:] = outputs["transl"][:, first_con_index : first_con_index + L]
                    new_transl[:, : first_con_index - 1] = outputs["transl"][:, : first_con_index - 1]
                    new_transl[:, first_con_index - 1 : PRE_N] = outputs["transl"][
                        :, first_con_index - 1 : first_con_index
                    ]
                    outputs["transl"] = new_transl

                    new_global_orient = torch.zeros_like(outputs["global_orient"])
                    new_global_orient[:, PRE_N:] = outputs["global_orient"][:, first_con_index : first_con_index + L]
                    new_global_orient[:, : first_con_index - 1] = outputs["global_orient"][:, : first_con_index - 1]
                    new_global_orient[:, first_con_index - 1 : PRE_N] = outputs["global_orient"][
                        :, first_con_index - 1 : first_con_index
                    ]
                    outputs["global_orient"] = new_global_orient

                    new_objmat = torch.zeros_like(outputs["obj"])
                    new_objmat[:, PRE_N:] = outputs["obj"][:, first_con_index : first_con_index + L]
                    new_objmat[:, : first_con_index - 1] = outputs["obj"][:, : first_con_index - 1]
                    new_objmat[:, first_con_index - 1 : PRE_N] = outputs["obj"][
                        :, first_con_index - 1 : first_con_index
                    ]
                    outputs["obj"] = new_objmat

                    contact_mask_new = torch.zeros_like(outputs["contact_mask"])
                    contact_mask_new[:, PRE_N:] = outputs["contact_mask"][:, first_con_index : first_con_index + L]
                    contact_mask_new[:, : first_con_index - 1] = outputs["contact_mask"][:, : first_con_index - 1]
                    contact_mask_new[:, first_con_index - 1 : PRE_N] = outputs["contact_mask"][
                        :, first_con_index - 1 : first_con_index
                    ]
                    outputs["contact_mask"] = contact_mask_new
                else:
                    new_transl = torch.zeros_like(outputs["transl"])
                    L = outputs["transl"].shape[1] - PRE_N
                    new_transl[:, PRE_N:] = outputs["transl"][:, first_con_index : first_con_index + L]
                    new_transl[:, :PRE_N] = outputs["transl"][:, :1]
                    outputs["transl"] = new_transl

                    new_global_orient = torch.zeros_like(outputs["global_orient"])
                    new_global_orient[:, PRE_N:] = outputs["global_orient"][:, first_con_index : first_con_index + L]
                    new_global_orient[:, :PRE_N] = outputs["global_orient"][:, :1]
                    outputs["global_orient"] = new_global_orient

                    new_objmat = torch.zeros_like(outputs["obj"])
                    new_objmat[:, PRE_N:] = outputs["obj"][:, first_con_index : first_con_index + L]
                    new_objmat[:, :PRE_N] = outputs["obj"][:, :1]
                    outputs["obj"] = new_objmat

                    contact_mask_new = torch.zeros_like(outputs["contact_mask"])
                    contact_mask_new[:, PRE_N:] = outputs["contact_mask"][:, first_con_index : first_con_index + L]
                    contact_mask_new[:, :PRE_N] = 0.0
                    outputs["contact_mask"] = contact_mask_new
        elif "ARCTIC" in inputs["meta"][0]["dataset_id"]:
            PRE_N = ARCTIC_PRE_N
            if PRE_N > 0:
                L = outputs["transl"].shape[1]
                if inputs["is_start"][0]:
                    new_transl = torch.zeros_like(outputs["transl"])
                    new_transl[:, :PRE_N] = outputs["transl"][:, :1]
                    new_transl[:, PRE_N:] = outputs["transl"][:, 1 : L - PRE_N + 1]
                    outputs["transl"] = new_transl

                    new_global_orient = torch.zeros_like(outputs["global_orient"])
                    new_global_orient[:, :PRE_N] = outputs["global_orient"][:, :1]
                    new_global_orient[:, PRE_N:] = outputs["global_orient"][:, 1 : L - PRE_N + 1]
                    outputs["global_orient"] = new_global_orient

                    new_angles = torch.zeros_like(outputs["angles"])
                    new_angles[:, :PRE_N] = outputs["angles"][:, :1]
                    new_angles[:, PRE_N:] = outputs["angles"][:, 1 : L - PRE_N + 1]
                    outputs["angles"] = new_angles

                    new_objmat = torch.zeros_like(outputs["obj"])
                    new_objmat[:, :PRE_N] = outputs["obj"][:, :1]
                    new_objmat[:, PRE_N:] = outputs["obj"][:, 1 : L - PRE_N + 1]
                    outputs["obj"] = new_objmat

                    contact_mask_new = torch.zeros_like(outputs["contact_mask"])
                    contact_mask_new[:, :PRE_N] = 0.0
                    contact_mask_new[:, PRE_N:] = outputs["contact_mask"][:, 1 : L - PRE_N + 1]
                    outputs["contact_mask"] = contact_mask_new

        outputs["gt_output"] = gt_output
        return outputs

    def obj_forward(self, inputs, obj_outputs):

        if "ARCTIC" in inputs["meta"][0]["dataset_id"]:
            obj_name = inputs["meta"][0]["obj_name"]
            arctic_obj_data = {}
            arctic_obj_data["angles"] = obj_outputs["angles"][0].cpu()
            arctic_obj_data["transl"] = obj_outputs["transl"][0].cpu()
            arctic_obj_data["global_orient"] = obj_outputs["global_orient"][0].cpu()
            with torch.no_grad():
                obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
                obj_out = self.object_tensor(**arctic_obj_data, query_names=obj_ns)
            obj_verts = obj_out["v"]
            obj_faces = obj_out["f"][0]
            obj_normals = obj_out["v_normal"].to(inputs["scale"].device)
            obj_verts = obj_verts.to(inputs["scale"].device)
            filter_obj_verts = obj_verts.clone()

            basis_point = inputs["basis_point"]
            M = basis_point.shape[-2]
            basis_point_bottom = basis_point.clone()
            basis_point_top = basis_point.clone()  # (B, M, 3)
            top_mask, bottom_mask = self.object_tensor.get_part_obj_mask(obj_name)

            top_verts = obj_verts[..., top_mask, :]
            bottom_verts = obj_verts[..., bottom_mask, :]
            center = inputs["center"]
            obj_globalmat = obj_outputs["obj"]  # (B, L, 4, 4)
            scale = inputs["scale"]  # (B, 1)
            global_obj_center = matrix.get_position_from(center[:, None], obj_globalmat)  # (B, N, 3)
            global_obj_center_mat = matrix.get_TRS(
                matrix.get_rotation(obj_globalmat), global_obj_center
            )  # (B, N, 4, 4)
            global_obj_center_mat = global_obj_center_mat[0]  # assume batch is always 0
            obj_local_top_verts = matrix.get_relative_position_to(top_verts, global_obj_center_mat)  # (N, M, 3)
            obj_local_bottom_verts = matrix.get_relative_position_to(bottom_verts, global_obj_center_mat)  # (N, M, 3)
            scaled_obj_local_top_verts = obj_local_top_verts / scale[0]
            scaled_obj_local_bottom_verts = obj_local_bottom_verts / scale[0]
            bps_bottom, bps_bottom_ind = calculate_bps(
                basis_point_bottom, scaled_obj_local_bottom_verts, return_xyz=False
            )
            bps_top, bps_top_ind = calculate_bps(basis_point_top, scaled_obj_local_top_verts, return_xyz=False)
            bps = torch.cat([bps_bottom, bps_top], dim=-1)

        else:
            obj_name = inputs["meta"][0]["obj_name"]
            grab_path = "./inputs/grab_extracted/tools/object_meshes/contact_meshes"
            obj_mesh = os.path.join(grab_path, f"{obj_name}.ply")
            obj_mesh = Mesh(filename=obj_mesh)
            obj_vtemp = np.array(obj_mesh.vertices)
            grab_object_tensor = ObjectModel(v_template=obj_vtemp)
            grab_object_tensor = grab_object_tensor.to(inputs["scale"].device)

            grab_obj_data = {}
            grab_obj_data["transl"] = obj_outputs["transl"][0]
            grab_obj_data["global_orient"] = obj_outputs["global_orient"][0]
            with torch.no_grad():
                obj_out = grab_object_tensor(**grab_obj_data)
            obj_verts = obj_out.vertices
            obj_faces = obj_mesh.faces
            N = 4096
            filter_obj_verts = obj_verts.clone()
            _, select_ind = sample_farthest_points(
                torch.tensor(obj_vtemp, dtype=torch.float32, device=obj_verts.device)[None], K=N
            )
            select_ind = select_ind[0]
            filter_obj_verts = filter_obj_verts[..., select_ind, :].clone()
            obj_normals = filter_obj_verts.clone()  # BUG: this is not normals

            basis_point = inputs["basis_point"]
            center = inputs["center"]
            obj_globalmat = obj_outputs["obj"]  # (B, L, 4, 4)
            scale = inputs["scale"]  # (B, 1)
            global_obj_center = matrix.get_position_from(center[:, None], obj_globalmat)  # (B, N, 3)
            global_obj_center_mat = matrix.get_TRS(
                matrix.get_rotation(obj_globalmat), global_obj_center
            )  # (B, N, 4, 4)
            global_obj_center_mat = global_obj_center_mat[0]  # assume batch is always 0
            obj_local_verts = matrix.get_relative_position_to(filter_obj_verts, global_obj_center_mat)  # (N, M, 3)
            scaled_obj_local_verts = obj_local_verts / scale[0]
            bps, bps_ind = calculate_bps(basis_point, scaled_obj_local_verts, return_xyz=False)

        return obj_verts, obj_faces, obj_normals, filter_obj_verts, bps

    def sample_bpstraj(self, inputs, obj_verts, obj_outputs):
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        length = inputs["length"]
        B = inputs["length"].shape[0]

        gt_x = self.endecoder_invtraj.encode(inputs)
        gt_output = self.endecoder_invtraj.decode(gt_x, inputs=inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        x = torch.randn((B, length, self.denoiser3d_invtraj.output_dim), device=length.device)

        # 2. Conditions
        condition_dict = self.endecoder_invtraj.encode_condition(inputs, obj_outputs)
        f_condition = condition_dict

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps, desc="DDIM Sampling invtraj")
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
            x0_, _ = self.cfg_denoise_func(self.denoiser3d_invtraj, model_kwargs, scheduler, enable_cfg)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        outputs = self.endecoder_invtraj.decode(x, inputs=inputs)

        CONTACT_TRESH = 0.95
        contact = outputs["contact"]
        # if no contact in objtraj, no finger contact
        contact_mask = (obj_outputs["contact_mask"] > CONTACT_TRESH).float() * contact
        contact_mask = contact_mask > CONTACT_TRESH
        # contact_mask_bkp = contact_mask.clone()
        for j in range(contact_mask.shape[-1]):
            for t_ in range(contact_mask.shape[1]):
                prev_t = max(0, t_ - 1)
                after_t = min(contact_mask.shape[1] - 1, t_ + 1)
                if contact_mask[0, t_, j]:
                    if contact_mask[0, prev_t, j] == 0 and contact_mask[0, after_t, j] == 0:
                        contact_mask[0, t_, j] = 0
        outputs["contact_mask"] = contact_mask
        left_wrist_mask = torch.any(contact_mask[..., :5], dim=-1)
        right_wrist_mask = torch.any(contact_mask[..., 5:], dim=-1)
        wrist_mask = torch.stack([left_wrist_mask, right_wrist_mask], dim=-1)
        outputs["wrist_mask"] = wrist_mask

        with torch.enable_grad():
            traj_mask = torch.cat([wrist_mask, contact_mask], dim=-1)  # (B, L, 12)
            basis_point = inputs["basis_finger_point"].clone().detach()  # (B, M, 3)
            M = basis_point.shape[1]
            center = inputs["center"]  # (B, 3)
            obj_mat = inputs["obj"]  # (B, L, 2, 4, 4)
            scale = inputs["scale"]  # (B, 1)
            global_obj_center_pos = matrix.get_position_from(center[:, None], obj_mat[..., 0, :, :])  # (B, L, 3)
            global_obj_center_rotmat = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
            global_obj_center_mat = matrix.get_TRS(global_obj_center_rotmat, global_obj_center_pos)  # (B, L, 4, 4)
            B, L = obj_mat.shape[:2]
            finger_dist = outputs["finger_dist"].clone().detach()
            finger_dist = torch.clamp(finger_dist, min=0.0, max=10.0)
            finger_dist = finger_dist.reshape(B, L, M, -1)
            threshold = 10.0
            finger_dist_mask = finger_dist < threshold
            J = finger_dist.shape[-1]

            finger_vert_dist = outputs["finger_vert_dist"].clone().detach()  # (B, 10, 3)

            optim_finger_traj = torch.zeros(B, L, J, 3).cuda().detach().requires_grad_()

            optimizer = torch.optim.Adam([optim_finger_traj], lr=self.args_traj_optimization.lr)
            lr_scheduler = []
            lr_warm_up_steps = self.args_traj_optimization.lr_warm_up_steps
            total_steps = self.args_traj_optimization.num_opt_steps
            lr_scheduler.append(lambda step: warmup_scheduler(step, lr_warm_up_steps))
            lr_scheduler.append(lambda step: cosine_decay_scheduler(step, total_steps, total_steps, decay_first=False))

            prog_bar = self.get_prog_bar(total_steps, desc="Traj Optimization")
            for i in range(total_steps):
                # set lr
                lr_frac = 1
                if len(lr_scheduler) > 0:
                    for scheduler in lr_scheduler:
                        lr_frac *= scheduler(i)
                lr = self.args_traj_optimization.lr * lr_frac
                for _, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = lr

                optim_finger_delta_pos = (
                    basis_point[:, None, :, None] - optim_finger_traj[:, :, None]
                )  # (B, L, M, J, 3)
                optim_finger_delta_dist = torch.norm(optim_finger_delta_pos, dim=-1)  # (B, L, M, J)

                loss_bps = (
                    (optim_finger_delta_dist - finger_dist) * traj_mask[:, :, None] * finger_dist_mask
                )  # (B, L, M, J)
                loss_bps = loss_bps.abs()  # (B, L, M, J)
                loss_bps = loss_bps.mean()

                unscale_optim_finger_traj = optim_finger_traj * scale[:, None, None]  # (B, L, J, 3)
                global_optim_finger_traj = matrix.get_position_from(
                    unscale_optim_finger_traj, global_obj_center_mat
                )  # (B, L, J, 3)
                global_optim_finger_traj = global_optim_finger_traj[..., 2:, :]  # only fingertip
                # obj_verts: (L, M, 3)
                optim_finger_vert_dist = (
                    global_optim_finger_traj[..., None, :] - obj_verts[None, :, None]
                )  # (B, L, J, M, 3)
                optim_finger_vert_dist = torch.norm(optim_finger_vert_dist, dim=-1)  # (B, L, J, M)
                optim_finger_vert_dist = optim_finger_vert_dist.min(dim=-1)[0]  # (B, L, J)
                loss_vert_dist = (finger_vert_dist - optim_finger_vert_dist) * traj_mask[..., 2:]  # (B, L, J)
                loss_vert_dist = loss_vert_dist.abs()
                loss_vert_dist = loss_vert_dist.mean()

                vert_dist_w = 0.0 if i > 300 else 0.0
                loss = loss_bps + vert_dist_w * loss_vert_dist
                # print(f"{i:03d}: loss_bps: {loss_bps.item()}, loss_vert_dist: {loss_vert_dist.item()}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if prog_bar is not None:
                    prog_bar.update()
                    prog_bar.prog_bar.set_postfix(
                        {
                            "loss": loss.item(),
                            "loss_bps": loss_bps.item(),
                            "loss_vert_dist": loss_vert_dist.item(),
                            "lr": lr,
                        }
                    )

            unscale_optim_finger_traj = optim_finger_traj.clone().detach() * scale[:, None, None]  # (B, L, J, 3)
            global_optim_finger_traj = matrix.get_position_from(unscale_optim_finger_traj, global_obj_center_mat)

            outputs["wrist_traj_global"] = global_optim_finger_traj[..., :2, :]
            outputs["finger_traj_global"] = global_optim_finger_traj[..., 2:, :]

        outputs["gt_output"] = gt_output
        outputs["gt_output"]["wrist_traj_global"] = get_wrist_trajectory(inputs["humanoid"])
        gt_left_finger_traj = get_finger_trajectory(inputs["humanoid"], is_right=False, istip=True)
        gt_right_finger_traj = get_finger_trajectory(inputs["humanoid"], is_right=True, istip=True)
        outputs["gt_output"]["finger_traj_global"] = torch.cat([gt_left_finger_traj, gt_right_finger_traj], dim=-2)

        contact_mask = outputs["contact_mask"].clone()
        wrist_mask = outputs["wrist_mask"].clone()
        wrist_traj_global = outputs["wrist_traj_global"].clone()
        finger_traj_global = outputs["finger_traj_global"].clone()

        finger_obj_dist = torch.norm(
            global_optim_finger_traj - global_obj_center_pos[..., None, :], dim=-1
        )  # (B, L, J)
        for i in range(finger_obj_dist.shape[0]):
            for j in range(finger_obj_dist.shape[1]):
                for k in range(finger_obj_dist.shape[2]):
                    if finger_obj_dist[i, j, k] > 0.5:
                        if k == 0:
                            wrist_mask[i, j, k] = False
                            contact_mask[i, j, :5] = False
                            wrist_traj_global[i, j, k] = 0.0
                            finger_traj_global[i, j, :5] = 0.0
                        elif k == 1:
                            wrist_mask[i, j, k] = False
                            contact_mask[i, j, 5:] = False
                            wrist_traj_global[i, j, k] = 0.0
                            finger_traj_global[i, j, 5:] = 0.0
                        else:
                            contact_mask[i, j, k - 2] = False
                            finger_traj_global[i, j, k - 2] = 0.0

        outputs["contact_mask"] = contact_mask
        outputs["wrist_traj_global"] = wrist_traj_global
        outputs["finger_traj_global"] = finger_traj_global
        return outputs

    def sample_hand_instance(self, inputs, is_right=False, is_vis=False):
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        if is_right:
            denoiser3d = self.denoiser3d_righthand
            endecoder = self.endecoder_righthand
        else:
            denoiser3d = self.denoiser3d_lefthand
            endecoder = self.endecoder_lefthand

        max_length = inputs["length"].max()
        max_length = max(inputs["transl"].shape[1], max_length)
        B = inputs["length"].shape[0]
        hand_inputs = {}
        for k, v in inputs.items():
            if is_right:
                if k in [
                    "right_base_rotmat",
                    "right_base_pos",
                    "right_handpose",
                    "right_hand_localpos",
                    "right_handtip_localpos",
                ]:
                    hand_inputs[k.replace("right_", "")] = v.clone()
                    continue
            else:
                if k in [
                    "left_base_rotmat",
                    "left_base_pos",
                    "left_handpose",
                    "left_hand_localpos",
                    "left_handtip_localpos",
                ]:
                    hand_inputs[k.replace("left_", "")] = v.clone()
                    continue
            hand_inputs[k] = v

        gt_x = endecoder._encode(hand_inputs)
        gt_output = endecoder._decode(gt_x, inputs=hand_inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        if endecoder.vae is None:
            x = torch.randn((B, max_length, denoiser3d.output_dim), device=gt_x.device)
        else:
            x = torch.randn((B, denoiser3d.latent_size, denoiser3d.latent_dim), device=gt_x.device)
        init_noise = x.clone()

        # 2. Conditions
        condition_dict = endecoder.encode_condition(hand_inputs)
        f_condition = condition_dict

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prefix = "right" if is_right else "left"
        prog_bar = self.get_prog_bar(self.num_inference_steps, desc=f"DDIM Sampling {prefix}hand")
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler)  # for scheduler.step()
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                inputs=hand_inputs,
                f_condition=f_condition,
                enable_cfg=enable_cfg,
            )
            x0_, _ = self.cfg_denoise_func(denoiser3d, model_kwargs, scheduler, enable_cfg)

            scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
            x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

            # *. Update and store intermediate results
            x = xprev_

            # progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                if prog_bar is not None:
                    prog_bar.update()

        outputs = endecoder.decode(x, inputs=hand_inputs)
        outputs["init_noise"] = init_noise
        outputs["pred_sample"] = x
        outputs["gt_output"] = gt_output

        if is_vis:
            for i in range(1):
                wis3d = make_wis3d(name=f"debug_wholebody_{i:03d}")
                l = hand_inputs["length"][i]
                pred_motion = outputs["tip_global_pos"][i]
                gt_motion = outputs["gt_output"]["tip_global_pos"][i]
                pred_motion[l:] = pred_motion[l - 1 : l]
                gt_motion[l:] = gt_motion[l - 1 : l]
                prefix = "right" if is_right else "left"
                add_motion_as_lines(
                    pred_motion, wis3d, name=f"pred-{prefix}hand-ddpm", skeleton_type="handtip", radius=0.005
                )
                add_motion_as_lines(gt_motion, wis3d, name=f"gt-{prefix}hand", skeleton_type="handtip", radius=0.005)
        return outputs

    def sample_body_instance(self, inputs, is_vis=False):
        # Setup
        outputs = dict()
        enable_cfg = False if self.guidance_scale == 0 else True
        scheduler = self.test_scheduler

        max_length = inputs["length"].max()
        max_length = max(inputs["transl"].shape[1], max_length)
        B = inputs["length"].shape[0]
        body_inputs = {}
        for k, v in inputs.items():
            if k == "beta":
                # this makes smplx shape different, but we use skeleton in endecoder, so it is fine.
                body_inputs[k] = v[..., :10]
            else:
                body_inputs[k] = v

        # will make foot at ground, cause slightly height difference here for gt.
        gt_x = self.endecoder._encode(body_inputs)
        gt_output = self.endecoder._decode(gt_x, inputs=body_inputs)

        # 1. Prepare target variable x, which will be denoised progressively
        if self.endecoder.vae is None:
            x = torch.randn((B, max_length, self.denoiser3d.output_dim), device=gt_x.device)
        else:
            x = torch.randn((B, self.denoiser3d.latent_size, self.denoiser3d.latent_dim), device=gt_x.device)
        init_noise = x.clone()

        # 2. Conditions
        condition_dict = self.endecoder.encode_condition(body_inputs)
        f_condition = condition_dict

        # Encode CLIP embedding
        text = body_inputs["caption"]

        if not enable_cfg:
            text = ["" for _ in range(len(text))]

        clip_text = self.clip.encode_text(text, enable_cfg=enable_cfg, with_projection=True)  # (B, D)
        f_condition["f_text"] = clip_text.f_text

        # *. Denoising loop
        # scheduler: timestep, extra_step_kwargs
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.num_inference_steps * scheduler.order
        prog_bar = self.get_prog_bar(self.num_inference_steps, desc="DDIM Sampling body")
        extra_step_kwargs = self.prepare_extra_step_kwargs(scheduler)  # for scheduler.step()
        for i, t in enumerate(timesteps):
            # 1. Denoiser + Sampler.step
            model_kwargs = self.build_model_kwargs(
                x=x,
                timesteps=t,
                inputs=body_inputs,
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

        outputs = self.endecoder.decode(x, inputs=body_inputs)
        outputs["init_noise"] = init_noise
        outputs["pred_sample"] = x
        outputs["gt_output"] = gt_output

        if is_vis:
            for i in range(1):
                wis3d = make_wis3d(name=f"debug_wholebody_{self.vis_i:03d}")
                text_ = body_inputs["caption"][i]
                l = body_inputs["length"][i]
                pred_motion = outputs["global_pos"][i]
                gt_motion = outputs["gt_output"]["global_pos"][i]
                pred_motion[l:] = pred_motion[l - 1 : l]
                gt_motion[l:] = gt_motion[l - 1 : l]
                add_motion_as_lines(pred_motion, wis3d, name=f"pred-{text_}-ddpm")
                add_motion_as_lines(gt_motion, wis3d, name=f"gt-{text_}", radius=0.005)
        return outputs

    # ========== Sample ========== #
    def forward_sample(self, inputs, is_vis=False):
        is_vis = False

        obj_outputs = self.sample_obj_instance(inputs, is_vis=is_vis, ARCTIC_PRE_N=20)

        obj_verts, obj_faces, obj_normals, filter_obj_verts, input_bps = self.obj_forward(inputs, obj_outputs)
        gt_obj_verts, gt_obj_faces, gt_obj_normals, gt_filter_obj_verts, gt_input_bps = self.obj_forward(
            inputs, obj_outputs["gt_output"]
        )

        if is_vis:
            wis3d = make_wis3d(name=f"debug_wholebody_{self.vis_i:03d}")
            for i in range(obj_verts.shape[0]):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(obj_verts[i], obj_faces, name="ori_obj")
                wis3d.add_mesh(gt_obj_verts[i], gt_obj_faces, name="gt_obj")

        body_outputs = self.sample_body_instance(inputs, is_vis=is_vis)
        lefthand_outputs = self.sample_hand_instance(inputs, is_right=False, is_vis=is_vis)
        righthand_outputs = self.sample_hand_instance(inputs, is_right=True, is_vis=is_vis)

        obj_inputs = {}
        obj_inputs["length"] = inputs["length"]
        obj_inputs["obj"] = obj_outputs["obj"][..., None, :, :]  # (B, N, 4, 4) -> (B, N, 1, 4, 4)
        obj_inputs["obj_frame0"] = inputs["obj_frame0"]
        obj_inputs["scale"] = inputs["scale"]
        obj_inputs["center"] = inputs["center"]
        if "angles" in obj_outputs:
            obj_inputs["angles"] = obj_outputs["angles"]
        obj_inputs["transl"] = obj_outputs["transl"]
        obj_inputs["beta"] = inputs["beta"]
        obj_inputs["bps"] = input_bps[None]  # (B, N, M)
        obj_inputs["basis_finger_point"] = inputs["basis_finger_point"]

        obj_inputs["contact"] = inputs["contact"]
        obj_inputs["finger_dist"] = inputs["finger_dist"]
        obj_inputs["finger_vert_dist"] = inputs["finger_vert_dist"]
        obj_inputs["humanoid"] = inputs["humanoid"]

        invtraj_outputs = self.sample_bpstraj(obj_inputs, filter_obj_verts, obj_outputs)
        invtraj_outputs["obj"] = obj_outputs["obj"]

        if is_vis:
            gt_wrist_traj = invtraj_outputs["gt_output"]["wrist_traj_global"][0]
            gt_finger_traj = invtraj_outputs["gt_output"]["finger_traj_global"][0]
            wrist_traj = invtraj_outputs["wrist_traj_global"][0]
            finger_traj = invtraj_outputs["finger_traj_global"][0]
            wis3d = make_wis3d(name=f"debug_wholebody_{self.vis_i:03d}")
            for i in range(gt_wrist_traj.shape[0]):
                wis3d.set_scene_id(i)
                prefix = ""
                wis3d.add_point_cloud(gt_wrist_traj[i, [0]], name="gt_left_wrist_traj")
                wis3d.add_point_cloud(gt_wrist_traj[i, [1]], name="gt_right_wrist_traj")
                wis3d.add_point_cloud(wrist_traj[i, [0]], name=f"{prefix}reverse_bps_left_wrist_traj")
                wis3d.add_point_cloud(wrist_traj[i, [1]], name=f"{prefix}reverse_bps_right_wrist_traj")
                wis3d.add_point_cloud(gt_finger_traj[i, :5], name="gt_left_finger_traj")
                wis3d.add_point_cloud(gt_finger_traj[i, 5:], name="gt_right_finger_traj")
                wis3d.add_point_cloud(finger_traj[i, :5], name=f"{prefix}reverse_bps_left_finger_traj")
                wis3d.add_point_cloud(finger_traj[i, 5:], name=f"{prefix}reverse_bps_right_finger_traj")

        center = inputs["center"]  # (B, 3)
        obj_mat = obj_outputs["obj"][..., None, :, :]  # (B, L, 1, 4, 4)
        global_obj_center_pos = matrix.get_position_from(center[:, None], obj_mat[..., 0, :, :])  # (B, L, 3)
        obj_local_verts = obj_verts - global_obj_center_pos[0, :, None]  # (L, M, 3)
        obj_verts = obj_local_verts + global_obj_center_pos[0, :, None]  # (L, M, 3)

        if is_vis:
            wis3d = make_wis3d(name=f"debug_wholebody_{self.vis_i:03d}")
            for i in range(obj_verts.shape[0]):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(obj_verts[i], obj_faces, name="obj")

        body_pred_sample = body_outputs["pred_sample"]
        lefthand_pred_sample = lefthand_outputs["pred_sample"]
        righthand_pred_sample = righthand_outputs["pred_sample"]
        inputs["init_noise_body"] = torch.randn_like(body_pred_sample)
        inputs["init_noise_lefthand"] = torch.randn_like(lefthand_pred_sample)
        inputs["init_noise_righthand"] = torch.randn_like(righthand_pred_sample)

        inputs["gt_body_global_pos"] = body_outputs["gt_output"]["global_pos"]
        inputs["gt_lefthand_global_pos"] = lefthand_outputs["gt_output"]["tip_global_pos"]
        inputs["gt_righthand_global_pos"] = righthand_outputs["gt_output"]["tip_global_pos"]
        torch.cuda.empty_cache()

        outputs = self.diffusion_optim(
            inputs, invtraj_outputs, filter_obj_verts[None], obj_normals[None], is_vis=is_vis
        )

        outputs["target_finger_traj"] = invtraj_outputs["finger_traj_global"]
        outputs["target_wrist_traj"] = invtraj_outputs["wrist_traj_global"]
        outputs["gt_output"]["finger_traj_global"] = invtraj_outputs["gt_output"]["finger_traj_global"]
        outputs["gt_output"]["wrist_traj_global"] = invtraj_outputs["gt_output"]["wrist_traj_global"]
        outputs["invtraj_outputs"] = invtraj_outputs

        outputs["obj"] = obj_outputs["obj"]
        if "angles" in obj_outputs:
            outputs["obj_angles"] = obj_outputs["angles"]
        outputs["obj_global_orient"] = obj_outputs["global_orient"]
        outputs["obj_transl"] = obj_outputs["transl"]

        if is_vis:
            pass
            # raise NotImplementedError
        self.vis_i += 1
        return outputs

    def load_pretrained_model(self, ckpt_path):
        main_ckpt_path = ckpt_path["body"]
        invtraj_ckpt_path = ckpt_path["invtraj"]
        obj_ckpt_path = ckpt_path["obj"]
        lefthand_ckpt_path = ckpt_path.get("lefthand", None)
        righthand_ckpt_path = ckpt_path.get("righthand", None)

        body_state_dict = torch.load(main_ckpt_path, "cpu")["state_dict"]

        new_state_dict = {}
        for k, v in body_state_dict.items():
            new_state_dict[k] = v
        if lefthand_ckpt_path is not None:
            lefthand_state_dict = torch.load(lefthand_ckpt_path, "cpu")["state_dict"]
            for k, v in lefthand_state_dict.items():
                new_state_dict[k.replace("denoiser3d", "denoiser3d_lefthand")] = v
        if righthand_ckpt_path is not None:
            righthand_state_dict = torch.load(righthand_ckpt_path, "cpu")["state_dict"]
            for k, v in righthand_state_dict.items():
                new_state_dict[k.replace("denoiser3d", "denoiser3d_righthand")] = v
        invtraj_state_dict = torch.load(invtraj_ckpt_path, "cpu")["state_dict"]
        for k, v in invtraj_state_dict.items():
            new_state_dict[k.replace("denoiser3d", "denoiser3d_invtraj")] = v
        obj_state_dict = torch.load(obj_ckpt_path, "cpu")["state_dict"]
        for k, v in obj_state_dict.items():
            new_state_dict[k.replace("denoiser3d", "denoiser3d_obj")] = v

        state_dict = new_state_dict
        return state_dict


@torch.enable_grad()
def body_ddim_loop_with_gradient(init_noise, pipeline, inputs):
    # Setup
    outputs = dict()
    enable_cfg = False if pipeline.guidance_scale == 0 else True
    scheduler = pipeline.test_scheduler

    max_length = inputs["length"].max()
    max_length = max(inputs["transl"].shape[1], max_length)
    B = inputs["length"].shape[0]
    body_inputs = {}
    for k, v in inputs.items():
        if k == "beta":
            body_inputs[k] = v[..., :10]
        else:
            body_inputs[k] = v

    x = init_noise

    # 2. Conditions
    condition_dict = pipeline.endecoder.encode_condition(body_inputs)
    f_condition = condition_dict
    # use zero condition
    for k in f_condition.keys():
        f_condition[k] = torch.zeros_like(f_condition[k])

    # Encode CLIP embedding
    text = body_inputs["caption"]

    if not enable_cfg:
        text = ["" for _ in range(len(text))]

    # Follow DNO, do not use text during optimization
    text = ["" for _ in range(len(text))]

    clip_text = pipeline.clip.encode_text(text, enable_cfg=enable_cfg, with_projection=True)  # (B, D)
    f_condition["f_text"] = clip_text.f_text

    # *. Denoising loop
    # scheduler: timestep, extra_step_kwargs
    scheduler.set_timesteps(pipeline.num_optim_ddim_steps)
    timesteps = scheduler.timesteps
    num_warmup_steps = len(timesteps) - pipeline.num_optim_ddim_steps * scheduler.order
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(scheduler)  # for scheduler.step()
    for i, t in enumerate(timesteps):
        # 1. Denoiser + Sampler.step
        model_kwargs = pipeline.build_model_kwargs(
            x=x,
            timesteps=t,
            inputs=body_inputs,
            f_condition=f_condition,
            enable_cfg=enable_cfg,
        )
        x0_, _ = pipeline.cfg_denoise_func(pipeline.denoiser3d, model_kwargs, scheduler, enable_cfg)

        scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
        x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

        # *. Update and store intermediate results
        x = xprev_
    return x


@torch.enable_grad()
def hand_ddim_loop_with_gradient(init_noise, pipeline, inputs, is_right=False):
    # Setup
    enable_cfg = False if pipeline.guidance_scale == 0 else True
    scheduler = pipeline.test_scheduler
    if is_right:
        denoiser3d = pipeline.denoiser3d_righthand
        endecoder = pipeline.endecoder_righthand
    else:
        denoiser3d = pipeline.denoiser3d_lefthand
        endecoder = pipeline.endecoder_lefthand

    max_length = inputs["length"].max()
    max_length = max(inputs["transl"].shape[1], max_length)
    B = inputs["length"].shape[0]
    hand_inputs = {}
    for k, v in inputs.items():
        if is_right:
            if k in ["right_base_rotmat", "right_base_pos", "right_handpose", "right_hand_localpos"]:
                hand_inputs[k.replace("right_", "")] = v
                continue
        else:
            if k in ["left_base_rotmat", "left_base_pos", "left_handpose", "left_hand_localpos"]:
                hand_inputs[k.replace("left_", "")] = v
                continue
        hand_inputs[k] = v

    x = init_noise

    # 2. Conditions
    condition_dict = endecoder.encode_condition(hand_inputs)
    f_condition = condition_dict

    # *. Denoising loop
    # scheduler: timestep, extra_step_kwargs
    scheduler.set_timesteps(pipeline.num_optim_ddim_steps)
    timesteps = scheduler.timesteps
    num_warmup_steps = len(timesteps) - pipeline.num_optim_ddim_steps * scheduler.order
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(scheduler)  # for scheduler.step()
    for i, t in enumerate(timesteps):
        # 1. Denoiser + Sampler.step
        model_kwargs = pipeline.build_model_kwargs(
            x=x,
            timesteps=t,
            inputs=hand_inputs,
            f_condition=f_condition,
            enable_cfg=enable_cfg,
        )
        x0_, _ = pipeline.cfg_denoise_func(denoiser3d, model_kwargs, scheduler, enable_cfg)

        scheduler_out = scheduler.step(x0_, t, x, **extra_step_kwargs)
        x0_, xprev_ = scheduler_out.pred_original_sample, scheduler_out.prev_sample

        # *. Update and store intermediate results
        x = xprev_
    return x
