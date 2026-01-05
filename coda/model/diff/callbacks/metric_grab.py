import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from pytorch_lightning.utilities import rank_zero_only
from coda.configs import MainStore, builds
from hydra.utils import instantiate

from coda.utils.pylogger import Log

import coda.utils.matrix as matrix
from coda.utils.eval.metric_utils import ListAggregator
from coda.model.diff.utils.motion3d_endecoder import (
    HmlobjvecGrabEnDecoder,
)
from coda.utils.hml3d.metric import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_diversity_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_multimodality_np,
)
import coda.network.evaluator.t2m_motionenc as t2m_motionenc
import coda.network.evaluator.t2m_textenc as t2m_textenc

from coda.network.evaluator.word_vectorizer import POS_enumerator
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from coda.model.diff.callbacks.utils import compute_mpjpe, compute_distance_error, calculate_foot_sliding
import imageio
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2


class MetricWholeBodyObjPose(pl.Callback):
    def __init__(self, prefix=""):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        # vid->result
        self.target_dataset_id = "GRAB_WHOLEBODYPOSE"
        self.metric_aggregator = {}
        self.prefix = prefix

        self.top_k = 3
        self.R_size = 32
        self.diversity_times = 300
        self.checkpoint_name = "grab_short"
        self.endecoder_opt = {"_target_": "coda.model.diff.utils.motion3d_endecoder.HmlobjvecGrabEnDecoder"}

        self.fs_aggregator = ListAggregator()
        self.gt_fs_aggregator = ListAggregator()

        self.is_fid = True
        try:
            self.text_embeddings = ListAggregator()
            self.gen_motion_embeddings = ListAggregator()
            self.gt_motion_embeddings = ListAggregator()
            self._get_t2m_evaluator()
        except Exception as e:
            self.is_fid = False

        self.data_endecoder: HmlobjvecGrabEnDecoder = instantiate(self.endecoder_opt, _recursive_=False)
        self.encoder_motion3d = self.data_endecoder.encode

        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    def _get_t2m_evaluator(self):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        ######
        # OPT is from https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/networks/evaluator_wrapper.py
        ######
        checkpoints_dir = f"./inputs/checkpoints/{self.checkpoint_name}"
        Log.info(f"Loading {self.checkpoint_name} evaluators")

        opt = {
            "dim_word": 300,
            "max_motion_length": 300,
            "dim_pos_ohot": len(POS_enumerator),
            "dim_motion_hidden": 1024,
            "max_text_len": 20,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": 52 * 3 * 2 + 3,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "checkpoints_dir": checkpoints_dir,
            "unit_length": 4,
        }
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=opt["dim_word"],
            pos_size=opt["dim_pos_ohot"],
            hidden_size=opt["dim_text_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=opt["dim_pose"] - 4,
            hidden_size=opt["dim_movement_enc_hidden"],
            output_size=opt["dim_movement_latent"],
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=opt["dim_movement_latent"],
            hidden_size=opt["dim_motion_hidden"],
            output_size=opt["dim_coemb_hidden"],
        )
        # load pretrianed
        t2m_checkpoint = torch.load(
            os.path.join(opt["checkpoints_dir"], "text_mot_match/model/finest.tar"),
            map_location="cpu",
        )
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != self.target_dataset_id:
            return

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()

        finger_ind = [
            1,
            2,
            3,  # index
            5,
            6,
            7,  # middle
            9,
            10,
            11,  # pinky
            13,
            14,
            15,  # ring
            17,
            18,
            19,  # thumb
        ]

        # Groundtruth
        body_pos = outputs["gt_output"]["global_pos"][0]
        lefthand_pos = outputs["gt_output"]["lefthand_global_pos"][0]
        righthand_pos = outputs["gt_output"]["righthand_global_pos"][0]
        gt_pos = torch.cat([body_pos, lefthand_pos[..., finger_ind, :], righthand_pos[..., finger_ind, :]], dim=-2)
        gt_pos = gt_pos[None]

        foot_sliding = calculate_foot_sliding(gt_pos)
        self.gt_fs_aggregator.update(foot_sliding)

        obj_root_mat = batch["obj"][..., 0, :, :]
        center = batch["center"]
        obj_center = matrix.get_position_from(center[:, None], obj_root_mat)
        obj_center_mat = matrix.get_TRS(matrix.get_rotation(obj_root_mat), obj_center)

        # pred
        if "pred_joint_pos" in outputs:
            pred_pos = outputs["pred_joint_pos"]
            pred_obj_root_mat = outputs["pred_objmat"]
        else:
            body_pos = outputs["global_pos"][0]
            lefthand_pos = outputs["lefthand_global_pos"][0]
            righthand_pos = outputs["righthand_global_pos"][0]
            pred_pos = torch.cat(
                [body_pos, lefthand_pos[..., finger_ind, :], righthand_pos[..., finger_ind, :]], dim=-2
            )
            pred_pos = pred_pos[None]
            pred_obj_root_mat = outputs["obj"]

        foot_sliding = calculate_foot_sliding(pred_pos)
        self.fs_aggregator.update(foot_sliding)

        pred_obj_center = matrix.get_position_from(center[:, None], pred_obj_root_mat)  # (B, L, 3)
        pred_obj_center_mat = matrix.get_TRS(matrix.get_rotation(pred_obj_root_mat), pred_obj_center)  # (B, L, 4, 4)

        pred_invpos1 = matrix.get_relative_position_to(pred_pos, pred_obj_center_mat)
        pred_vec = torch.cat([pred_pos.flatten(-2), pred_invpos1.flatten(-2)], dim=-1)
        pred_vec = torch.cat([pred_vec, pred_obj_center], dim=-1)

        # identtity_mat = matrix.identity_mat(obj_center_mat)
        # obj_center_mat = matrix.get_TRS(matrix.get_rotation(identtity_mat), matrix.get_position(obj_center_mat))
        gt_invpos1 = matrix.get_relative_position_to(gt_pos, obj_center_mat)
        gt_vec = torch.cat([gt_pos.flatten(-2), gt_invpos1.flatten(-2)], dim=-1)
        gt_vec = torch.cat([gt_vec, obj_center], dim=-1)

        # normalize
        pred_ayfz_motion_vec = self.encoder_motion3d(pred_vec)
        gt_ayfz_motion_vec = self.encoder_motion3d(gt_vec)
        # (B, C, L) -> (B, L, C)
        pred_ayfz_motion_vec = pred_ayfz_motion_vec.transpose(1, 2)
        gt_ayfz_motion_vec = gt_ayfz_motion_vec.transpose(1, 2)

        # t2m motion encoder
        m_lens = torch.div(batch["length"], 4, rounding_mode="floor")

        word_embs = batch["word_embs"]
        pos_onehot = batch["pos_onehot"]
        text_length = batch["text_len"]

        # motion length should be sorted in decreasing order for RNN batch forward
        align_m_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        pred_ayfz_motion_vec = pred_ayfz_motion_vec[align_m_idx]
        gt_ayfz_motion_vec = gt_ayfz_motion_vec[align_m_idx]
        m_lens = m_lens[align_m_idx]

        motion_mov = self.t2m_moveencoder(pred_ayfz_motion_vec[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        gt_motion_mov = self.t2m_moveencoder(gt_ayfz_motion_vec[..., :-4]).detach()
        gt_motion_emb = self.t2m_motionencoder(gt_motion_mov, m_lens)

        # t2m text encoder
        # text length should be sorted in decreasing order for RNN batch forward
        align_t_idx = np.argsort(text_length.data.tolist())[::-1].copy()
        word_embs = word_embs[align_t_idx]
        pos_onehot = pos_onehot[align_t_idx]
        text_length = text_length[align_t_idx]
        text_emb = self.t2m_textencoder(word_embs, pos_onehot, text_length)
        # text order convert to motion order
        inverse_align_t_idx = np.argsort(align_t_idx)
        text_emb = text_emb[inverse_align_t_idx][align_m_idx]

        self.text_embeddings.update(text_emb)
        self.gen_motion_embeddings.update(motion_emb)
        self.gt_motion_embeddings.update(gt_motion_emb)

    def on_predict_epoch_start(self, trainer, pl_module):

        self.wis_i = 0

        if not self.is_fid:
            return

        self.text_embeddings.reset()
        self.gen_motion_embeddings.reset()
        self.gt_motion_embeddings.reset()

        # NOTE: not sure whether this way is beautiful
        self.t2m_textencoder = self.t2m_textencoder.to(pl_module.device)
        self.t2m_moveencoder = self.t2m_moveencoder.to(pl_module.device)
        self.t2m_motionencoder = self.t2m_motionencoder.to(pl_module.device)
        self.data_endecoder = self.data_endecoder.to(pl_module.device)

        return

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""

        metrics = {
            "Matching_score": 0.0,
            "gt_Matching_score": 0.0,
            "Diversity": 0.0,
            "gt_Diversity": 0.0,
        }
        for i in range(self.top_k):
            metrics[f"R_precision_top_{str(i + 1)}"] = 0.0
            metrics[f"gt_R_precision_top_{str(i + 1)}"] = 0.0

        count_seq = self.text_embeddings.length()

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = self.text_embeddings.get_tensor()[shuffle_idx]
        all_genmotions = self.gen_motion_embeddings.get_tensor()[shuffle_idx]
        all_gtmotions = self.gt_motion_embeddings.get_tensor()[shuffle_idx]

        device = all_genmotions.device

        # Compute r-precision
        # assert count_seq > self.R_size
        R_N = count_seq // self.R_size
        if R_N == 0:
            R_N = 1
            Log.warn(
                f"Generation metric - Matching_score and R_precision required at least {self.R_size} sequences, "
                f"but only uses {count_seq} sequences to calculate!"
            )
        matching_score = 0.0
        top_k_mat = torch.zeros((self.top_k,), device=device)
        for i in range(R_N):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_genmotions[i * self.R_size : (i + 1) * self.R_size]
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            # print(dist_mat[:5])
            matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        # assert count_seq >= self.R_size
        matching_score = 0.0
        top_k_mat = torch.zeros((self.top_k,), device=device)
        for i in range(R_N):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_gtmotions[i * self.R_size : (i + 1) * self.R_size]
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            # match score
            matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        metrics["gt_Matching_score"] = matching_score / R_count
        for k in range(self.top_k):
            metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.detach().cpu().numpy()
        all_gtmotions = all_gtmotions.detach().cpu().numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        if count_seq > self.diversity_times:
            diversity_times = self.diversity_times
        else:
            diversity_times = count_seq
            Log.warn(
                f"Generation metric - Diversity required {self.diversity_times} sequences, "
                f"but only uses {diversity_times} sequences to calculate!"
            )
        metrics["Diversity"] = calculate_diversity_np(all_genmotions, diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(all_gtmotions, diversity_times)

        metrics["foot_sliding"] = self.fs_aggregator.get_tensor().mean()
        metrics["gt_foot_sliding"] = self.gt_fs_aggregator.get_tensor().mean()

        # log to stdout
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item()
            Log.info(f"{k}: {v:.3f}")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                pl_module.logger.log_metrics({f"val_metric_{self.target_dataset_id}/{k}": v}, step=cur_epoch)

        self.text_embeddings.reset()
        self.gen_motion_embeddings.reset()
        self.gt_motion_embeddings.reset()

        self.t2m_textencoder = self.t2m_textencoder.cpu()
        self.t2m_moveencoder = self.t2m_moveencoder.cpu()
        self.t2m_motionencoder = self.t2m_motionencoder.cpu()
        self.data_endecoder = self.data_endecoder.cpu()


wholebodyobjpose_node = builds(MetricWholeBodyObjPose)
MainStore.store(
    name="metric_grab_wholebodyobjpose",
    node=wholebodyobjpose_node,
    group="callbacks",
    package="callbacks.metric_grab_wholebodyobjpose",
)
