import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from coda.configs import MainStore, builds

from coda.network.base_arch.transformer.encoder import EncoderBlockDIT
from coda.network.base_arch.embeddings.pos_emb import PositionalEncoding, TimestepEmbedder
from coda.network.base_arch.transformer.layer import zero_module

from coda.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


class NetworkEncoder(nn.Module):
    def __init__(
        self,
        # x
        output_dim=46,  # 3 * 12 + 10
        max_len=300,
        # condition
        objtraj_dim=19,  # 6+12+1
        obj_dim=0,
        beta_dim=16,
        skeleton_dim=0,
        bps_dim=0,
        clip_dim=0,
        verts_dim=0,
        first_x_dim=0,
        contact_mask_dim=0,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # training
        dropout=0.1,
        is_dit=False,
    ):
        super().__init__()

        # input
        self.output_dim = output_dim
        self.max_len = max_len

        # condition
        self.objtraj_dim = objtraj_dim
        self.obj_dim = obj_dim
        self.beta_dim = beta_dim
        self.skeleton_dim = skeleton_dim
        self.bps_dim = bps_dim
        self.verts_dim = verts_dim
        self.clip_dim = clip_dim
        self.first_x_dim = first_x_dim
        self.contact_mask_dim = contact_mask_dim
        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.is_dit = is_dit
        assert self.is_dit

        # ===== build model ===== #
        self._build_condition_embedder()

        block_cls = EncoderBlockDIT

        # Transformer
        self.blocks = nn.ModuleList(
            [
                block_cls(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.input_layer = nn.Sequential(
            nn.Linear(self.output_dim, latent_dim),
            nn.SiLU(),
        )
        if self.objtraj_dim > 0:
            self.objtraj_embedder = nn.Sequential(
                nn.Linear(self.objtraj_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.obj_dim > 0:
            self.obj_embedder = nn.Sequential(
                nn.Linear(self.obj_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.beta_dim > 0:
            self.beta_embedder = nn.Sequential(
                nn.Linear(self.beta_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.skeleton_dim > 0:
            self.skeleton_embedder = nn.Sequential(
                nn.Linear(self.skeleton_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.clip_dim > 0:
            self.clip_embedder = nn.Sequential(
                nn.Linear(self.clip_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.bps_dim > 0:
            self.bps_embedder = nn.Sequential(
                nn.Linear(self.bps_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.verts_dim > 0:
            self.verts_embedder = nn.Sequential(
                nn.Linear(self.verts_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.first_x_dim > 0:
            self.first_x_embedder = nn.Sequential(
                nn.Linear(self.first_x_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )
        if self.contact_mask_dim > 0:
            self.contact_mask_embedder = nn.Sequential(
                nn.Linear(self.contact_mask_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim), nozero=self.is_dit),
            )

    def forward(
        self,
        x,
        timesteps,
        length,
        obj=None,
        objtraj=None,
        beta=None,
        skeleton=None,
        bps=None,
        verts=None,
        f_text=None,
        first_x=None,
        contact_mask=None,
        **kwargs
    ):
        """
        Args:
            length: (B), valid length of x, if None then use x.shape[2]
            x: (B, L, C)
            timesteps: (B) int
            obj: (B, C)
            objtraj: (B, L, C)
            beta: (B, L, C)
            bps: (B, L, C)
            verts: (B, L, C)
            f_text: (B, D)
            first_x: (B, 1, C)
        """
        B, L, _ = x.shape

        emb = self.embed_timestep(timesteps, is_BLC=True)  # (B, 1, C)
        x = self.input_layer(x)  # (B, L, C)
        x = x + emb

        # Condition
        f_to_add = []
        if obj is not None and hasattr(self, "obj_embedder"):
            f_to_add.append(self.obj_embedder(obj))
        if objtraj is not None and hasattr(self, "objtraj_embedder"):
            f_to_add.append(self.objtraj_embedder(objtraj))
        if beta is not None and hasattr(self, "beta_embedder"):
            f_to_add.append(self.beta_embedder(beta))
        if bps is not None and hasattr(self, "bps_embedder"):
            f_to_add.append(self.bps_embedder(bps))
        if f_text is not None and hasattr(self, "clip_embedder"):
            f_to_add.append(self.clip_embedder(f_text)[..., None, :])
        if verts is not None and hasattr(self, "verts_embedder"):
            f_to_add.append(self.verts_embedder(verts))
        if skeleton is not None and hasattr(self, "skeleton_embedder"):
            f_to_add.append(self.skeleton_embedder(skeleton))
        if first_x is not None and hasattr(self, "first_x_embedder"):
            first_x_emb = self.first_x_embedder(first_x)
            # x = torch.cat([x[:, :1] + first_x_emb, x[:, 1:]], dim=1)
            f_to_add.append(first_x_emb)
        if contact_mask is not None and hasattr(self, "contact_mask_embedder"):
            f_to_add.append(self.contact_mask_embedder(contact_mask))

        if self.is_dit:
            c = torch.zeros_like(x)
            for f_delta in f_to_add:
                c = c + f_delta
        else:
            for f_delta in f_to_add:
                x = x + f_delta

        x = self.sequence_pos_encoder(x, is_BLC=True)

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = length_to_mask(length, L)  # (B, L)

        attnmask = None

        # Transformer
        for block in self.blocks:
            if self.is_dit:
                x = block(x, c=c, attn_mask=attnmask, tgt_key_padding_mask=~pmask)
            else:
                x = block(x, attn_mask=attnmask, tgt_key_padding_mask=~pmask)

        # Output
        sample = self.final_layer(x)  # (B, L, C)

        output = {
            "pred_context": x,
            "sample": sample,
            "mask": pmask,
        }
        return output
