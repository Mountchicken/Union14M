# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Tuple
import timm.models.vision_transformer
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn as nn
import math
from mmocr.registry import MODELS


@MODELS.register_module()
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer.

    Args:
        global_pool (bool): If True, apply global pooling to the output
            of the last stage. Default: False.
        patch_size (int): Patch token size. Default: 8.
        img_size (tuple[int]): Input image size. Default: (32, 128).
        embed_dim (int): Number of linear projection output channels.
            Default: 192.
        depth (int): Number of blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 3.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key,
            value. Default: True.
        norm_layer (nn.Module): Normalization layer. Default:
            partial(nn.LayerNorm, eps=1e-6).
        pretrained (str): Path to pre-trained checkpoint. Default: None.
    """

    def __init__(self,
                 global_pool: bool = False,
                 patch_size: int = 8,
                 img_size: Tuple[int, int] = (32, 128),
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained: bool = None,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            **kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.reset_classifier(0)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % pretrained)
            checkpoint_model = checkpoint['model']
            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[
                        k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            msg = self.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_features(x)


class _LoRA_qkv_timm(nn.Module):
    """LoRA layer for query and value projection in Vision Transformer of timm.

    Args:
        qkv (nn.Module): qkv projection layer in Vision Transformer of timm.
        linear_a_q (nn.Module): Linear layer for query projection.
        linear_b_q (nn.Module): Linear layer for query projection.
        linear_a_v (nn.Module): Linear layer for value projection.
        linear_b_v (nn.Module): Linear layer for value projection.
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, 3*dim
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


@MODELS.register_module()
class VisionTransformer_LoRA(nn.Module):
    """Vision Transformer with LoRA. For each block, we add a LoRA layer for
    the linear projection of query and value.

    Args:
        vit_config (dict): Config dict for VisionTransformer.
        rank (int): Rank of LoRA layer. Default: 4.
        lora_layers (int): Stages to add LoRA layer. Defaults None means
            add LoRA layer to all stages.
        pretrained_lora (str): Path to pre-trained checkpoint of LoRA layer.
    """

    def __init__(self,
                 vit_config: dict,
                 rank: int = 4,
                 lora_layers: int = None,
                 pretrained_lora: str = None):
        super(VisionTransformer_LoRA, self).__init__()
        self.vit = VisionTransformer(**vit_config)
        assert rank > 0
        if lora_layers:
            self.lora_layers = lora_layers
        else:
            self.lora_layers = list(range(len(self.vit.blocks)))
        # creat list of LoRA layers
        self.query_As = nn.Sequential()  # matrix A for query linear projection
        self.query_Bs = nn.Sequential()
        self.value_As = nn.Sequential()  # matrix B for value linear projection
        self.value_Bs = nn.Sequential()

        # freeze the original vit
        for param in self.vit.parameters():
            param.requires_grad = False

        # compose LoRA layers
        for block_idx, block in enumerate(self.vit.blocks):
            if block_idx not in self.lora_layers:
                continue
            # create LoRA layer
            w_qkv_linear = block.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, self.dim, bias=False)
            self.query_As.append(w_a_linear_q)
            self.query_Bs.append(w_b_linear_q)
            self.value_As.append(w_a_linear_v)
            self.value_Bs.append(w_b_linear_v)
            # replace the original qkv layer with LoRA layer
            block.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self._init_lora()
        if pretrained_lora is not None:
            self._load_lora(pretrained_lora)

    def _init_lora(self):
        """Initialize the LoRA layers to be identity mapping."""
        for query_A, query_B, value_A, value_B in zip(self.query_As,
                                                      self.query_Bs,
                                                      self.value_As,
                                                      self.value_Bs):
            nn.init.kaiming_uniform_(query_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(value_A.weight, a=math.sqrt(5))
            nn.init.zeros_(query_B.weight)
            nn.init.zeros_(value_B.weight)

    def _load_lora(self, checkpoint_lora: str):
        """Load pre-trained LoRA checkpoint.

        Args:
            checkpoint_lora (str): Path to pre-trained LoRA checkpoint.
        """
        assert checkpoint_lora.endswith(".safetensors")
        with safe_open(checkpoint_lora, framework="pt") as f:
            for i, q_A, q_B, v_A, v_B in zip(
                    range(len(self.query_As)),
                    self.query_As,
                    self.query_Bs,
                    self.value_As,
                    self.value_Bs,
            ):
                q_A.weight = nn.Parameter(f.get_tensor(f"q_a_{i:03d}"))
                q_B.weight = nn.Parameter(f.get_tensor(f"q_b_{i:03d}"))
                v_A.weight = nn.Parameter(f.get_tensor(f"v_a_{i:03d}"))
                v_B.weight = nn.Parameter(f.get_tensor(f"v_b_{i:03d}"))

    def forward(self, x):
        x = self.vit(x)
        return x


def extract_lora_from_vit(checkpoint_path: str,
                          save_path: str,
                          ckpt_key: str = None):
    """Given a checkpoint of VisionTransformer_LoRA, extract the LoRA weights
    and save them to a new checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint of VisionTransformer_LoRA.
        ckpt_key (str): Key of model in the checkpoint.
        save_path (str): Path to save the extracted LoRA checkpoint.
    """
    assert save_path.endswith(".safetensors")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # travel throung the ckpt to find the LoRA layers
    query_As = []
    query_Bs = []
    value_As = []
    value_Bs = []
    ckpt = ckpt if ckpt_key is None else ckpt[ckpt_key]
    for k, v in ckpt.items():
        if k.startswith("query_As"):
            query_As.append(v)
        elif k.startswith("query_Bs"):
            query_Bs.append(v)
        elif k.startswith("value_As"):
            value_As.append(v)
        elif k.startswith("value_Bs"):
            value_Bs.append(v)
    # save the LoRA layers to a new checkpoint
    ckpt_dict = {}
    for i in range(len(query_As)):
        ckpt_dict[f"q_a_{i:03d}"] = query_As[i]
        ckpt_dict[f"q_b_{i:03d}"] = query_Bs[i]
        ckpt_dict[f"v_a_{i:03d}"] = value_As[i]
        ckpt_dict[f"v_b_{i:03d}"] = value_Bs[i]
    save_file(ckpt_dict, save_path)
