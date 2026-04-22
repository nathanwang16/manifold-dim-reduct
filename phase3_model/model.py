"""Phase 3 model: `ChromatinCNNAttentionV2`.

Keeps the interpretability-friendly stem+residual+attention+bottleneck shape
from the previous iteration but scales up for the 1M-row roadmap_18state_full
training set:

    Stem:             Conv1d(4 -> 256, k=19, padding='same')
                      BatchNorm + ReLU (data-driven filter init supported)
    ResBlock1:        Conv(k=7), Conv(k=7) @ 256ch  -> MaxPool(2)     # 200->100
    Conv expand:      Conv(k=5) 256 -> 384                             # 100
    ResBlock2:        Conv(k=5), Conv(k=5) @ 384ch -> MaxPool(2)      # 100->50
    Conv expand:      Conv(k=3) 384 -> 512                             # 50
    ResBlock3:        Conv(k=3), Conv(k=3) @ 512ch                    # 50
    Attention:        2 x MultiheadAttention (512d, 8 heads)
    GAP:              (B, 512)
    Bottleneck:       Linear(512 + 5_engineered? -> 384) + ReLU + Dropout
    Heads:
        - class_head:       Linear(384 -> 18)
        - family_head:      Linear(384 -> 6)      (if use_hierarchy)
        - subcluster_head:  Linear(384 -> 7)      (if use_hierarchy)

Bottleneck activations (384-d) are the canonical MI extraction point used by
phases 4 (SAE), 5 (patching), 6 (steering), 7 (diagnostics), 8 (motifs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_classes: int = 18
    sequence_length: int = 200
    stem_filters: int = 256
    stem_kernel: int = 19
    mid_filters: int = 384
    wide_filters: int = 512
    attn_heads: int = 8
    attn_layers: int = 2
    bottleneck_dim: int = 384
    dropout: float = 0.25
    use_engineered_features: bool = True
    engineered_feature_dim_in: int = 5
    engineered_feature_dim_out: int = 64
    use_hierarchy: bool = True
    n_families: int = 6
    n_subclusters: int = 7
    extra: Dict[str, Any] = field(default_factory=dict)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + res)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.mlp(x)))
        return x


class ChromatinCNNAttentionV2(nn.Module):
    """Motif → residual tower → transformer → bottleneck → {class, family, subcluster} heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Stem (motif scanning)
        self.stem_conv = nn.Conv1d(4, cfg.stem_filters, kernel_size=cfg.stem_kernel,
                                   padding="same", bias=True)
        self.stem_bn = nn.BatchNorm1d(cfg.stem_filters)

        # Residual tower
        self.res_block1 = ResidualBlock(cfg.stem_filters, kernel_size=7, dropout=cfg.dropout * 0.5)
        self.pool1 = nn.MaxPool1d(2)  # 200 -> 100

        self.conv_expand1 = nn.Conv1d(cfg.stem_filters, cfg.mid_filters, kernel_size=5,
                                      padding="same")
        self.bn_expand1 = nn.BatchNorm1d(cfg.mid_filters)
        self.res_block2 = ResidualBlock(cfg.mid_filters, kernel_size=5, dropout=cfg.dropout * 0.5)
        self.pool2 = nn.MaxPool1d(2)  # 100 -> 50

        self.conv_expand2 = nn.Conv1d(cfg.mid_filters, cfg.wide_filters, kernel_size=3,
                                      padding="same")
        self.bn_expand2 = nn.BatchNorm1d(cfg.wide_filters)
        self.res_block3 = ResidualBlock(cfg.wide_filters, kernel_size=3, dropout=cfg.dropout * 0.5)

        # Transformer blocks (capture 200bp-scale context)
        self.attn_blocks = nn.ModuleList([
            TransformerBlock(cfg.wide_filters, cfg.attn_heads, dropout=cfg.dropout)
            for _ in range(cfg.attn_layers)
        ])

        # Global avg pool + bottleneck
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        bottleneck_in = cfg.wide_filters
        self.feature_mlp: Optional[nn.Module] = None
        if cfg.use_engineered_features:
            self.feature_mlp = nn.Sequential(
                nn.Linear(cfg.engineered_feature_dim_in, cfg.engineered_feature_dim_out),
                nn.ReLU(),
                nn.Dropout(cfg.dropout * 0.5),
            )
            bottleneck_in = cfg.wide_filters + cfg.engineered_feature_dim_out

        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_in, cfg.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # Heads
        self.class_head = nn.Linear(cfg.bottleneck_dim, cfg.n_classes)
        self.family_head = (
            nn.Linear(cfg.bottleneck_dim, cfg.n_families) if cfg.use_hierarchy else None
        )
        self.subcluster_head = (
            nn.Linear(cfg.bottleneck_dim, cfg.n_subclusters) if cfg.use_hierarchy else None
        )

    # ------------------------------------------------------------------
    # Forward path: returns dict of logits + intermediate activations
    # ------------------------------------------------------------------

    def _trunk(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, L, 4) -> (B, 4, L)
        if x.shape[-1] == 4 and x.dim() == 3:
            x = x.transpose(1, 2)
        acts: Dict[str, torch.Tensor] = {}

        h = F.relu(self.stem_bn(self.stem_conv(x)))
        acts["stem"] = h

        h = self.res_block1(h)
        acts["res_block1"] = h
        h = self.pool1(h)

        h = F.relu(self.bn_expand1(self.conv_expand1(h)))
        acts["conv_expand1"] = h
        h = self.res_block2(h)
        acts["res_block2"] = h
        h = self.pool2(h)

        h = F.relu(self.bn_expand2(self.conv_expand2(h)))
        acts["conv_expand2"] = h
        h = self.res_block3(h)
        acts["res_block3"] = h  # (B, wide, 50)

        # Transformer
        x_perm = h.permute(0, 2, 1)  # (B, 50, wide)
        for i, blk in enumerate(self.attn_blocks):
            x_perm = blk(x_perm)
            acts[f"attn_block{i}"] = x_perm.permute(0, 2, 1)
        h = x_perm.permute(0, 2, 1)
        acts["attn_out"] = h

        pooled = self.global_pool(h).squeeze(-1)
        acts["pooled"] = pooled
        return acts

    def forward(
        self,
        x: torch.Tensor,
        engineered: Optional[torch.Tensor] = None,
        *,
        return_activations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        acts = self._trunk(x)
        pooled = acts["pooled"]
        if self.feature_mlp is not None:
            if engineered is None:
                raise ValueError("Model built with use_engineered_features=True but got None.")
            feat = self.feature_mlp(engineered)
            pooled = torch.cat([pooled, feat], dim=1)
            acts["pooled_with_feat"] = pooled

        bn = self.bottleneck(pooled)
        acts["bottleneck"] = bn

        logits_class = self.class_head(bn)
        out: Dict[str, torch.Tensor] = {"logits": logits_class, "bottleneck": bn}
        if self.family_head is not None:
            out["logits_family"] = self.family_head(bn)
        if self.subcluster_head is not None:
            out["logits_subcluster"] = self.subcluster_head(bn)

        if return_activations:
            out["activations"] = acts
        return out

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_stem_from_data(self, sequences_onehot: torch.Tensor) -> None:
        """Seed stem filters with random DNA chunks (mech-interp trick).

        `sequences_onehot` is a (N, 4, L) float tensor on any device.
        """
        import random
        n, c, length = sequences_onehot.shape
        if c != 4:
            raise ValueError("Expected channel-first one-hot (N, 4, L).")
        k = self.stem_conv.weight.shape[-1]
        device = self.stem_conv.weight.device
        for i in range(self.stem_conv.weight.shape[0]):
            idx = random.randint(0, n - 1)
            pos = random.randint(0, length - k)
            chunk = sequences_onehot[idx, :, pos : pos + k].to(device)
            noise = torch.randn_like(chunk) * 0.01
            self.stem_conv.weight[i] = chunk + noise


def build_model(cfg: Dict[str, Any]) -> ChromatinCNNAttentionV2:
    """Construct a ChromatinCNNAttentionV2 from a `config.json`-style dict.

    Accepts either a full config dict (containing `phase3` and `dataset`) or
    a flat dict with the relevant keys.
    """
    if "phase3" in cfg:
        p3 = cfg["phase3"]
        data = cfg.get("dataset", {})
    else:
        p3 = cfg
        data = {}
    mc = ModelConfig(
        n_classes=int(data.get("n_classes", p3.get("n_classes", 18))),
        sequence_length=int(data.get("sequence_length", 200)),
        stem_filters=int(p3.get("stem_filters", 256)),
        stem_kernel=int(p3.get("stem_kernel", 19)),
        mid_filters=int(p3.get("mid_filters", 384)),
        wide_filters=int(p3.get("wide_filters", 512)),
        attn_heads=int(p3.get("attn_heads", 8)),
        attn_layers=int(p3.get("attn_layers", 2)),
        bottleneck_dim=int(p3.get("bottleneck_dim", 384)),
        dropout=float(p3.get("dropout", 0.25)),
        use_engineered_features=bool(p3.get("use_engineered_features", True)),
        engineered_feature_dim_in=int(p3.get("engineered_feature_dim_in", 5)),
        engineered_feature_dim_out=int(p3.get("engineered_feature_out_dim", 64)),
        use_hierarchy=bool(p3.get("use_hierarchy", True)),
        n_families=int(data.get("n_families", 6)),
        n_subclusters=int(data.get("n_subclusters", 7)),
    )
    return ChromatinCNNAttentionV2(mc)
