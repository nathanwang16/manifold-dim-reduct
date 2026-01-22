import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class HierarchicalClassifier(nn.Module):
    def __init__(self, head_in: int, n_classes: int, family_class_indices: list[list[int]]):
        super().__init__()
        self.n_classes = n_classes
        self.family_class_indices = family_class_indices
        self.n_families = len(family_class_indices)
        self.family_head = nn.Linear(head_in, self.n_families)
        self.cond_heads = nn.ModuleList(
            nn.Linear(head_in, len(cls_list)) if len(cls_list) > 0 else None
            for cls_list in family_class_indices
        )

    def forward(self, h: torch.Tensor):
        logits_family = self.family_head(h)
        cond_logits: list[torch.Tensor | None] = []
        combined = h.new_full((h.size(0), self.n_classes), -65000.0)
        for fam_id, cls_list in enumerate(self.family_class_indices):
            head = self.cond_heads[fam_id]
            if head is None or len(cls_list) == 0:
                cond_logits.append(None)
                continue
            fam_cond = head(h)
            cond_logits.append(fam_cond)
            idx = torch.tensor(cls_list, device=h.device, dtype=torch.long)
            fam_bias = logits_family[:, fam_id].unsqueeze(1)
            logits = fam_bias + fam_cond[:, :len(cls_list)]
            combined.index_copy_(1, idx, logits)
        return combined, logits_family, cond_logits


class ConditionalSubclusterHead(nn.Module):
    def __init__(self, head_in: int, family_subcluster_indices: list[list[int]], n_subclusters: int):
        super().__init__()
        self.family_subcluster_indices = family_subcluster_indices
        self.n_subclusters = n_subclusters
        self.sub_heads = nn.ModuleList(
            nn.Linear(head_in, len(sub_ids)) if len(sub_ids) > 0 else None
            for sub_ids in family_subcluster_indices
        )

    def forward(self, h: torch.Tensor):
        logits_global = h.new_full((h.size(0), self.n_subclusters), -65000.0)
        cond_logits: list[torch.Tensor | None] = []
        for fam_id, sub_ids in enumerate(self.family_subcluster_indices):
            head = self.sub_heads[fam_id]
            if head is None or len(sub_ids) == 0:
                cond_logits.append(None)
                continue
            fam_logits = head(h)
            cond_logits.append(fam_logits)
            idx = torch.tensor(sub_ids, device=h.device, dtype=torch.long)
            logits_global.index_copy_(1, idx, fam_logits[:, :len(sub_ids)])
        return logits_global, cond_logits


class EnsembleModel(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        if len(members) < 1:
            raise ValueError("Ensemble must contain at least one member.")
        self.members = nn.ModuleList(members)
        self.ensemble_size = len(members)

    def init_filters_from_data(self, loader):
        for member in self.members:
            member.init_filters_from_data(loader)

    def forward(self, *args, **kwargs):
        outputs = [member(*args, **kwargs) for member in self.members]
        first = outputs[0]
        if isinstance(first, dict):
            stacked = {}
            for key in first.keys():
                values = [out[key] for out in outputs if out[key] is not None]
                if not values:
                    stacked[key] = None
                elif isinstance(values[0], list):
                    merged_list = []
                    for idx in range(len(values[0])):
                        elem_values = [v[idx] for v in values if v[idx] is not None]
                        merged_list.append(torch.stack(elem_values, dim=0) if elem_values else None)
                    stacked[key] = merged_list
                else:
                    stacked[key] = torch.stack(values, dim=0)
            return stacked
        return torch.stack(outputs, dim=0)


class ChromatinCNNAttention(nn.Module):
    """
    Improved Architecture with Data-Driven Initialization Support.
    """
    def __init__(
        self,
        n_classes=18,
        input_len=200,
        n_families: int = 0,
        n_subclusters: int = 0,
        use_hierarchy: bool = False,
        use_engineered_features: bool = False,
        engineered_feature_dim_in: int = 5,
        feature_dim: int = 128,
        use_hierarchical_classifier: bool = False,
        family_class_indices: list[list[int]] | None = None,
        family_subcluster_indices: list[list[int]] | None = None,
        family_embed_dim: int = 0,
    ):
        super().__init__()
        self.use_hierarchy = use_hierarchy
        self.use_engineered_features = use_engineered_features
        self.use_hierarchical_classifier = use_hierarchical_classifier
        self.n_classes = n_classes
        self.n_families = n_families
        self.n_subclusters = n_subclusters
        self.engineered_feature_dim_in = engineered_feature_dim_in
        self.feature_dim = feature_dim
        self.family_class_indices = family_class_indices
        self.family_subcluster_indices = family_subcluster_indices

        # 1. Stem (Motif Discovery)
        # Use bias=True to allow shifting the baseline
        self.stem_conv = nn.Conv1d(4, 128, kernel_size=19, padding='same', bias=True)
        self.stem_bn = nn.BatchNorm1d(128)

        # 2. Residual Tower
        self.res_block1 = ResidualBlock(128, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2) # 200 -> 100

        self.conv_expand = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.res_block2 = ResidualBlock(256, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2) # 100 -> 50

        # 3. Attention
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.norm_attn = nn.LayerNorm(256)

        # 4. Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.feature_mlp = None
        if self.use_engineered_features:
            self.feature_mlp = nn.Sequential(
                nn.Linear(engineered_feature_dim_in, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, feature_dim),
                nn.ReLU(),
            )

        base_head_in = 128 + (feature_dim if self.use_engineered_features else 0)
        self.family_context_linear = (
            nn.Linear(base_head_in, n_families) if (self.use_hierarchy and n_families > 0) else None
        )

        self.family_embed_dim = family_embed_dim if (self.use_hierarchy and n_families > 0 and family_embed_dim > 0) else 0
        self.family_embed = (
            nn.Embedding(n_families, self.family_embed_dim) if self.family_embed_dim > 0 else None
        )

        head_in = base_head_in + self.family_embed_dim

        self.hier_classifier = None
        self.class_head = None

        if self.use_hierarchical_classifier:
            if not self.use_hierarchy:
                raise ValueError("use_hierarchical_classifier=True requires use_hierarchy=True.")
            if self.family_class_indices is None:
                raise ValueError("use_hierarchical_classifier=True requires family_class_indices.")
            self.hier_classifier = HierarchicalClassifier(head_in, n_classes, self.family_class_indices)
        else:
            self.class_head = nn.Linear(head_in, n_classes)

        if self.use_hierarchy and n_subclusters > 0:
            if self.family_subcluster_indices is None:
                raise ValueError("Conditional subcluster head requires family_subcluster_indices.")
            self.subcluster_head = ConditionalSubclusterHead(head_in, self.family_subcluster_indices, n_subclusters)
        else:
            self.subcluster_head = None

    def forward(self, x, engineered: torch.Tensor | None = None):
        # x: (B, 200, 4) or (B, 4, 200)
        if x.shape[1] == 200:
            x = x.transpose(1, 2)

        # Stem
        x = F.relu(self.stem_bn(self.stem_conv(x)))

        # Tower
        x = self.res_block1(x)
        x = self.pool1(x)

        x = F.relu(self.conv_expand(x))
        x = self.res_block2(x)
        x = self.pool2(x) # (B, 256, 50)

        # Attention (Permute to B, L, C)
        x_perm = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x_perm, x_perm, x_perm)
        x_perm = self.norm_attn(x_perm + attn_out)
        x = x_perm.permute(0, 2, 1)

        # Classifier
        x = self.global_pool(x).squeeze(-1)
        h = self.shared_mlp(x)
        if self.use_engineered_features:
            if engineered is None:
                raise ValueError("use_engineered_features=True requires engineered features tensor.")
            h_feat = self.feature_mlp(engineered)
            h = torch.cat([h, h_feat], dim=1)

        logits_family_context = None
        if self.family_context_linear is not None:
            logits_family_context = self.family_context_linear(h)
            if self.family_embed is not None:
                fam_probs = torch.softmax(logits_family_context, dim=-1)
                fam_context = fam_probs @ self.family_embed.weight
                h = torch.cat([h, fam_context], dim=1)

        family_cond_logits = None
        if self.use_hierarchical_classifier:
            logits_class, logits_family, family_cond_logits = self.hier_classifier(h)
        else:
            logits_class = self.class_head(h)
            logits_family = logits_family_context

        logits_subcluster = None
        sub_cond_logits = None
        if self.subcluster_head is not None:
            logits_subcluster, sub_cond_logits = self.subcluster_head(h)

        if not self.use_hierarchy:
            return logits_class

        return {
            "logits_class": logits_class,
            "logits_family": logits_family,
            "logits_subcluster": logits_subcluster,
            "family_cond_logits": family_cond_logits,
            "sub_cond_logits": sub_cond_logits,
        }

    def init_filters_from_data(self, sequences_loader, n_filters=128, kernel_size=19):
        import random
        """
        MI Technique: Initialize filters with random real data chunks.
        """
        print(f"\n[MI Fix] Initializing {n_filters} filters from training data chunks...")

        # Get a batch of data
        sequences_batch = next(iter(sequences_loader))
        if isinstance(sequences_batch, list) or isinstance(sequences_batch, tuple):
            sequences_batch = sequences_batch[0] # Get just the sequences

        # Ensure (B, 4, 200)
        if sequences_batch.shape[1] == 200:
            sequences_batch = sequences_batch.transpose(1, 2)

        n_seqs, _, seq_len = sequences_batch.shape

        with torch.no_grad():
            for i in range(n_filters):
                # Sample random sequence and random position
                seq_idx = random.randint(0, n_seqs - 1)
                pos_idx = random.randint(0, seq_len - kernel_size)

                # Extract chunk (4, 19)
                chunk = sequences_batch[seq_idx, :, pos_idx:pos_idx+kernel_size]

                # Set weights: Add slight noise to break perfect symmetry if any
                self.stem_conv.weight[i] = chunk + torch.randn_like(chunk) * 0.01

        print("[MI Fix] Data-driven initialization complete. Filters now look like DNA.")
