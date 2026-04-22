"""Phase 3: interpretable CNN + attention classifier.

Public surface:
    build_model(cfg)                      -- construct model from config.json
    ChromatinCNNAttentionV2, ModelConfig  -- classes
    train / train_ddp.main                -- DDP training entrypoint
    precompute_cache.build_cache          -- pre-bake one-hot/feature caches
"""

from .model import ChromatinCNNAttentionV2, ModelConfig, build_model

__all__ = ["ChromatinCNNAttentionV2", "ModelConfig", "build_model"]
