ChromatinCNN Inference Instructions
====================================

Quick Start:
-----------
1. Edit the CONFIG section at the top of phase3_inference.py:
   - Set 'checkpoint_path' to your trained model
   - Set 'test_sequences_csv' to your test data
   - Toggle 'use_rc_tta' (True = better accuracy, False = faster)

2. Run inference:
   python phase3_inference.py

3. Zip predictions for submission:
   python zip_predictions.py

Output:
-------
- predictions.csv: One label per line (1-18), no header
- predictions.zip: Compressed version ready for submission

Configuration Options:
---------------------
CONFIG = {
    'checkpoint_path': './checkpoints_improved/best_model_improved.pt',
    'test_sequences_csv': './data/testsequences.csv',
    'output_csv': 'predictions.csv',
    'batch_size': 512,
    'use_rc_tta': True,  # Reverse complement test-time augmentation
    'device': None,      # Auto-detect, or specify 'cuda'/'mps'/'cpu'
}

Notes:
------
- RC-TTA averages forward and reverse complement predictions for better accuracy
- The script automatically detects model configuration from checkpoint
- Supports all model variants: hierarchical classifier, engineered features, etc.
- Output format matches competition requirements (labels only, no header)
