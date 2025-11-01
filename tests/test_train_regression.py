"""Regression test for the training and evaluation process.

NOTE: We can only evaluate CPU-bound models due to constraint of GitHub.
"""

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import torch
import os
import numpy as np

from dmg.core.utils import set_randomseed
from dmg.models.model_handler import ModelHandler
from dmg.trainers.trainer import Trainer

# --- Expected Train Loss + Test NSE  ---
# NOTE: If you change the model, data, or training process, these values must
# need to be updated.
EXP_FINAL_LOSS = 32.135179460048676
EXP_NSE = -2.989192485809326
# --------------------------------------------


def test_training_regression(config, mock_dataset, tmp_path):
    """
    Tests the full training and evaluation pipeline for reproducibility.
    """
    set_randomseed(config['seed'])

    # 2. Force PyTorch to use deterministic (slower) algorithms
    try:
        print("Attempting to set deterministic algorithms...")
        torch.use_deterministic_algorithms(True)
        # This is for CUDA, but good to include.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except RuntimeError:
        print("Warning: Could not set deterministic algorithms.")

    model = ModelHandler(config)
    trainer = Trainer(
        config,
        model,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
        write_out=True,
    )

    # --- Training ---
    trainer.train()

    assert np.isclose(trainer.total_loss, EXP_FINAL_LOSS, atol=1e-5), (
        f"Training loss regression failed. Expected: {EXP_FINAL_LOSS}, Got: {trainer.total_loss}"
    )

    # --- Evaluation ---
    config['mode'] = 'test'
    config['test']['test_epoch'] = 2
    model_eval = ModelHandler(config)

    trainer_eval = Trainer(config, model_eval, eval_dataset=mock_dataset)
    trainer_eval.evaluate()

    metrics_path = Path(config['output_dir']) / 'metrics_agg.json'
    import json

    with open(metrics_path) as f:
        metrics = json.load(f)

    actual_nse = metrics['nse']['median']
    assert np.isclose(actual_nse, EXP_NSE, atol=1e-3), (
        f"Evaluation NSE regression failed. Expected: {EXP_NSE}, Got: {actual_nse}"
    )

    shutil.rmtree(tmp_path)
