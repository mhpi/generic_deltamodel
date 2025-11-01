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
# TODO: local versus github I guess because lstm on cpu is nondeterministic
EXP_FINAL_LOSS_VALUES = [
    32.135179460048676,  # The GHA runner loss
    32.115299105644226,  # Your local machine loss
]
EXP_NSE_VALUES = [
    -2.989192485809326,  # The GHA runner loss
    -2.989192485809326,  # Your local machine loss
]
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

    is_loss_close = np.isclose(
        trainer.total_loss, EXP_FINAL_LOSS_VALUES, rtol=1e-4, atol=1e-5
    )
    assert np.any(is_loss_close), (
        f"Training loss regression failed. "
        f"Got: {trainer.total_loss}, "
        f"Expected one of: {EXP_FINAL_LOSS_VALUES}"
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
    is_nse_close = np.isclose(actual_nse, EXP_NSE_VALUES, atol=1e-3)
    assert np.any(is_nse_close), (
        f"Evaluation NSE regression failed. "
        f"Got: {actual_nse}, "
        f"Expected one of: {EXP_NSE_VALUES}"
    )

    shutil.rmtree(tmp_path)
