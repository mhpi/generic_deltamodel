"""Regression test for the training and evaluation process."""

import sys
from pathlib import Path
import os
import shutil

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from dmg.trainers.trainer import Trainer
from dmg.models.model_handler import ModelHandler
from dmg.core.utils import set_randomseed, initialize_config


def test_training_regression(config, mock_dataset, tmp_path):
    """
    Tests the full training and evaluation pipeline for reproducibility.
    """
    # Use a temporary directory for outputs
    config['output_dir'] = str(tmp_path / 'output')
    config['model_dir'] = str(tmp_path / 'model')
    os.makedirs(config['model_dir'], exist_ok=True)

    # Set a fixed random seed
    set_randomseed(config['seed'])

    # Initialize model and trainer
    config = initialize_config(config)
    model = ModelHandler(config)
    trainer = Trainer(
        config,
        model,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
    )

    # --- Training ---
    trainer.train()

    # Expected loss after 2 epochs
    # This value is obtained by running the test once and recording the output.
    # If you change the model, data, or training process, this value will need
    # to be updated.
    expected_final_loss = 2.001859664916992
    assert np.isclose(trainer.total_loss, expected_final_loss, atol=1e-5), (
        f"Training loss regression failed. Expected: {expected_final_loss}, Got: {trainer.total_loss}"
    )

    # --- Evaluation ---
    # Load the trained model for evaluation
    config['mode'] = 'test'
    config['test']['test_epoch'] = 2
    model_eval = ModelHandler(config)

    trainer_eval = Trainer(config, model_eval, eval_dataset=mock_dataset)
    trainer_eval.evaluate()

    # Expected evaluation metrics
    # These are also obtained by running the test once and recording the output.
    expected_nse = -0.923
    metrics_path = Path(config['output_dir']) / 'metrics_agg.json'
    import json

    with open(metrics_path) as f:
        metrics = json.load(f)

    actual_nse = metrics['nse']['median']
    assert np.isclose(actual_nse, expected_nse, atol=1e-3), (
        f"Evaluation NSE regression failed. Expected: {expected_nse}, Got: {actual_nse}"
    )

    # Clean up the temporary directory
    shutil.rmtree(tmp_path)
