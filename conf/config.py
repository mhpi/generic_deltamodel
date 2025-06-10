"""Validation of basic configuration file parameters using Pydantic.

Run this script to validate an example config object (see bottom of file).
"""
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Union

from pydantic import (BaseModel, Field, ValidationError, field_validator,
                      model_validator)

log = logging.getLogger(__name__)


def check_path(v: str) -> Path:
    """Checks if a given path exists and is valid."""
    path = Path(v)
    if not path.exists():
        log_str = f"Path '{v}' does not exist."
        log.error(log_str)
        raise ValueError(log_str)
    return path


class ModeEnum(str, Enum):
    """Enumeration for different modes of operation."""
    train = 'train'
    test = 'test'
    train_test = 'train_test'
    simulation = 'simulation'


class LossFunctionConfig(BaseModel):
    """Configuration class for loss functions."""
    model: str = Field(..., description="The name of the loss function.")
    weight: float = Field(default=1.0, description="Weight of the loss function.")


class TrainingConfig(BaseModel):
    """Configuration class for training."""
    start_time: str
    end_time: str
    target: list[str]
    optimizer: str
    batch_size: int
    epochs: int
    start_epoch: int = 0
    save_epoch: int = 5
    learning_rate: float = Field(gt=0, description="Learning rate for training.")

    @model_validator(mode='after')
    def validate_training_times(cls, values):
        """Validates the training start and end times."""
        start_time = datetime.strptime(values.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(values.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Training `start_time` must be earlier than `end_time`.")
        return values

    @field_validator('target')
    @classmethod
    def validate_targets(cls, v: list[str]) -> list[str]:
        """Validates the training target."""
        if not v:
            raise ValueError("Training target cannot be empty.")
        return v


class TestingConfig(BaseModel):
    """Configuration class for testing."""
    start_time: str
    end_time: str
    batch_size: int
    test_epoch: int

    @model_validator(mode='after')
    def validate_testing_times(cls, values):
        """Validates the testing start and end times."""
        start_time = datetime.strptime(values.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(values.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Testing start_time must be earlier than end_time.")
        return values


class PhyModelConfig(BaseModel):
    """Configuration class for the physics-based model."""
    model: list[str]
    nmul: int
    warm_up: int
    dynamic_params: dict[str, list[str]]
    forcings: list[str] = Field(default_factory=list, description="List of dynamic input variables.")
    attributes: list[str] = Field(default_factory=list, description="List of static input variables.")


class NeuralNetworkModelConfig(BaseModel):
    """Configuration class for the neural network model."""
    model: str = Field(..., description="The type of neural network model (e.g., LSTM).")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate.")
    hidden_size: int = Field(..., gt=0, description="Number of hidden units.")
    learning_rate: float = Field(..., gt=0, description="Learning rate.")
    forcings: list[str] = Field(default_factory=list, description="List of dynamic input variables.")
    attributes: list[str] = Field(default_factory=list, description="List of static input variables.")


class DeltaModelConfig(BaseModel):
    """Configuration class for the DPL model."""
    rho: int
    phy_model: PhyModelConfig
    nn_model: NeuralNetworkModelConfig


class ObservationConfig(BaseModel):
    """Configuration class for observations."""
    name: str = 'not_defined'
    train_path: str = 'not_defined'
    test_path: str = 'not_defined'
    start_time: str = 'not_defined'
    end_time_all: str = 'not_defined'
    all_forcings: list[str] = Field(default_factory=list, description="List of dynamic input variables.")
    all_attributes: list[str] = Field(default_factory=list, description="List of static input variables.")

    @field_validator('train_path', 'test_path')
    @classmethod
    def validate_dir(cls, v: str) -> Union[Path, str]:
        """Validates the directory paths."""
        if v == 'not_defined':
            return v
        return check_path(v)

    @model_validator(mode='after')
    def validate_dataset_times(cls, values):
        """Validates the dataset start and end times."""
        if values.start_time_all == 'not_defined' and values.end_time_all == 'not_defined':
            return values
        elif values.start_time_all == 'not_defined' or values.end_time_all == 'not_defined':
            raise ValueError("Both train_path and test_path must be defined if either is specified.")
        else:
            start_time = datetime.strptime(values.start_time_all, '%Y/%m/%d')
            end_time = datetime.strptime(values.end_time_all, '%Y/%m/%d')
            if start_time >= end_time:
                raise ValueError("Dataset start time must be earlier than end time.")
            return values

class Config(BaseModel):
    """Configuration class for the model."""
    mode: ModeEnum = Field(default=ModeEnum.train_test)
    do_tune: bool = False
    multimodel_type: str = 'none'
    random_seed: int = 0
    device: str = 'cpu'
    gpu_id: int = 0
    data_loader: str = 'none'
    data_sampler: str = 'none'
    trainer: str = 'none'
    save_path: str
    train: TrainingConfig
    test: TestingConfig
    loss_function: LossFunctionConfig
    delta_model: DeltaModelConfig
    observations: ObservationConfig

    @field_validator('save_path')
    @classmethod
    def validate_save_path(cls, v: str) -> Path:
        """Validates the save_path directory."""
        path = Path(v)
        if not path.exists():
            log_str = f"Save path '{v}' does not exist."
            log.error(log_str)
            raise ValueError(log_str)
        if not path.is_dir():
            log_str = f"Save path '{v}' is not a directory."
            log.error(log_str)
            raise ValueError(log_str)
        if not os.access(path, os.W_OK):
            log_str = f"Save path '{v}' is not writable."
            log.error(log_str)
            raise ValueError(log_str)
        return path

    @model_validator(mode='after')
    def check_device(cls, values):
        """Validates device configuration."""
        device = values.device
        gpu_id = values.gpu_id
        if device == 'cuda' and gpu_id < 0:
            raise ValueError("GPU ID must be >= 0 when using CUDA.")
        return values


# Example to demo field validation
if __name__ == '__main__':
    try:
        config = Config(
            mode='train',
            multimodel_type='none',
            random_seed=42,
            device='cuda',
            gpu_id=0,
            data_loader='base_loader',
            data_sampler='base_sampler',
            trainer='trainer',
            save_path='../results',
            train={
                'start_time': '2000/01/01',
                'end_time': '2000/12/31',
                'target': ['target1'],
                'optimizer': 'Adadelta',
                'batch_size': 100,
                'epochs': 50,
                'start_epoch': 0,
                'save_epoch': 5,
            },
            test={
                'start_time': '2001/01/01',
                'end_time': '2001/12/31',
                'batch_size': 100,
                'test_epoch': 50,
            },
            loss_function={
                'model': 'RmseLossComb',
            },
            delta_model={
                'rho': 365,
                'phy_model': {
                    'model': ['None_model'],
                    'nmul': 1,
                    'warm_up': 365,
                    'dynamic_params': {
                        'None_model': ['z1', 'z2'],
                    },
                    'forcings': ['x1_var', 'x2_var'],
                    'attributes': ['attr1', 'attr2'],
                    },
                'nn_model': {
                    'model': 'LSTM',
                    'dropout': 0.5,
                    'hidden_size': 256,
                    'learning_rate': 1.0,
                    'forcings': ['x1_var', 'x2_var'],
                    'attributes': ['attr1', 'attr2'],
                },
            },
            observations={
                'name': 'example',
                'train_path': '.',
                'test_path': '.',
                'start_time_all': '2000/01/01',
                'end_time_all': '2024/12/31',
                'all_forcings': ['x1_var', 'x2_var'],
                'all_attributes': ['attr1', 'attr2'],
            },
        )
        print("Configuration is valid.")
    except ValidationError as e:
        print(f"Configuration is invalid: {e}")
