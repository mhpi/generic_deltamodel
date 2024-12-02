"""Validation of basic configuration file parameters using Pydantic.

Run this script to validate an example config object located below the
validation classes.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
from datetime import datetime


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
    train = 'train'
    test = 'test'
    train_test = 'train_test'


class LossFunctionConfig(BaseModel):
    model: str = Field(..., description="The name of the loss function.")


class TrainingConfig(BaseModel):
    start_time: str
    end_time: str
    target: List[str]
    optimizer: str
    batch_size: int
    epochs: int
    start_epoch: int = 0
    save_epoch: int = 5

    @model_validator(mode='after')
    def validate_training_times(cls, values):
        start_time = datetime.strptime(values.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(values.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Training `start_time` must be earlier than `end_time`.")
        return values

    @field_validator('target')
    @classmethod
    def validate_targets(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Training target cannot be empty.")
        return v


class TestingConfig(BaseModel):
    start_time: str
    end_time: str
    batch_size: int
    test_epoch: int

    @model_validator(mode='after')
    def validate_testing_times(cls, values):
        start_time = datetime.strptime(values.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(values.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Testing start_time must be earlier than end_time.")
        return values


class PhyModelConfig(BaseModel):
    model: List[str]
    nmul: int
    warm_up: int
    dynamic_params: Dict[str, List[str]]
    forcings: List[str] = Field(default_factory=list, description="List of dynamic input variables.")
    attributes: List[str] = Field(default_factory=list, description="List of static input variables.")


class NeuralNetworkModelConfig(BaseModel):
    model: str = Field(..., description="The type of neural network model (e.g., LSTM).")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate.")
    hidden_size: int = Field(..., gt=0, description="Number of hidden units.")
    learning_rate: float = Field(..., gt=0, description="Learning rate.")
    forcings: List[str] = Field(default_factory=list, description="List of dynamic input variables.")
    attributes: List[str] = Field(default_factory=list, description="List of static input variables.")


class DPLModelConfig(BaseModel):
    rho: int
    phy_model: PhyModelConfig
    nn_model: NeuralNetworkModelConfig


class ObservationConfig(BaseModel):
    name: str = 'not_defined'
    train_path: str = 'not_defined'
    test_path: str = 'not_defined'
    data_start_time: str = 'not_defined'
    data_end_time: str = 'not_defined'
    forcings_all: List[str] = Field(default_factory=list, description="List of dynamic input variables.")
    attributes_all: List[str] = Field(default_factory=list, description="List of static input variables.")

    @field_validator('train_path', 'test_path')
    @classmethod
    def validate_dir(cls, v: str) -> Union[Path, str]:
        if v == 'not_defined':
            return v
        return check_path(v)


class Config(BaseModel):
    mode: ModeEnum = Field(default=ModeEnum.train_test)
    multimodel_type: str = 'none'
    random_seed: int = 0
    device: str = 'cpu'
    gpu_id: int = 0
    save_path: str
    train: TrainingConfig
    test: TestingConfig
    loss_function: LossFunctionConfig
    dpl_model: DPLModelConfig
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
        device = values.get('device')
        gpu_id = values.get('gpu_id')
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
            save_path='../results',
            train={
                'start_time': '2000/01/01',
                'end_time': '2000/12/31',
                'target': ['target1'],
                'optimizer': 'Adadelta',
                'batch_size': 100,
                'epochs': 50,
                'start_epoch': 0,
                'save_epoch': 5
            },
            test={
                'start_time': '2001/01/01',
                'end_time': '2001/12/31',
                'batch_size': 100,
                'test_epoch': 50
            },
            loss_function={
                'model': 'RmseLossComb'
            },
            dpl_model={
                'rho': 365,
                'phy_model': {
                    'model': ['None_model'],
                    'nmul': 1,
                    'warm_up': 365,
                    'dynamic_params': {
                        'None_model': ['z1', 'z2']
                    },
                    'forcings': ['x1_var', 'x2_var'],
                    'attributes': ['attr1', 'attr2']
                    },
                'nn_model': {
                    'model': 'LSTM',
                    'dropout': 0.5,
                    'hidden_size': 256,
                    'learning_rate': 1.0,
                    'forcings': ['x1_var', 'x2_var'],
                    'attributes': ['attr1', 'attr2']
                }
            },
            observations={
                'name': 'example',
                'train_path': '.',
                'test_path': '.',
                'data_start_time': '2000/01/01',
                'data_end_time': '2024/12/31',
                'forcings_all': ['x1_var', 'x2_var'],
                'attributes_all': ['attr1', 'attr2']
            },
        )
        print("Configuration is valid.")
    except ValidationError as e:
        print(f"Configuration is invalid: {e}")
