import os
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
from datetime import datetime


log = logging.getLogger(__name__)


def check_path(v: str) -> Path:
    """Checks if a given path exists and is valid."""
    path = Path(v)
    if not path.exists():
        log.error(f"Path '{v}' does not exist.")
        raise ValueError(f"Path '{v}' does not exist.")
    return path


class ModeEnum(str, Enum):
    train = "train"
    test = "test"
    train_test = "train_test"


class EnsembleEnum(str, Enum):
    none = "none"
    pnn_parallel = "pnn_parallel"
    pnn_sequential = "pnn_sequential"
    avg = "avg"
    reg_max = "reg_max"


class LossFunctionConfig(BaseModel):
    model: str = Field(..., description="The name of the loss function.")
    target: List[str] = Field(..., description="Target variables for the loss function.")

    @field_validator("target")
    @classmethod
    def validate_targets(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Loss function targets cannot be empty.")
        return v


class TrainingConfig(BaseModel):
    start_time: str
    end_time: str
    target: List[str]
    optimizer: str
    batch_size: int
    epochs: int
    start_epoch: int = 0
    save_epoch: int = 5

    @model_validator(mode="after")
    def validate_training_times(cls, values):
        start_time = datetime.strptime(values.start_time, "%Y/%m/%d")
        end_time = datetime.strptime(values.end_time, "%Y/%m/%d")
        if start_time >= end_time:
            raise ValueError("Training `start_time` must be earlier than `end_time`.")
        return values


class TestingConfig(BaseModel):
    start_time: str
    end_time: str
    batch_size: int
    test_epoch: int

    @model_validator(mode="after")
    def validate_testing_times(cls, values):
        start_time = datetime.strptime(values.start_time, "%Y/%m/%d")
        end_time = datetime.strptime(values.end_time, "%Y/%m/%d")
        if start_time >= end_time:
            raise ValueError("Testing start_time must be earlier than end_time.")
        return values


class NeuralNetworkModelConfig(BaseModel):
    model: str = Field(..., description="The type of neural network model (e.g., LSTM).")
    dropout: float = Field(..., ge=0.0, le=1.0, description="Dropout rate.")
    hidden_size: int = Field(..., gt=0, description="Number of hidden units.")
    learning_rate: float = Field(..., gt=0, description="Learning rate.")
    input_vars: List[str] = Field(..., description="Input variables for the NN model.")

    @field_validator("input_vars")
    @classmethod
    def validate_input_vars(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Training `start_time` must be earlier than `end_time`.")
        return v


class DPLModelConfig(BaseModel):
    nmul: int
    rho: int
    dy_drop: float
    phy_model: Dict[str, Union[str, List[str]]]
    nn_model: NeuralNetworkModelConfig


class ObservationConfig(BaseModel):
    name: str = "not_defined"
    gage_info: str = "not_defined"
    forcing_path: str = "not_defined"
    attr_path: str = "not_defined"
    forcings_all: List[str] = Field(default_factory=list)
    attributes_all: List[str] = Field(default_factory=list)
    phy_forcings_model: List[str] = Field(default_factory=list)

    @field_validator("forcing_path", "attr_path")
    @classmethod
    def validate_dir(cls, v: str) -> Union[Path, str]:
        if v == "not_defined":
            return v
        return check_path(v)


class Config(BaseModel):
    mode: ModeEnum = Field(default=ModeEnum.train_test)
    multimodel_type: EnsembleEnum = Field(default=EnsembleEnum.none)
    random_seed: int = 0
    device: str = "cpu"
    gpu_id: int = 0
    save_path: str
    train: TrainingConfig
    test: TestingConfig
    loss_function: LossFunctionConfig
    dpl_model: DPLModelConfig
    observations: ObservationConfig

    @field_validator("save_path")
    @classmethod
    def validate_save_path(cls, v: str) -> Path:
        """Validates the save_path directory."""
        path = Path(v)
        if not path.exists():
            log.error(f"Save path '{v}' does not exist.")
            raise ValueError(f"Save path '{v}' does not exist.")
        if not path.is_dir():
            log.error(f"Save path '{v}' is not a directory.")
            raise ValueError(f"Save path '{v}' is not a directory.")
        if not os.access(path, os.W_OK):
            log.error(f"Save path '{v}' is not writable.")
            raise ValueError(f"Save path '{v}' is not writable.")
        return path

    @model_validator(mode="after")
    def check_device(cls, values):
        """Validates device configuration."""
        device = values.get("device")
        gpu_id = values.get("gpu_id")
        if device == "cuda" and gpu_id < 0:
            raise ValueError("GPU ID must be >= 0 when using CUDA.")
        return values


# Example of using the updated config model
if __name__ == "__main__":
    try:
        config = Config(
            mode="train",
            multimodel_type="none",
            random_seed=42,
            device="cuda",
            gpu_id=0,
            save_path="../results",
            train={
                "start_time": "2000/01/01",
                "end_time": "2000/12/31",
                "target": ["flow_sim"],
                "optimizer": "Adadelta",
                "batch_size": 100,
                "epochs": 50,
            },
            test={
                "start_time": "2001/01/01",
                "end_time": "2001/12/31",
                "batch_size": 100,
                "test_epoch": 50,
            },
            loss_function={
                "model": "RmseLossComb",
                "target": ["flow_sim"],
            },
            dpl_model={
                "nmul": 16,
                "rho": 365,
                "dy_drop": 0.0,
                "phy_model": {
                    "model": ["None_model"],
                    "warm_up": 365,
                    "input_vars": ["x1_var", "x2_var"],
                },
                "nn_model": {
                    "model": "LSTM",
                    "dropout": 0.5,
                    "hidden_size": 256,
                    "learning_rate": 1.0,
                    "input_vars": ["x1_var", "x2_var"],
                },
            },
            observations={
                "name": "example",
                "gage_info": "info",
                "forcing_path": "../data",
                "attr_path": "../attributes",
            },
        )
        print("Configuration is valid.")
    except ValidationError as e:
        print(f"Configuration is invalid: {e}")
