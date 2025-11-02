"""Validation of basic configuration file parameters using Pydantic.

Run this script to validate an example config object (see bottom of file).

-leoglonz, taddbindas 2024
"""

import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

log = logging.getLogger(__name__)


def check_path(key: str, v: str) -> Path:
    """Checks if a given path exists and is valid."""
    path = Path(v)
    if not path.exists():
        log_str = f"Path does not exist for {key}: {v}"
        log.error(log_str)
        raise ValueError(log_str)
    return path


## ------ Enums ------- ##
class ModeEnum(str, Enum):
    """Enumeration for different modes of operation."""

    train = 'train'
    test = 'test'
    sim = 'sim'
    train_test = 'train_test'


class LoggingEnum(str, Enum):
    """Enumeration for different experiment logging."""

    none = 'none'
    tensorboard = 'tensorboard'
    wandb = 'wandb'


class MultimodelEnum(str, Enum):
    """Enumeration for different multimodel types."""

    none = 'none'
    nn_parallel = 'nn_parallel'
    nn_sequential = 'nn_sequential'


class OptimizerNameEnum(str, Enum):
    """Enumeration for different PyTorch optimizers."""

    adadelta = 'Adadelta'
    adam = 'Adam'


class LRSchedulerNameEnum(str, Enum):
    """Enumeration for different PyTorch learning rate schedulers."""

    none = 'none'
    steplr = 'StepLR'
    exponentiallr = 'ExponentialLR'
    cosineannealinglr = 'CosineAnnealingLR'


class TestNameEnum(str, Enum):
    """Enumeration for different test types."""

    temporal = 'temporal'
    spatial = 'spatial'


class FlowRegimeEnum(str, Enum):
    """Enumeration for different flow regimes."""

    none = 'none'
    low = 'low'
    high = 'high'


## ------ Training Utilities ------- ##
class OptimizerConfig(BaseModel):
    """Configuration for optimizer."""

    model_config = {'extra': 'allow'}

    name: Optional[OptimizerNameEnum] = Field(default=OptimizerNameEnum.adadelta)


class LRSchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler parameters."""

    model_config = {'extra': 'allow'}

    name: Optional[LRSchedulerNameEnum] = None
    step_size: Optional[int] = None
    gamma: Optional[float] = None


class LossFunctionConfig(BaseModel):
    """Configuration for the loss function used in training."""

    model_config = {'extra': 'allow'}

    name: str


## ------ Experiment Modes ------- ##
class TrainConfig(BaseModel):
    """Configuration for training."""

    model_config = {'extra': 'allow'}

    start_time: str
    end_time: str
    target: list[str]
    optimizer: OptimizerConfig
    lr: float = Field(
        ...,
        gt=0,
    )
    lr_scheduler: Optional[LRSchedulerConfig] = None
    loss_function: LossFunctionConfig
    batch_size: int
    epochs: int
    start_epoch: Optional[int] = 0
    save_epoch: Optional[int] = 1

    @model_validator(mode='after')
    def validate_training_times(self) -> 'TrainConfig':
        """Validates the training start and end times."""
        start_time = datetime.strptime(self.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(self.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Training `start_time` must be earlier than `end_time`.")

        if self.lr_scheduler:
            if self.lr_scheduler.name.lower() == 'none':
                self.lr_scheduler = None
        return self

    @field_validator('target')
    @classmethod
    def validate_targets(cls, v: list[str]) -> list[str]:
        """Validates the training target."""
        if not v:
            raise ValueError("Training target list cannot be empty.")
        return v


class TestConfig(BaseModel):
    """Configuration for testing."""

    model_config = {'extra': 'allow'}

    name: Optional[TestNameEnum] = Field(default=TestNameEnum.temporal)
    start_time: str
    end_time: str
    batch_size: int
    test_epoch: int

    @model_validator(mode='after')
    def validate_testing_times(self) -> 'TestConfig':
        """Validates the testing start and end times."""
        start_time = datetime.strptime(self.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(self.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Testing `start_time` must be earlier than `end_time`.")
        return self


class SimConfig(BaseModel):
    """Configuration for simulation."""

    model_config = {'extra': 'allow'}

    start_time: str
    end_time: str
    batch_size: int

    @model_validator(mode='after')
    def validate_testing_times(self) -> 'SimConfig':
        """Validates the testing start and end times."""
        start_time = datetime.strptime(self.start_time, '%Y/%m/%d')
        end_time = datetime.strptime(self.end_time, '%Y/%m/%d')
        if start_time >= end_time:
            raise ValueError("Testing `start_time` must be earlier than `end_time`.")
        return self


## ------ Model Configurations ------- ##
class PhyModelConfig(BaseModel):
    """Configuration class for the physics-based model."""

    model_config = {'extra': 'allow'}

    name: list[str]
    dynamic_params: dict[str, list[str]]
    nearzero: Optional[float] = 1e-5
    forcings: Optional[list[str]] = Field(default_factory=list)
    attributes: Optional[list[str]] = Field(default_factory=list)

    ## Optional defaults for mhpi models ##
    nmul: Optional[int] = None
    warm_up_states: Optional[bool] = None
    dy_drop: Optional[float] = None
    routing: Optional[bool] = None

    @model_validator(mode='after')
    def validate_dynamic_params(self) -> 'PhyModelConfig':
        """Validates that keys in dynamic_params are present in the model 'name' list."""
        dynamic_params = self.dynamic_params
        model_names = self.name
        if not dynamic_params:
            return self
        for key in dynamic_params.keys():
            if key not in model_names:
                raise ValueError(
                    f"Dynamic parameter key '{key}' is not a valid model name. Must be one of {model_names}."
                )
        return self


class NeuralNetworkModelConfig(BaseModel):
    """Configuration for the neural network model."""

    model_config = {'extra': 'allow'}

    name: str
    forcings: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)

    ## Optional defaults for mhpi models ##
    dropout: Optional[float] = Field(None, ge=0.0, le=1.0)
    hidden_size: Optional[int] = Field(None, gt=0)


class ModelConfig(BaseModel):
    """Configuration for the differentiable model."""

    model_config = {'extra': 'allow'}

    rho: int
    nn: Optional[NeuralNetworkModelConfig] = Field(
        default_factory=NeuralNetworkModelConfig
    )

    ## Optional defaults for mhpi models ##
    warm_up: Optional[int] = None
    use_log_norm: Optional[list[str]] = None
    flow_regime: Optional[FlowRegimeEnum] = None
    phy: Optional[PhyModelConfig] = None


## ------ Dataset/Observation Configurations ------- ##
class ObservationConfig(BaseModel):
    """Configuration for observations/datasets."""

    model_config = {'extra': 'allow'}

    name: str
    data_path: str
    start_time: str
    end_time: str
    all_forcings: list[str] = Field(default_factory=list)
    all_attributes: list[str] = Field(default_factory=list)

    ## Optional defaults for mhpi models ##
    gage_info: Optional[str] = None
    subset_path: Optional[str] = None
    area_name: Optional[str] = None

    @model_validator(mode='after')
    def validate_data_path(self) -> 'ObservationConfig':
        """Validates data paths exists."""
        check_path('observations.data_path', self.data_path)
        if self.gage_info:
            check_path('observations.gage_info', self.gage_info)
        if self.subset_path:
            check_path('observations.subset_path', self.subset_path)
        return self

    @model_validator(mode='after')
    def validate_dataset_times(self):
        """Validates the dataset start and end times."""
        if self.start_time == 'not_defined' and self.end_time == 'not_defined':
            return self
        elif self.start_time == 'not_defined' or self.end_time == 'not_defined':
            raise ValueError(
                "Both train_path and test_path must be defined if either is specified."
            )
        else:
            start_time = datetime.strptime(self.start_time, '%Y/%m/%d')
            end_time = datetime.strptime(self.end_time, '%Y/%m/%d')
            if start_time >= end_time:
                raise ValueError("Dataset start time must be earlier than end time.")
            return self


## ------ Root Configuration ------- ##
class Config(BaseModel):
    """Root configuration model for the entire application."""

    model_config = {'extra': 'allow'}

    name: Optional[str] = None
    mode: Optional[ModeEnum] = Field(default=ModeEnum.train_test)
    do_tune: Optional[bool] = False
    multimodel_type: Optional[MultimodelEnum] = None
    seed: Optional[int] = 0
    logging: Optional[LoggingEnum] = None
    cache_states: Optional[bool] = False
    device: str
    gpu_id: Optional[int] = 0
    verbose: Optional[bool] = False

    data_loader: Optional[str] = None
    data_sampler: Optional[str] = None
    trainer: Optional[str] = None
    trained_model: Optional[str] = ''

    output_dir: Optional[str] = None
    model_dir: Optional[str] = None
    plot_dir: Optional[str] = None
    sim_dir: Optional[str] = None
    log_dir: Optional[str] = None
    load_state_path: Optional[str] = None

    train: TrainConfig
    test: TestConfig
    sim: Optional[SimConfig] = None

    model: ModelConfig
    observations: ObservationConfig
    tune: Optional[dict] = None

    @model_validator(mode='after')
    def check_paths(self) -> 'Config':
        """Validate paths in the configuration."""
        check_path('trained_model', self.trained_model)
        return self

    @model_validator(mode='after')
    def populate_fields(self) -> 'Config':
        """
        Validates device configuration and populates other config fields.
        that are not explicitly provided (reduce config clutter for redundant
        settings).
        """
        ### Auto-assignments if not provided. ###
        if not self.name:
            self.name = 'dmg-exp'
            log.warning(f"No `name` provided in config. Auto-assigning: {self.name}")

        if not self.multimodel_type or self.multimodel_type.lower() == 'none':
            self.multimodel_type = None

        if not self.logging or self.logging.lower() == 'none':
            self.logging = None

        # Assign state caching and warmup to sub-models.
        if self.model.phy:
            self.model.phy.cache_states = self.cache_states
            self.model.phy.warm_up = self.model.warm_up
        if self.model.nn:
            self.model.nn.cache_states = self.cache_states

        if self.device == 'cuda' and self.gpu_id < 0:
            raise ValueError("GPU ID must be >= 0 when using CUDA.")

        ### Create output directories and add path to config. ###
        if not self.output_dir:
            self.output_dir = os.getcwd()
        if not self.model_dir:
            self.model_dir = os.path.join(self.output_dir, 'model/')
        if not self.plot_dir:
            self.plot_dir = os.path.join(self.output_dir, 'plot/')
        if not self.sim_dir:
            self.sim_dir = os.path.join(self.output_dir, 'sim/')
        if (not self.log_dir) and self.logging:
            self.log_dir = os.path.join(self.output_dir, self.logging)

        # ### Convert timestamps to list format. ###
        # exp_time_start = min(
        #     self.train_time.start_time,
        #     self.train_time.end_time,
        #     self.test_time.start_time,
        #     self.test_time.end_time,
        # )
        # exp_time_end = max(
        #     self.train_time.start_time,
        #     self.train_time.end_time,
        #     self.test_time.start_time,
        #     self.test_time.end_time,
        # )

        # self.train_time = [self.train.start_time, self.train.end_time]
        # self.test_time = [self.test.start_time, self.test.end_time]
        # self.sim_time = [self.sim.start_time, self.sim.end_time]
        # self.exp_time = [exp_time_start, exp_time_end]
        # self.all_time = [self.observations.start_time, self.observations.end_time]

        return self
