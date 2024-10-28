import dataclasses
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass
from torch import float16

# from dPLHydro_multimodel.models import hydro_models
# from old_files.hydroDL import dataset

log = logging.getLogger(__name__)


def check_path(v: str) -> Path:
    path = Path(v)
    if not path.exists():
        log.exception(f"Path {v} does not exist")
        raise ValueError(f"Path {v} does not exist")
    return path


def check_dictionary_paths(v: Dict[str, str]) -> Dict[str, Path]:
    path_dict = {}
    for k, val in v.items():
        path = Path(val)
        if not path.exists():
            log.exception(f"Path {v} does not exist")
            raise ValueError(f"Path {v} does not exist")
        path_dict[k] = path
    return path_dict


class ModeEnum(str, Enum):
    train = "train"
    test = "test"
    train_test = "train_test"
    train_wnn_only = "train_wnn"
    train_conus = "train_conus"
    test_conus = "test_conus"

    # test_bmi = "test_bmi"


class EnsembleEnum(str, Enum):
    none = 'none'
    free_pnn = 'free_pnn'
    frozen_pnn = 'frozen_pnn'
    avg = 'avg'
    reg_max = 'reg_max'


class InitalizationEnum(str, Enum):
    kaiming_normal = "kaiming_normal"
    kaiming_uniform = "kaiming_uniform"
    orthogonal = "orthogonal"
    sparse = "sparse"
    trunc_normal = "trunc_normal"
    xavier_normal = "xavier_normal"
    xavier_uniform = "xavier_uniform"
    default = "default"


@dataclass
class AttributeIndices:
    area: int = 5
    length: int = 0
    slope: int = 3


@dataclass
class AttributeMaximums:
    slope: float = 1e-4
    velocity: float = 0.3


@dataclass
class AttributeMinimums:
    q_prime: int = 0
    slope: float = 1e-4
    velocity: float = 0.3


# @dataclass
# class DataSources:
#     edges: str
#     gage_coo_indices: str
#     HUC_TM: Optional[str]
#     MERIT_TM: str
#     streamflow: str

#     @field_validator(
#         "edges", "gage_coo_indices", "HUC_TM", "MERIT_TM", "streamflow", "statistics"
#     )
#     @classmethod
#     def validate_data_dir(cls, v: str) -> Path:
#         return check_path(v)


@dataclass
class PhysicsVariables:
    n: Optional[float] = 0.03
    q: Optional[float] = 1.5
    p: Optional[float] = 21.0
    t: float = 3600
    x: float = 0.3


class ParameterRange(BaseModel):
    range: Dict = Field(
        default_factory=lambda: {
            "n": [0.01, 0.3],
            "q_spatial": [0.0, 3.0],
            "p_spatial": [0.0, 42.0],
        }
    )


@dataclass
class Params:
    save_path: str = "None"
    attributes: List[str] = dataclasses.field(
        default_factory=lambda: [
            "len",
            "len_dir",
            "sinuosity",
            "slope",
            "stream_drop",
            "uparea",
            "N",
            "alpha",
            "aridity",
            "clay_mean_05",
            "glacier",
            "ksat",
            "mean_elevation",
            "mean_p",
            "ormc",
            "porosity",
            "sand_mean_05",
            "sat-field",
            "silt_mean_05",
        ]
    )
    attribute_defaults: List[float] = dataclasses.field(
        default_factory=lambda: [
            2117.4235,  # len
            1410.9703,  # len_dir
            1.5602,  # sinuosity
            0.0005,  # slope
            42.7428,  # stream_drop
            15088.1365,  # uparea
            1.30268,  # N
            0.0144,  # alpha
            1.54986,  # aridity
            22.542,  # clay_mean_05
            0.0,  # glacier
            22.7993,  # ksat
            601.3174,  # mean_elevation
            737.6875,  # mean_p
            7.961,  # ormc
            0.1151,  # porosity
            40.5571,  # sand_mean_05
            0.1414,  # sat-field
            36.1142,  # silt_mean_05
        ]
    )
    attribute_indices: AttributeIndices = Field(default_factory=AttributeIndices)
    attribute_maximums: AttributeMaximums = Field(default_factory=AttributeMaximums)
    attribute_minimums: AttributeMinimums = Field(default_factory=AttributeMinimums)
    parameter_ranges: ParameterRange = Field(default_factory=ParameterRange)
    physics_variables: PhysicsVariables = Field(default_factory=PhysicsVariables)
    warmup: int = 72


@dataclass
class LSTM:
    hidden_size: int
    input_size: int
    output_size: int
    learnable_parameters: List[str]
    fan: Optional[str] = "fan_in"
    gain: Optional[float] = 0.7
    initialization: Optional[InitalizationEnum] = Field(
        default=InitalizationEnum.xavier_normal
    )


class ExperimentConfig(BaseModel):
    start_time: str = "1994/10/01"
    end_time: str = "1995/09/30"


# class DynamicConfig(BaseModel):
#     HBV: list
#     marrmot_PRMS: list
#     SACSMA_with_snow: list
    

class LossFunc(BaseModel):
    w1: float = 11.0
    w2: float = 1.0 


class WeightingNNConfig(BaseModel):
    dropout: float
    hidden_size: int
    loss_factor: int
    learning_rate: float = 0.1
    method: str
    loss_function: str
    loss_lower_bound: float = 0.95
    loss_upper_bound: float = 1.05
    loss_function_weights: LossFunc


# class Checkpoint(BaseModel):
#     start_epoch: int
#     HBV: str
#     marrmot_PRMS: str
#     SACSMA_with_snow: str
#     weighting_nn: str


@dataclass
class ObservationConfig:
    name: str = "not_defined"
    gage_info: str = "not_defined"
    forcing_path: str = "not_defined"
    attr_path: str = "not_defined"
    var_t_NN: list = "not_defined"
    var_c_NN: list = "not_defined"
    var_t_hydro_model: list = "not_defined"
    var_c_hydro_model: list = "not_defined"

    @field_validator('forcing_path', 'attr_path')
    @classmethod
    def validate_dir(cls, v: str) -> Union[Path, str]:
        if v == "not_defined":
            return v
        return check_path(v)


# @dataclass
# class LoadModelConfig:
#     name: str = "not_defined"
#     gage_info: str = "not_defined"
#     attr_path: str = "not_defined"
#     observations_path: str = "not_defined"
#     var_t_NN: list = "not_defined"
#     var_c_NN: list = "not_defined"
#     var_t_hydro_model: list = "not_defined"
#     var_c_hydro_model: list = "not_defined"

#     @field_validator("gage_info", "observations_path")
#     @classmethod
#     def validate_dir(cls, v: str) -> Union[Path, str]:
#         if v == "not_defined":
#             return v
#         return check_path(v)


class Config(BaseModel):
    observations: ObservationConfig

    mode: ModeEnum = Field(default=ModeEnum.train_test)
    pnn_model: str
    hydro_models: Union[List[str], str] = Field(default_factory=lambda: ['HBV'])
    ensemble_type: EnsembleEnum = Field(default=EnsembleEnum.none)
    # dy_params: DynamicConfig = Field(default_factory=ExperimentConfig)

    random_seed: int = 0
    device: str = 'cpu'
    gpu_id: int = 0
    dtype: str = ''

    routing_hydro_model: bool = True
    pet_module: str
    pet_dataset_name: str
    target: list

    loss_function: str
    loss_function_weights: LossFunc
    train: ExperimentConfig = Field(default_factory=ExperimentConfig)
    test: ExperimentConfig = ExperimentConfig(batch_size=365)
    params: Params = Field(default_factory=Params)

    nmul: int = 1
    warm_up: int
    rho: int
    batch_size: int
    epochs: int
    hidden_size: int
    dropout: float
    learning_rate: float = 0.1
    nearzero: float

    weighting_nn: WeightingNNConfig

    test_batch: int
    save_epoch: int = 10

    name: str
    data_dir: str
    output_dir: str
    use_checkpoint: bool
    # checkpoint: Checkpoint

    # gage_info: str = "not_defined"
    # attr_path: str = "not_defined"
    # observations_path: str = "not_defined"
    # var_t_NN: list = "not_defined"
    # var_c_NN: list = "not_defined"
    # var_t_hydro_model: list = "not_defined"
    # var_c_hydro_model: list = "not_defined"

    def __init__(self, **data):
        super(Config, self).__init__(**data)
        if self.params.save_path == "None":
            try:
                self.params.save_path = HydraConfig.get().run.dir
            except ValueError:
                log.info(
                    "HydraConfig is not set. If using a jupyter notebook"
                    "You must manually set your save_path: \n"
                    "cfg.params.save_path = Path(__file__) "
                )

    # def merge_yaml_configs(source_file, target_file):
    #     with open(source_file, 'r') as source_yaml_file:
    #         source_config = yaml.safe_load(source_yaml_file)

    #     with open(target_file, 'r+') as target_yaml_file:
    #         target_config = yaml.safe_load(target_yaml_file)
    #         target_config.update(source_config)
    #         target_yaml_file.seek(0)
    #         yaml.dump(target_config, target_yaml_file)

    @field_validator("data_dir")
    @classmethod
    def validate_dir(cls, v: str) -> Path:
        return check_path(v)

    # @field_validator("output_dir")
    # @classmethod
    # def validate_output_dir(cls, v: str) -> Path:
    #     return check_path(v)


    # @model_validator(mode="after")
    # @classmethod
    # def validate_devices(cls, config: Any) -> Any:
    #     device = config.device
    #     if isinstance(device, str):
    #         log.info("Running dMC using the CPU")
    #     elif len(device) < world_size:
    #         msg = "length of device must be >= to the number of processes (world size)"
    #         log.exception(msg)
    #         raise ValueError(msg)
    #     return config
