import logging

from conf.config import Config, ModeEnum
from experiment.test import TestModel
# from archive.experiment.test_bmi import TestBMIModel
from experiment.train import TrainModel
from experiment.train_conus import TrainModel as TrainModelConus
from experiment.test_conus import TestModel as TestModelConus

from experiment.train_wnn import TrainWNNModel

# from experiment.factory import Factory
# from experiment.test_handler import TestHandler
# from experiment.train_handler import TrainHandler

# from typing import Union

# from injector import Injector, singleton


log = logging.getLogger(__name__)



def build_handler(cfg: Config, config_dict: dict): #-> Union[TrainHandler, TestHandler]:
    # injector = Injector([configure(cfg), Factory(cfg)])
    if cfg.mode == ModeEnum.train:
        # return injector.get(TrainHandler)
        return TrainModel(config_dict)
    elif cfg.mode == ModeEnum.test:
        # return injector.get(TestHandler)
        return TestModel(config_dict)
    elif cfg.mode == ModeEnum.train_wnn_only:
        return TrainWNNModel(config_dict)
    elif cfg.mode == ModeEnum.train_conus:
        return TrainModelConus(config_dict)
    elif cfg.mode == ModeEnum.test_conus:
        return TestModelConus(config_dict)
    # elif cfg.mode == ModeEnum.test_bmi:
    #     return TestBMIModel(config_dict)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


# def configure(cfg: Config):
#     def _bind(binder):
#         """
#         Binds the Configuration to a singleton scope.

#         :param binder: Binder object.
#         """
#         binder.bind(Config, to=cfg, scope=singleton)

#     return _bind
