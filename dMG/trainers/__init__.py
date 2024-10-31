import logging

from conf.config import ModeEnum
from trainers.test import TestModel
from trainers.train import TrainModel

log = logging.getLogger(__name__)



def build_handler(cfg: dict):
    if cfg['mode'] == ModeEnum.train:
        return TrainModel(cfg)
    elif cfg['mode'] == ModeEnum.test:
        return TestModel(cfg)
    else:
        raise ValueError(f"Unsupported mode: {cfg['mode']}")


# def configure(cfg: Config):
#     def _bind(binder):
#         """ Binds the Configuration to a singleton scope.

#         :param binder: Binder object.
#         """
#         binder.bind(Config, to=cfg, scope=singleton)
#     return _bind
