import logging
import math
from abc import ABC
from typing import Callable, Dict

import torch
import torch.nn as nn
from conf.config import InitalizationEnum

log = logging.getLogger(__name__)


class Initialization(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Initialization, self).__init__()
        self.cfg = kwargs["cfg"]

    def kaiming_normal_initializer(self, x) -> None:
        nn.init.kaiming_normal_(x, mode=self.cfg.fan, nonlinearity="sigmoid")

    def xavier_normal_initializer(self, x) -> None:
        nn.init.xavier_normal_(x, gain=self.cfg.gain)

    @staticmethod
    def sparse_initializer(x) -> None:
        nn.init.sparse_(x, sparsity=0.5)

    @staticmethod
    def uniform_initializer(x) -> None:
        # hardcoding hidden size for now
        stdv = 1.0 / math.sqrt(6)
        nn.init.uniform_(x, a=-stdv, b=stdv)

    def forward(self, *args, **kwargs) -> Callable[[torch.Tensor], None]:
        init = self.cfg.initialization
        log.debug(f"Initializing weight states using the {init} function")
        if init == InitalizationEnum.kaiming_normal:
            func = self.kaiming_normal_initializer
        elif init == InitalizationEnum.kaiming_uniform:
            func = nn.init.kaiming_uniform_
        elif init == InitalizationEnum.orthogonal:
            func = nn.init.orthogonal_
        elif init == InitalizationEnum.sparse:
            func = self.sparse_initializer
        elif init == InitalizationEnum.trunc_normal:
            func = nn.init.trunc_normal_
        elif init == InitalizationEnum.xavier_normal:
            func = self.xavier_normal_initializer
        elif init == InitalizationEnum.xavier_uniform:
            func = nn.init.xavier_uniform_
        else:
            log.info("Defaulting to a uniform initialization")
            func = self.uniform_initializer
        return func


class NeuralNetwork(ABC, torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(NeuralNetwork, self).__init__()
        self.cfg = kwargs["cfg"]
        self.Initialization = Initialization(cfg=self.cfg)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("The forward function must be implemented")
