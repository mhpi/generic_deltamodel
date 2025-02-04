import datetime as dt
import logging
from abc import ABC, abstractmethod
from re import I
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind2


def time_to_date(t, hr=False):
    """Convert time to date or datetime object.
    
    Adapted from Farshid Rahmani.
    
    Parameters
    ----------
    t : int, datetime, date
        Time object to convert.
    hr : bool
        If True, return datetime object.
    """
    tOut = None

    if type(t) is str:
        t = int(t.replace('/', ''))

    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception("Failed to change time to date.")
    return tOut


def trange_to_array(tRange, *, step=np.timedelta64(1, "D")):
    sd = time_to_date(tRange[0])
    ed = time_to_date(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def random_index(
    ngrid: int,
    nt: int,
    dim_subset: Tuple[int, int],
    warm_up: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, size=batch_size)
    i_t = np.random.randint(0 + warm_up, nt - rho, size=batch_size)
    return i_grid, i_t


def create_training_grid(
    x: np.ndarray,
    config: Dict[str, Any],
    n_samples: int = None
) -> Tuple[int, int, int]:
    """Define a training grid of samples x iterations per epoch x time.

    Parameters
    ----------
    x : np.ndarray
        The input data for a model.
    config : dict
        The configuration dictionary.
    n_samples : int, optional
        The number of samples to use in the training grid.
    
    Returns
    -------
    Tuple[int, int, int]
        The number of samples, the number of iterations per epoch, and the
        number of timesteps.
    """
    t_range = config['train_time']
    n_t = x.shape[0]

    if n_samples is None:
        n_samples = x.shape[1]

    t = trange_to_array(t_range)
    rho = min(t.shape[0], config['dpl_model']['rho'])

    # Calculate number of iterations per epoch.
    n_iter_ep = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['train']['batch_size'] * rho / n_samples
                     / (n_t - config['dpl_model']['phy_model']['warm_up']))
        )
    )
    return n_samples, n_iter_ep, n_t,
