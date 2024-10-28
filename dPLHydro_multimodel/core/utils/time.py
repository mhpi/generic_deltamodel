# Adapted from dPL_Hydro_SNTEMP @ Farshid Rahmani.
import datetime as dt

import numpy as np


def time_to_date(t, hr=False):
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception("hydroDL.utils.t2dt failed")
    return tOut


def trange_to_array(tRange, *, step=np.timedelta64(1, "D")):
    sd = time_to_date(tRange[0])
    ed = time_to_date(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind1, ind2
