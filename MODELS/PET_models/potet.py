import numpy as np
import pandas as pd
import torch
import os
# from core.read_configurations import config


def potet_hamon(mean_air_temp, dayl, hamon_coef=0.0055):  # hamon_coef=0.1651
    """
    :param mean_air_temp: daily mean air temperature (celecius)
    :param dayl: seconds of sunshine(number of hours between sunshine and sunset), need to convert to hour
    :param hamon_coef: coefficient for Hamon equation
    :return: PET potential evapotranspiration (m/sec after multiplying to conversion factors)
    """

    e_s = 6.108 * torch.exp(17.26939 * mean_air_temp / (mean_air_temp + 237.3))  # mbar

    # rho is saturated water-vapor density (absolute humidity)
    rho = (216.7 / (mean_air_temp + 273.3)) * e_s

    PET = (
        hamon_coef * torch.pow((dayl / 3600) / 12, 2) * rho * 0.0254 / 86400
    )  # 25.4 is converting inches/day to m/s

    # replacing negative values with zero
    mask_PET = PET.ge(0)
    PET = PET * mask_PET.int().float()

    return PET * (86400 * 1000)   # converting m / sec to mm / day

def potet_hargreaves(tmin, tmax, tmean, lat, day_of_year):
    trange = tmax - tmin
    trange[trange < 0] = 0
    latitude = torch.deg2rad(lat)
    SOLAR_CONSTANT = 0.0820
    sol_dec = 0.409 * torch.sin(((2.0 * 3.14159 / 365.0) * day_of_year - 1.39))
    sha = torch.acos(torch.clamp(-torch.tan(latitude) * torch.tan(sol_dec), min=-1.0, max=1.0))
    ird = 1 + (0.033 * torch.cos((2.0 * 3.14159 / 365.0) * day_of_year))
    tmp1 = (24.0 * 60.0) / 3.14159
    tmp2 = sha * torch.sin(latitude) * torch.sin(sol_dec)
    tmp3 = torch.cos(latitude) * torch.cos(sol_dec) * torch.sin(sha)
    et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)
    pet = 0.0023 * (tmean + 17.8) * trange ** 0.5 * 0.408 * et_rad
    pet[pet < 0] = 0
    return pet


def get_potet(args, **kwargs):
    if args["potet_module"] == "potet_hamon":
        PET = potet_hamon(kwargs["mean_air_temp"], kwargs["dayl"], kwargs["hamon_coef"])
    elif args["potet_module"] == "potet_pm":
        print("this PET method is not ready yet")
    elif args["potet_module"] == "potet_hargreaves":
        PET = potet_hargreaves(kwargs["tmin"], kwargs["tmax"], kwargs["tmean"], kwargs["lat"], kwargs["day_of_year"])
    return PET
