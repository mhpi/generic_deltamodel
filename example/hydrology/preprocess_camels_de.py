"""Build the CAMELS-DE input pickle for dMG's HydroLoader.

Produces a ``(forcings, target, attributes)`` tuple matching the contract that
the CAMELS-USA pickle uses, so the existing ``HydroLoader`` can consume CAMELS-DE
without any framework changes:

* ``forcings`` -- ``(N, T, 3)`` float32, channels ``[prcp, tmean, pet]`` in mm/day,
  Celsius, and mm/day respectively.
* ``target`` -- ``(N, T, 1)`` float32, ``discharge_spec_obs`` in mm/day.
* ``attributes`` -- ``(N, A)`` float32 over a 15-attribute schema chosen to
  overlap with the CAMELS-USA dHBV 1.1p attribute list (see ``ATTR_SPEC``).

Time axis: 1980-01-02 .. 2020-12-31 (14975 days), matching the v1.0.0 record.

Inputs (CAMELS-DE v1.0.0 layout; ``--de-root``):

* ``timeseries/CAMELS_DE_hydromet_timeseries_<id>.csv``  -- P, T, obs Q.
* ``timeseries_simulated/CAMELS_DE_discharge_sim_<id>.csv``  -- pet_hargreaves.
* ``CAMELS_DE_{climatic,topographic,soil,landcover}_attributes.csv``.

Outputs (``--out-dir``): ``camels_de.pkl`` and ``gage_info.npy``.

CAMELS-DE provides 1,582 catchments; this script restricts to a user-supplied
list (``--selected``, one ``catchment_name`` per row in CSV). The default
selection matches the 1,347 catchments used by Bashyal & Song et al. (2026).

Reference:
    Loritz, R., et al. (2024). CAMELS-DE: hydro-meteorological time series and
    attributes for 1582 catchments in Germany.  Earth System Science Data.
    https://doi.org/10.5194/essd-16-5625-2024
"""
import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd

ALL_TIME = pd.date_range("1980-01-02", "2020-12-31", freq="D")
N_T = len(ALL_TIME)  # 14975

# Attribute names follow the CAMELS-USA dHBV 1.1p schema so the same NN
# attribute list can be reused. Unavailable USA-only attributes are dropped;
# proxies are noted in comments.
ATTR_SPEC = [
    "p_mean",            # climatic.p_mean
    "pet_mean",          # derived: mean(PET) per catchment over ALL_TIME
    "aridity",           # derived: pet_mean / p_mean
    "p_seasonality",     # climatic.p_seasonality
    "frac_snow",         # climatic.frac_snow
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "elev_mean",         # topographic.elev_mean
    "area_gages2",       # renamed: topographic.area
    "frac_forest",       # proxy: landcover.forests_and_seminatural_areas_perc / 100
    "sand_frac",         # proxy: soil.sand_0_30cm_mean / 100  (top 0-30 cm layer)
    "silt_frac",         # proxy: soil.silt_0_30cm_mean / 100
    "clay_frac",         # proxy: soil.clay_0_30cm_mean / 100
]


def _read_catchment_series(de_root: pathlib.Path, cid: str):
    """Return a per-catchment DataFrame on the ALL_TIME index with P, T, PET, Q."""
    ts = pd.read_csv(
        de_root / "timeseries" / f"CAMELS_DE_hydromet_timeseries_{cid}.csv",
        usecols=["date", "discharge_spec_obs", "precipitation_mean", "temperature_mean"],
        parse_dates=["date"],
    ).rename(columns={
        "discharge_spec_obs": "Q",
        "precipitation_mean": "P",
        "temperature_mean": "T",
    }).set_index("date")
    sim = pd.read_csv(
        de_root / "timeseries_simulated" / f"CAMELS_DE_discharge_sim_{cid}.csv",
        usecols=["date", "pet_hargreaves"],
        parse_dates=["date"],
    ).rename(columns={"pet_hargreaves": "PET"}).set_index("date")
    df = ts.join(sim, how="outer")
    # CAMELS-DE encodes missing/invalid obs Q as negative values.
    df.loc[df["Q"] < 0, "Q"] = np.nan
    return df.reindex(ALL_TIME)


def build_forcings_target(de_root: pathlib.Path, ids):
    forcings = np.full((len(ids), N_T, 3), np.nan, dtype=np.float32)
    target = np.full((len(ids), N_T, 1), np.nan, dtype=np.float32)
    for i, cid in enumerate(ids):
        df = _read_catchment_series(de_root, cid)
        forcings[i, :, 0] = df["P"].to_numpy(dtype=np.float32)
        forcings[i, :, 1] = df["T"].to_numpy(dtype=np.float32)
        forcings[i, :, 2] = df["PET"].to_numpy(dtype=np.float32)
        target[i, :, 0] = df["Q"].to_numpy(dtype=np.float32)
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(ids)} catchments")
    return forcings, target


def build_attributes(de_root: pathlib.Path, ids, forcings):
    clim = pd.read_csv(de_root / "CAMELS_DE_climatic_attributes.csv").set_index("gauge_id")
    topo = pd.read_csv(de_root / "CAMELS_DE_topographic_attributes.csv").set_index("gauge_id")
    soil = pd.read_csv(de_root / "CAMELS_DE_soil_attributes.csv").set_index("gauge_id")
    land = pd.read_csv(de_root / "CAMELS_DE_landcover_attributes.csv").set_index("gauge_id")

    A = pd.DataFrame(index=ids)
    A["p_mean"] = clim["p_mean"]
    A["p_seasonality"] = clim["p_seasonality"]
    A["frac_snow"] = clim["frac_snow"]
    A["high_prec_freq"] = clim["high_prec_freq"]
    A["high_prec_dur"] = clim["high_prec_dur"]
    A["low_prec_freq"] = clim["low_prec_freq"]
    A["low_prec_dur"] = clim["low_prec_dur"]
    A["elev_mean"] = topo["elev_mean"]
    A["area_gages2"] = topo["area"]
    A["frac_forest"] = land["forests_and_seminatural_areas_perc"] / 100.0
    A["sand_frac"] = soil["sand_0_30cm_mean"] / 100.0
    A["silt_frac"] = soil["silt_0_30cm_mean"] / 100.0
    A["clay_frac"] = soil["clay_0_30cm_mean"] / 100.0

    pet_mean = np.nanmean(forcings[..., 2], axis=1)
    A["pet_mean"] = pet_mean
    A["aridity"] = pet_mean / np.maximum(A["p_mean"].to_numpy(dtype=np.float32), 1e-3)

    attributes = A.reindex(columns=ATTR_SPEC).to_numpy(dtype=np.float32)

    # Median-fill any remaining NaNs so HydroLoader's normalization stays defined.
    nan_per_col = np.isnan(attributes).sum(axis=0)
    if nan_per_col.any():
        print(f"  NaNs per column before fill: {dict(zip(ATTR_SPEC, nan_per_col.tolist()))}")
        col_med = np.nanmedian(attributes, axis=0)
        rows, cols = np.where(np.isnan(attributes))
        attributes[rows, cols] = col_med[cols]
    return attributes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--de-root", type=pathlib.Path, required=True,
                        help="CAMELS-DE v1.0.0 root containing timeseries/, timeseries_simulated/, CAMELS_DE_*.csv")
    parser.add_argument("--selected", type=pathlib.Path, required=True,
                        help="CSV with one column 'catchment_name' listing the catchment IDs to keep.")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True,
                        help="Output directory for camels_de.pkl and gage_info.npy.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.selected)["catchment_name"].tolist()
    print(f"N catchments: {len(ids)} | T: {N_T} | attrs: {len(ATTR_SPEC)}")

    forcings, target = build_forcings_target(args.de_root, ids)
    print(f"forcings={forcings.shape}, target={target.shape}")

    attributes = build_attributes(args.de_root, ids, forcings)
    print(f"attributes={attributes.shape}")

    pkl_path = args.out_dir / "camels_de.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump((forcings, target, attributes), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {pkl_path} ({pkl_path.stat().st_size / 1e9:.2f} GB)")

    np.save(args.out_dir / "gage_info.npy", np.array(ids, dtype=object))
    print(f"wrote {args.out_dir / 'gage_info.npy'}")


if __name__ == "__main__":
    main()
