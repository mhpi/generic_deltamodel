import os
import sys
import json
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

import re


def convert_nested(obj):
    def convert_value(v):
        """Convert a single value from string to its proper dtype if possible."""
        if isinstance(v, str):
            # Handle special cases
            if v == "None":
                return None
            if v == "True":
                return True
            if v == "False":
                return False

            # Handle numeric forms (integer, float, scientific notation)
            if re.fullmatch(r"[-+]?\d+", v):
                return int(v)
            if re.fullmatch(
                r"[-+]?\d*\.\d+(e[-+]?\d+)?", v, re.IGNORECASE
            ) or re.fullmatch(r"[-+]?\d+e[-+]?\d+", v, re.IGNORECASE):
                try:
                    return float(v)
                except ValueError:
                    return v

            # Keep as string otherwise
            return v
        return v

    """Recursively traverse nested dicts/lists and convert all values."""
    if isinstance(obj, dict):
        return {k: convert_nested(convert_value(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nested(convert_value(i)) for i in obj]
    else:
        return convert_value(obj)


# ==========================================
# USER SETTINGS
# ==========================================
target_gauge_id = '2453'  # <--- REPLACE WITH YOUR DESIRED GAUGE ID (String)
output_dir = '/projects/mhpi/leoglonz/ciroh-ua/_ngen-data/extracted_data'
model_version = 3
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# SETUP PATHS & CONFIG
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load Configs to get variable lists
config_path = f'/projects/mhpi/yxs275/hourly_model/DShourly/trainedModel/h-dhbv2_{model_version}/config.json'
with open(Path(config_path)) as f:
    config = json.load(f)
config = convert_nested(config)

obs_cfg_path = os.path.join(
    PROJECT_ROOT, "distributedDS", "config", "observations_camels.yaml"
)
obs_config = yaml.safe_load(open(Path(obs_cfg_path)))
config['observations'] = obs_config

# Define Variable Lists
var_x_list = config['delta_model']['nn_model']['high_freq_model']['forcings']
var_c_list = config['delta_model']['nn_model']['high_freq_model']['attributes']
var_c_list2 = config['delta_model']['nn_model']['high_freq_model']['attributes2']

# Define Time Range (Matching your original script)
zTest_full_time = pd.date_range('2004-10-01 00:00:00', '2018-10-01 00:00:00', freq='h')[
    :-1
]

# ==========================================
# LOCATE GAUGE & LOAD DATA
# ==========================================
print(f"Locating gauge {target_gauge_id}...")

# 1. Open Attribute file to find the index of the gauge
attr_path = '/projects/mhpi/yxs275/hourly_model/mtsHBV/data/CAMELS_HFs_attr_new.nc'
attrs_ds_all = xr.open_dataset(attr_path)

# Normalize attribute gauges to string for search
all_gauges_str = attrs_ds_all['gauge'].values.astype(str)

if str(target_gauge_id) not in all_gauges_str:
    raise ValueError(f"Gauge {target_gauge_id} not found in attribute dataset.")

# Find index and calculate chunk
gauge_idx = np.where(all_gauges_str == str(target_gauge_id))[0][0]
gauge_chunk_size = 500
ichunk = gauge_idx // gauge_chunk_size

g_start = ichunk * gauge_chunk_size
g_end = min((ichunk + 1) * gauge_chunk_size, len(all_gauges_str))

print(
    f"Gauge found at index {gauge_idx}. Loading chunk {ichunk} ({g_start:05d}-{g_end - 1:05d})..."
)

# 2. Define File Paths for this chunk
hourly_x_path = f'/gpfs/yxs275/data/hourly/CAMELS_HF/forcing/forcing_1990_2018_gauges_hourly_{g_start:05d}_{g_end - 1:05d}.nc'

# 3. Load Data
print(f"Reading {hourly_x_path}")
hourly_x = xr.open_dataset(hourly_x_path).sel(time=zTest_full_time)
hourly_x = hourly_x.rename({"T": "Temp"})

# --- FIX: Match the Data Type of the NetCDF ---
search_id = target_gauge_id

# Check if the NetCDF 'gauge' coordinate is Integers
if np.issubdtype(hourly_x['gauge'].dtype, np.integer):
    print("Dataset uses Integer IDs. Converting target ID to integer...")
    try:
        search_id = int(target_gauge_id)
    except ValueError:
        print(
            f"WARNING: Could not convert '{target_gauge_id}' to integer, but dataset expects integers."
        )

# Select specific gauge
ds_forcing = hourly_x.sel(gauge=search_id)

# For attributes, we go back to the original attribute file
# We use the integer index we found earlier to be safe, or select by ID if possible
try:
    # Try selecting by ID (handling type match again just in case)
    if np.issubdtype(attrs_ds_all['gauge'].dtype, np.integer):
        attr_search_id = int(target_gauge_id)
    else:
        attr_search_id = str(target_gauge_id)
    ds_attr = attrs_ds_all.sel(gauge=attr_search_id)
except KeyError:
    # Fallback: select by integer index (isel) if label selection fails
    print("Selection by label failed for attributes, using index position instead.")
    ds_attr = attrs_ds_all.isel(gauge=gauge_idx)

# ==========================================
# EXTRACT & SAVE CSVs
# ==========================================

# --- 1. Forcings CSV ---
print("Extracting Forcings...")
# Filter only the variables defined in config
forcing_df = ds_forcing[var_x_list].to_dataframe()

# Clean up dataframe (remove gauge index column if it exists, keep time index)
if 'gauge' in forcing_df.columns:
    forcing_df = forcing_df.drop(columns=['gauge'])

forcing_save_path = os.path.join(output_dir, f'{target_gauge_id}_hourly_forcings.csv')

forcing_df.rename(
    columns={'P': 'precip_rate', 'Temp': 'TMP_2maboveground', 'PET': 'PET_hargreaves'},
    inplace=True,
)

forcing_df.to_csv(forcing_save_path)
print(f"Saved forcings to: {forcing_save_path}")

# --- 2. Attributes CSV ---
print("Extracting Attributes...")
# Combine both attribute lists
all_attrs = list(set(var_c_list + var_c_list2))

# Add crucial metadata if present
extra_attrs = ['catchsize', 'meanelevation', 'uparea']
for attr in extra_attrs:
    if attr in ds_attr:
        all_attrs.append(attr)

# Select the specific variables we want
subset_attr = ds_attr[all_attrs]

# FIX: Check if dataset is 0-dimensional (scalar) and add a dimension back if needed
if len(subset_attr.dims) == 0:
    # We add the 'gauge' dimension back so Pandas has an index to hold onto
    subset_attr = subset_attr.expand_dims("gauge")

attr_df = subset_attr.to_dataframe()

attr_save_path = os.path.join(output_dir, f'{target_gauge_id}_attributes.csv')
attr_df.to_csv(attr_save_path)
print(f"Saved attributes to: {attr_save_path}")

# Cleanup
hourly_x.close()
attrs_ds_all.close()
print("Done.")
