## add a class
class HydroScaler:
  def __init__(self, attrLst, seriesLst,log_norm_cols):
    self.log_norm_cols = log_norm_cols
    self.attrLst = attrLst
    self.seriesLst = seriesLst
    self.stat_dict = None
    

  def fit(self, attrdata, seriesdata):
    self.stat_dict = getStatDic(
      log_norm_cols=self.log_norm_cols,
      attrLst=self.attrLst,
      attrdata=attrdata,
      seriesLst=self.seriesLst,
      seriesdata=seriesdata,
    )

  def transform(self, data, var_list,):

    norm_data = transNormbyDic(
      data, var_list, self.stat_dict, log_norm_cols = self.log_norm_cols, to_norm=True)

    return norm_data

  def fit_transform(self, attrdata, seriesdata):
    self.fit(attrdata, seriesdata)
    attr_norm = self.transform(attrdata, self.attrLst)
    series_norm = self.transform(seriesdata, self.seriesLst)
    return attr_norm, series_norm

def get_parameters_from_model(model, z_time_first, n_par, mu):
    """
    Use the inverse head (lstminv) to produce HBV parameters and (optionally) routing weights.
    Returns:
      pars_hbv: [T, B, n_par, mu]  in [0,1]
      rtwts:    [B, 2] or None     (routing parameters per basin)
    """
    with torch.no_grad():
        if not (hasattr(model, "lstminv") and callable(model.lstminv)):
            raise RuntimeError("Model has no callable 'lstminv' head to produce parameters.")

        out = model.lstminv(z_time_first)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("lstminv did not return a tensor.")

        if out.dim() == 4:
            # Assume already [T,B,n_par,mu]; no routing in this case
            return out.to(device), None

        if out.dim() != 3:
            raise RuntimeError(f"Unexpected lstminv rank {out.dim()} (expected 3 or 4).")

        T, B, F = out.shape
        # Cases:
        #  F == n_par       -> [T,B,n_par,1]
        #  F == n_par*mu    -> [T,B,n_par,mu]
        #  F == n_par*mu+2  -> split last 2 as routing, reshape first part to [T,B,n_par,mu]
        if F == n_par:
            return out.unsqueeze(-1).to(device), None
        elif F == n_par * mu:
            return out.view(T, B, n_par, mu).to(device), None
        elif F == n_par * mu + 2:
            flat = out[:, :, : n_par * mu]
            rts  = out[-1, :, n_par * mu : n_par * mu + 2]  # take routing from last time step -> [B,2]
            pars = flat.view(T, B, n_par, mu).to(device)
            return pars, rts.to(device)
        else:
            raise RuntimeError(f"Unexpected parameter feature size {F}; expected {n_par}, {n_par*mu}, or {n_par*mu+2}.")

def plot_ensemble_forecast(
    gage_id,
    out,
    start_date='2010-01-01',
    n_ensembles=5,
    obs=None,
    sim=None,
    title=None,
    save_path=None
):
    """
    Plot GEFS ensemble forecasts and historical simulation with observed streamflow.
    """
    # Load ensemble forecasts
    ens_preds = []
    for ens_id in range(n_ensembles):
        fpath = os.path.join(out, f"ens{ens_id}_Qs")
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue
        try:
            data = np.loadtxt(fpath, delimiter=',')
            data = np.squeeze(data)
            ens_preds.append(data)
            if data.size == 0 or np.all(np.isnan(data)):
              print(f"Warning: {fpath} is empty or all NaN.")
              continue
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")

    if len(ens_preds) == 0:
        raise ValueError("No ensemble forecast files loaded.")

    ens_preds = np.stack(ens_preds, axis=0)  # [n_ensembles, 15]
    forecast_len = ens_preds.shape[1]
    history_len = 60  # 2 months

    # Time axis: 60 days before + 15 forecast
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start=start_date - pd.Timedelta(days=history_len), periods=history_len + forecast_len)
 
    obs = np.squeeze(obs)
    sim = np.squeeze(sim) 
    # Extend ensemble forecasts with NaN padding before forecast
    padded_ensembles = np.full((n_ensembles, history_len + forecast_len), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    # Compute KGE
    kge_sim = calKGE(sim, obs)[0]
    kge_ens = [calKGE(ens_preds[i], obs[history_len:])[0] for i in range(n_ensembles)]
    kge_mean = calKGE(np.mean(ens_preds, axis=0), obs[history_len:])[0]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dates, sim, 'r-', label=f'Sim: KGE={kge_sim:.3f}')
    plt.plot(dates, obs, 'k-', linewidth=2, label="Observed")

    for i in range(n_ensembles):
        plt.plot(dates, padded_ensembles[i], alpha=0.4, label=f'Ens {i}: KGE={kge_ens[i]:.3f}')

    mean_ens = np.nanmean(padded_ensembles, axis=0)
    p5 = np.nanpercentile(padded_ensembles, 5, axis=0)
    p95 = np.nanpercentile(padded_ensembles, 95, axis=0)

    plt.plot(dates, mean_ens, 'k--', label=f'Ens Mean: KGE={kge_mean:.3f}')
    plt.fill_between(dates, p5, p95, where=~np.isnan(p5), color='gray', alpha=0.3, label='Ensemble CI')

    plt.axvspan(dates[0], dates[history_len - 1], color='blue', alpha=0.1, label='Simulation Period')
    plt.axvspan(dates[history_len], dates[-1], color='orange', alpha=0.2, label='Forecast Period')

    plt.title(title or f"GEFS Forecast – Gage {gage_id}")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (mm/day)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from hydrodl2.models.hbv import HBV1_1p as h1pp
from src.dmg.core.utils import Dates 

# ---------------- SETTINGS ---------------- #
GAGE_ID = 1022500
START_DATE = pd.to_datetime("2014-09-01")
HORIZON = 15
N_ENSEMBLES = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

staind   = -1
tdRep    = [1, 3, 13]
BUFFTIME = 365
routing  = True
comprout = False
dydrop   = 0.0

# ---------------- Load pretrained model ---------------- # 
import torch
ckpt = torch.load("./output/camels_531/train1999-2008/no_multi/CudnnLstmModel_E50_R365_B100_H256_n16_noLn_noWU_111111/Hbv_1_1p/NseBatchLoss/3dyn/parBETA_parK0_parBETAET/train_state_Ep50.pt", map_location="cpu")
print(type(ckpt))
  
modelFile = os.path.join(config["model_path"], "dHbv_1_1p_Ep" + str(config["train"]["epochs"]) + ".pt")
 
model = torch.load(modelFile, map_location=device)
model.to(device)
model.eval()
 
if not (hasattr(model, "lstminv") and callable(model.lstminv)):
    raise RuntimeError("Loaded model has no callable lstminv head")

# ---------------- Select basin index ---------------- #
gage_ids = np.load(config["observations"]["gage_info"], allow_pickle=True)

subset_file = config["observations"]["subset_path"]
with open(subset_file, "r") as f:
    content = f.read().strip()
if content.startswith("["):
    gage_ids_531 = json.loads(content)
else:
    gage_ids_531 = np.loadtxt(subset_file, dtype=int).tolist()

if config["observations"]["name"] == "camels_671":
    basin_idx = list(gage_ids).index(GAGE_ID)
elif config["observations"]["name"] == "camels_531":
    basin_idx = list(gage_ids_531).index(GAGE_ID)
else:
    raise ValueError("Unsupported dataset")

print(f"Selected basin {GAGE_ID} at index {basin_idx}")

# ---------------- Time axis ---------------- #
timesteps = Dates(config["simulation"], config["delta_model"]["rho"]).batch_daily_time_range
print("timesteps (full):", timesteps[:3], "...", timesteps[-3:], f"(len={len(timesteps)})")

sidx = np.where(timesteps == START_DATE)[0][0]
eidx = sidx + HORIZON
fcst_time = timesteps[sidx:eidx]
print("Forecast window:", fcst_time[0], "to", fcst_time[-1])

# ---------------- Dataset access ---------------- #
dataset = data_loader.dataset
print("Dataset keys:", dataset.keys())

# Observations
# Correct observation slice (time-first tensor)
obs = dataset["target"][sidx:eidx, basin_idx, 0].cpu().numpy()
print("obs shape after fix:", obs.shape)  # should be (15,)


# ---------------- HBV setup ---------------- #
hbv = h1pp.HBV().to(device)

# attributes
attrs_one = dataset["c_phy"][basin_idx:basin_idx+1, :].cpu().numpy()
attrs_one[np.isnan(attrs_one)] = 0.0
attrs_one = np.expand_dims(attrs_one, axis=1)

# forcings history
hist_forc_raw = dataset["x_phy"][basin_idx:basin_idx+1, :sidx, :].cpu().numpy()
hist_forc_raw[np.isnan(hist_forc_raw)] = 0.0

val_c_all_hist = np.repeat(attrs_one, hist_forc_raw.shape[1], axis=1)
z_hist = np.concatenate([hist_forc_raw, val_c_all_hist], axis=2)

def to_time_first(x_np):
    return torch.from_numpy(np.transpose(x_np, (1, 0, 2))).float().to(device)

x_hist_t = to_time_first(hist_forc_raw)
z_hist_t = to_time_first(z_hist)

# --- Get parameters + routing weights from trained NN ---
pars_hist, rtwts_hist = get_parameters_from_model(
    model,
    z_hist_t,
    n_par=14,
    mu=config["delta_model"]["phy_model"]["nmul"]
)

print("pars_hist shape:", pars_hist.shape)
print("rtwts_hist shape:", None if rtwts_hist is None else rtwts_hist.shape)

# --- Warm-up states ---
with torch.no_grad():
    Qs_hist, sp, mw, sm, suz, slz = hbv(
        x=x_hist_t[:, :, :3],
        parameters=pars_hist,
        staind=staind,
        tdlst=tdRep,
        mu=config["delta_model"]["phy_model"]["nmul"],
        muwts=None,
        rtwts=rtwts_hist,
        bufftime=BUFFTIME,
        outstate=True,
        instate=False,
        routOpt=routing,
        comprout=comprout,
        dydrop=dydrop,
    )
warm_states = (sp, mw, sm, suz, slz)

# ---------------- Ensemble forecasts ---------------- #
cold_fcst, warm_fcst = [], []

for ens_id in range(N_ENSEMBLES):
    ens_folder = f"_ens0{ens_id+1}"
    f_path = os.path.join(gefs_dir, ens_folder, f"{GAGE_ID:08d}.txt")
    if not os.path.exists(f_path):
        print(f"[warn] Missing GEFS file: {f_path}")
        continue

    df = pd.read_csv(f_path, sep=r"\s+", header=0)
    if "tmean(C)" not in df.columns and {"tmax(C)", "tmin(C)"}.issubset(df.columns):
        df["tmean(C)"] = (df["tmax(C)"] + df["tmin(C)"]) / 2.0

    df = df.rename(columns={
        "Year": "year","Mnth": "month","Day": "day",
        "prcp(mm/day)": "prcp","tmean(C)": "tmean","pet(mm/day)": "pet"})
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    idx_list = df.index[df["date"] == START_DATE].to_list()
    if not idx_list:
        print(f"[warn] GEFS {f_path} does not contain {START_DATE.date()} — skip")
        continue
    sidx_fc = idx_list[0]

    fc_block = df.iloc[sidx_fc : sidx_fc + HORIZON]
    if len(fc_block) < HORIZON:
        print(f"[warn] Not enough GEFS days from {START_DATE.date()} — skip")
        continue

    fc_forc_raw = fc_block[varF].to_numpy()[np.newaxis, :, :]
    fc_forc_raw[np.isnan(fc_forc_raw)] = 0.0

    val_c_all_fc = np.repeat(attrs_one, fc_forc_raw.shape[1], axis=1)
    z_fc = np.concatenate([fc_forc_raw, val_c_all_fc], axis=2)

    x_fc_t = to_time_first(fc_forc_raw)

    with torch.no_grad():
        Qs_cold = hbv(
            x=x_fc_t[:, :, :3],
            parameters=pars_hist,
            staind=staind,
            tdlst=tdRep,
            mu=config["delta_model"]["phy_model"]["nmul"],
            muwts=None,
            rtwts=rtwts_hist,
            bufftime=0,
            outstate=False,
            instate=False,
            routOpt=routing,
            comprout=comprout,
            dydrop=dydrop,
        )

        Qs_warm = hbv(
            x=x_fc_t[:, :, :3],
            parameters=pars_hist,
            staind=staind,
            tdlst=tdRep,
            mu=config["delta_model"]["phy_model"]["nmul"],
            muwts=None,
            rtwts=rtwts_hist,
            bufftime=0,
            outstate=False,
            instate=True,
            init_states=warm_states,
            routOpt=routing,
            comprout=comprout,
            dydrop=dydrop,
        )

    cold_fcst.append(Qs_cold[:, 0, 0].detach().cpu().numpy())
    warm_fcst.append(Qs_warm[:, 0, 0].detach().cpu().numpy())

cold_fcst = np.stack(cold_fcst, axis=1)
warm_fcst = np.stack(warm_fcst, axis=1)

# ---------------- Plotting ---------------- #
def plot_ensemble_forecast(t, cold, warm, obs=None, title="", save_path=None):
    plt.figure(figsize=(10, 5))
    for i in range(cold.shape[1]):
        plt.plot(t, cold[:, i], color="blue", alpha=0.25, label="Cold start" if i == 0 else "")
    for i in range(warm.shape[1]):
        plt.plot(t, warm[:, i], color="red", alpha=0.25, label="Warm start" if i == 0 else "")
    plt.plot(t, np.median(cold, axis=1), color="blue", lw=2)
    plt.plot(t, np.median(warm, axis=1), color="red", lw=2)
    if obs is not None:
        plt.plot(t, obs, "k-", lw=2, label="Observed")
    plt.title(title)
    plt.ylabel("Streamflow (mm/day)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

plot_ensemble_forecast(
    fcst_time,
    cold_fcst,
    warm_fcst,
    obs=obs,
    title=f"Cold vs Warm Start Ensemble — Gage {GAGE_ID}",
    save_path=f"figs/cold_warm_{GAGE_ID}.png",
)


# ==== Warm-start GEFS forecast (2012-10-01 → 2014-08-31 warm-up, then GEFS 2014-09-01 → 2014-09-15) ====

import os
import json
import numpy as np
import pandas as pd
import torch
from src.dmg.core.utils import Dates
import helper as hp
from hydrodl2.models.hbv import HBV1_1p as h1pp

# ---------------- SETTINGS ---------------- #
GAGE_ID     = 1022500
START_DATE  = pd.to_datetime("2014-09-01")
HORIZON     = 15
N_ENSEMBLES = 5
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# GEFS files have these column names after rename
VAR_F = ["prcp", "tmean", "pet"]
GEFS_DIR = "/u/st/dr/awwood/aw-ciroh-proj/projects/dl_da/daymet-gefs-camels-gII"

# HBV options (match training/config)
staind   = -1
tdRep    = [1, 3, 13]
BUFFTIME = config['delta_model']['phy_model']['warm_up']
routing  = config['delta_model']['phy_model']['routing']
comprout = False
dydrop   = 0.0

# ---------------- Basin index ---------------- #
gage_ids = np.load(config["observations"]["gage_info"], allow_pickle=True)
subset_file = config["observations"]["subset_path"]
with open(subset_file, "r") as f:
    content = f.read().strip()
gage_ids_subset = json.loads(content) if content.startswith("[") else np.loadtxt(subset_file, dtype=int).tolist()

if config["observations"]["name"] == "camels_671":
    basin_idx = list(gage_ids).index(GAGE_ID)
elif config["observations"]["name"] == "camels_531":
    basin_idx = list(gage_ids_subset).index(GAGE_ID)
else:
    raise ValueError(f"Unsupported dataset {config['observations']['name']}")

# ---------------- Time axis ---------------- #
timesteps = Dates(config["simulation"], config["delta_model"]["rho"]).batch_daily_time_range
sidx = np.where(timesteps == START_DATE)[0][0]
eidx = sidx + HORIZON

# ---------------- Dataset ---------------- #
dataset = data_loader.dataset

# ---------------- HBV setup ---------------- #
hbv = h1pp.HBV().to(device)

def to_time_first(x_torch):
    return x_torch.permute(1, 0, 2).float().to(device)  # [T,B,F]

# ---------------- Static attributes ---------------- #
if "c_nn_norm" in dataset:
    c_nn_all = dataset["c_nn_norm"]
else:
    print("[info] c_nn_norm not found, normalizing c_nn manually (z-score)")
    c_nn = dataset["c_nn"].float()
    mask = ~torch.isnan(c_nn)
    mean = torch.sum(torch.where(mask, c_nn, torch.zeros_like(c_nn)), dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)
    var  = torch.sum(torch.where(mask, (c_nn - mean)**2, torch.zeros_like(c_nn)), dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)
    std  = torch.sqrt(var)
    c_nn_all = (torch.where(mask, c_nn, mean) - mean) / (std + 1e-6)
c_nn_hist = c_nn_all[basin_idx:basin_idx+1, :].to(device)

# ---------------- History (up to START_DATE) ---------------- #
# Attributes for parameterization
xc_nn_hist = dataset["xc_nn_norm"][basin_idx:basin_idx+1, :sidx, :].clone().float()
xc_nn_hist[torch.isnan(xc_nn_hist)] = 0.0
if "c_nn_norm" in dataset:
    c_nn_hist = dataset["c_nn_norm"][basin_idx:basin_idx+1, :].to(device).float()
else:
    c_nn_hist = c_nn_all[basin_idx:basin_idx+1, :].to(device).float()

hist_dict = {
    "xc_nn_norm": to_time_first(xc_nn_hist),  # [Th,1,F]
    "c_nn_norm": c_nn_hist,                   # [1,F_attr]
}

# --- Get parameters from trained NN ---
pars_hist, rtwts_hist = get_parameters_from_model(
    model.model_dict["Hbv_1_1p"],
    hist_dict,
    n_par=14,
    mu=config["delta_model"]["phy_model"]["nmul"],
    device=device,
)
print("pars_hist:", pars_hist.shape)
print("rtwts_hist:", None if rtwts_hist is None else rtwts_hist.shape)

# --- Warm-up HBV with RAW forcings (Daymet) ---
with torch.no_grad():
    forc_hist = dataset["x_phy"][basin_idx:basin_idx+1, :sidx, :3]  # raw P,T,PET
    forc_hist = forc_hist.permute(1, 0, 2).float().to(device)       # [T,1,3]
    forc_hist[torch.isnan(forc_hist)] = 0.0

    Qs_hist, sp, mw, sm, suz, slz = hbv(
        x=forc_hist,
        parameters=pars_hist,
        staind=staind, tdlst=tdRep,
        mu=config["delta_model"]["phy_model"]["nmul"],
        muwts=None, rtwts=rtwts_hist,
        bufftime=BUFFTIME,
        outstate=True, instate=False,
        routOpt=routing, comprout=comprout, dydrop=dydrop,
    )
warm_states = (sp, mw, sm, suz, slz)
print("Qs_hist NaNs:", torch.isnan(Qs_hist).sum().item(),
      "min:", float(Qs_hist.min()), "max:", float(Qs_hist.max()))
print("forcing stats:",
      "prcp", float(forc_hist[:,0,0].min()), float(forc_hist[:,0,0].max()),
      "tmean", float(forc_hist[:,0,1].min()), float(forc_hist[:,0,1].max()),
      "pet", float(forc_hist[:,0,2].min()), float(forc_hist[:,0,2].max()))

print("pars_hist stats:",
      float(pars_hist.mean()), float(pars_hist.min()), float(pars_hist.max()))

print("Qs_hist first 10:", Qs_hist[:10,0,0].detach().cpu().numpy())

for name, st in zip(["sp","mw","sm","suz","slz"], [sp,mw,sm,suz,slz]):
    print(name, "nan#", torch.isnan(st).sum().item(),
          "min", float(st.min()), "max", float(st.max()))


# ---------------- Warm-start ensemble forecast ---------------- #
def run_warm_forecast_for_ensemble(hbv, pars_hist, rtwts_hist, warm_states,
                                   gage_id, ens_id, start_date, horizon,
                                   varF, device):
    ens_folder = f"ens0{ens_id+1}"
    f_path = os.path.join(GEFS_DIR, ens_folder, f"{gage_id:08d}.txt")
    if not os.path.exists(f_path):
        raise ValueError(f"Missing GEFS file: {f_path}")
        
    df = pd.read_csv(f_path, sep=r"\s+", header=0)
    df = df.rename(columns={
        "Year":"year","Mnth":"month","Day":"day",
        "prcp(mm/day)":"prcp","tmean(C)":"tmean","pet(mm/day)":"pet"
    })
    df["date"] = pd.to_datetime(df[["year","month","day"]])

    idx_list = df.index[df["date"] == start_date].to_list()
    if not idx_list: raise ValueError(f"Starting GEFS date {start_date.date()} not exist!") 
    fc_block = df.iloc[idx_list[0] : idx_list[0] + horizon]
    if len(fc_block) < horizon: raise ValueError("Extracted GEFS forcing not enough!")

    raw_np = fc_block[varF].to_numpy().astype(np.float32)
    raw_np[:, 0] = np.maximum(raw_np[:, 0], 0.0)  # prcp
    raw_np[:, 2] = np.maximum(raw_np[:, 2], 0.0)  # pet
    forc_raw = torch.tensor(raw_np[np.newaxis, :, :],
                            dtype=torch.float32, device=device)
    forc_raw[torch.isnan(forc_raw)] = 0.0

    pars_fc  = pars_hist[-1:].repeat(horizon, 1, 1, 1)
    rtwts_fc = rtwts_hist

    with torch.no_grad():
        Qs_warm = hbv(
            x=forc_raw.permute(1, 0, 2),
            parameters=pars_fc,
            staind=staind, tdlst=tdRep,
            mu=config["delta_model"]["phy_model"]["nmul"],
            muwts=None, rtwts=rtwts_fc,
            bufftime=0,
            outstate=False, instate=True,
            init_states=warm_states,
            routOpt=routing, comprout=comprout, dydrop=dydrop,
        )
    return Qs_warm[:, 0, 0].detach().cpu().numpy()

# ---------------- Run ensembles ---------------- #
warm_fcst = []
for ens_id in range(N_ENSEMBLES):
    qs_warm = run_warm_forecast_for_ensemble(
        hbv, pars_hist, rtwts_hist, warm_states,
        GAGE_ID, ens_id, START_DATE, HORIZON, VAR_F, device
    ) 
    warm_fcst.append(qs_warm)

warm_fcst = np.stack(warm_fcst, axis=0)
print("warm_fcst shape:", warm_fcst.shape)

# ---------------- Plotting ---------------- #
history_len  = 62   # last 2 months before START_DATE
forecast_len = HORIZON

obs_full_window = (
    dataset["target"][sidx - history_len : eidx, basin_idx, 0]
    .detach().cpu().numpy()
)

sim_full_window = np.full(history_len + forecast_len, np.nan)
sim_full_window[:history_len] = obs_full_window[:history_len]

plot_ensemble_forecast(
    gage_id=GAGE_ID,
    start_date=START_DATE,
    obs=obs_full_window,
    sim=sim_full_window,
    ens_preds=warm_fcst,
    history_len=history_len,
    save_path=f"figs/warm_GEFS_{GAGE_ID}.png",
)
