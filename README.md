# 𝛿MG: PyTorch Differentiable Modeling Framework

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?labelColor=333333)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/dynamic/json?label=PyTorch&query=info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorch%2Fjson&logo=pytorch&color=EE4C2C&logoColor=F900FF&labelColor=333333)](https://pypi.org/project/torch/)
[![PyPI](https://img.shields.io/pypi/v/dmg?logo=pypi&logoColor=white&labelColor=333333)](https://pypi.org/project/dmg/)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&labelColor=333333)](https://github.com/astral-sh/ruff)
[![Build](https://img.shields.io/github/actions/workflow/status/mhpi/generic_deltamodel/pytest.yaml?branch=master&logo=github&label=tests&labelColor=333333)](https://github.com/mhpi/generic_deltamodel/actions/workflows/pytest.yaml)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.14868671-blue?labelColor=333333&lab)](https://doi.org/10.5281/zenodo.14868671)

---

</br>

A generic framework for building [differentiable models](https://www.nature.com/articles/s43017-023-00450-9) that seamlessly couple neural networks with process-based equations, leveraging PyTorch's auto-differentiation for efficient, GPU-accelerated optimization. This is the spiritual successor to [HydroDL](https://github.com/zhonghu17/HydroDL).

- 🤝 **Hybrid Modeling** — Combine NNs with process-based equations; learn physical model parameters directly from data.
- 🔁 **PyTorch Integration** — Efficient training, modern ML tooling, and numerical solver compatibility.
- 🧩 **Modular Architecture** — Swap in domain-specific components (models, loss functions, data loaders) with ease.
- ⚡ **Benchmarking** — Rapid deployment and replication of published [MHPI results](https://mhpi.github.io/benchmarks/#10-year-training-comparison).
- 🌊 **NextGen-ready** — [CSDMS BMI](https://csdms.colorado.edu/wiki/BMI) compliant for [NOAA-OWP](https://water.noaa.gov/about/owp)'s [NextGen Framework](https://github.com/NOAA-OWP/ngen) and AWI's [NGIAB](https://github.com/CIROH-UA/NGIAB-CloudInfra).

</br>

## Installation

```bash
pip install dmg
```

Optional extras:

```bash
pip install "dmg[hydrodl2]"    # MHPI hydrologic models (δHBV, etc.)
pip install "dmg[logging]"     # TensorBoard and W&B
pip install "dmg[tune]"        # Hyperparameter tuning (Optuna/Ray)
```

For development installs, see [setup](./docs/setup.md).

</br>

## Quick Start

Use an LSTM to learn parameters for the [HBV](https://en.wikipedia.org/wiki/HBV_hydrology_model) hydrologic model:

```python
from hydrodl2.models.hbv.hbv import Hbv
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import load_nn_model
from dmg.models.delta_models import DplModel
from example import load_config, take_data_sample

config = load_config('../example/conf/config_dhbv.yaml')

# Build differentiable model: NN learns parameters for physics model.
phy_model = Hbv(config['model']['phy'])
nn = load_nn_model(config['model'], phy_model)
dpl_model = DplModel(phy_model=phy_model, nn_model=nn)

# Load data and forward.
dataset = HydroLoader(config).dataset
sample = take_data_sample(config, dataset, days=730, basins=100)
output = dpl_model(sample)
```

Internally, `DplModel` composes the NN and physics model — the NN generates parameters, the physics model produces predictions:

```python
parameters = self.nn_model(dataset_sample['xc_nn_norm'])
predictions = self.phy_model(dataset_sample, parameters)
```

We recommend starting with the [δHBV 1.0 tutorial](./example/hydrology/example_dhbv_1_0.ipynb) ([Colab](https://colab.research.google.com/drive/19PRLrI-L7cGeYzkk2tOetULzQK8s_W7v?usp=sharing)), then exploring the full [example notebooks](https://github.com/mhpi/generic_deltamodel/tree/master/example/hydrology). See [how to run](./docs/how_to_run.md) for CLI usage.

</br>

## Use Cases

### 1. Lumped Hydrology

Lumped differentiable rainfall-runoff models [𝛿HBV 1.0](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2022WR032404) and improved [𝛿HBV 1.1p](https://essopenarchive.org/doi/full/10.22541/essoar.172304428.82707157).

### 2. Unseen Extreme Events Test with 𝛿HBV 1.1p

In the unseen extreme events spatial test, we used water years with a 5-year or lower return period peak flow from 1990/10/01 to 2014/09/30 for training, and held out the water years with greater than a 5-year return period peak flow for testing. The spatial test was conducted using a 5-fold cross-validation approach for basins in the [CAMELS dataset](https://gdex.ucar.edu/dataset/camels.html). This application has been benchmarked against LSTM and demonstrates better extrapolation abilities. Find more details and results in [Song, Sawadekar, et al. (2024)](https://essopenarchive.org/doi/full/10.22541/essoar.172304428.82707157).

![Alt text](./docs/images/extreme_temporal.png)

### 3. National- and Global-scale Distributed Modeling

A national-scale water modeling study on approximately 180,000 river reaches (with a median length of 7 km) across CONUS using the high-resolution, multiscale, differentiable water model 𝛿HBV 2.0. This model is also operating at global scales ([Ji, Song, et al., 2025](https://www.nature.com/articles/s41467-025-64367-1)) and has been used to generate high-quality, seamless simulations for both [CONUS](https://zenodo.org/records/15784945) and the [globe](https://zenodo.org/records/17552954). Find more details and results in [Song, Bindas, et al. (2025)](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024WR038928).

![Alt text](./docs/images/conus_dataset.jpg)

### 4. Global-scale Photosynthesis Modeling

Differentiable modeling has also been applied to parameterize global-scale sapflow simulations. This work is currently in development; see [Aboelyazeed et al. (2024)](https://doi.org/10.22541/au.173101418.87755465/v1) for more details.

![Alt text](./docs/images/ecosystems_global_vcmax.png)

</br>

## Documentation

| | |
|---|---|
| [Setup](./docs/setup.md) | Installation options (PyPI, pip, UV, Conda) |
| [How to Run](./docs/how_to_run.md) | CLI usage and custom model development |
| [Configuration](./docs/configuration.md) | Config file system and full settings glossary |
| [API Reference](./docs/api_reference.md) | Public API — models, loss functions, NNs, utilities |
| [Examples](./example/hydrology/) | Jupyter notebook tutorials |
| [Changelog](./docs/CHANGELOG.md) | Release history |

</br>

## Architecture

```text
src/dmg/
├── core/
│   ├── calc/                   # Metrics and calculation utilities
│   ├── data/                   # Data loaders and samplers
│   ├── logging/                # TensorBoard and W&B logging
│   ├── post/                   # Post-processing and plotting
│   └── utils/                  # Factory functions and helpers
├── models/
│   ├── criterion/              # Loss functions (MSE, NSE, KGE, ...)
│   ├── delta_models/           # Differentiable model types (DplModel, ...)
│   ├── neural_networks/        # NN architectures (LSTM, ANN, MLP, ...)
│   ├── phy_models/             # Physical model wrappers
│   └── model_handler.py        # High-level model manager
└── trainers/                   # Training orchestration
```

## Ecosystem

- **[`hydrodl2`](https://github.com/mhpi/hydrodl2)** — MHPI's suite of process-based hydrology models (lumped + distributed).
- **[`diffEcosys`](https://github.com/hydroPKDN/diffEcosys/)** — Physics-informed ML for ecosystem modeling (photosynthesis via FATES).
- **In development** — Numerical PDE solvers, [adjoint](https://doi.org/10.5194/hess-28-3051-2024) sensitivity, surrogate models, data assimilation, and more.

## Citation

This work is maintained by [MHPI](http://water.engr.psu.edu/shen/) and advised by [Dr. Chaopeng Shen](https://water.engr.psu.edu/shen/). If you find it useful, please cite:

> Shen, C., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. *Nature Reviews Earth & Environment*, 4(8), 552–567. https://doi.org/10.1038/s43017-023-00450-9

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for details.

---

*Please submit an [issue](https://github.com/mhpi/generic_deltamodel/issues) to report any questions, concerns, bugs, etc.*
