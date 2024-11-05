## Generic, Scalable Differentiable Modeling Framework

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/pypi/l/ruff.svg)](https://github.com/astral-sh/ruff/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)]()
[![Actions status](https://github.com/astral-sh/ruff/workflows/CI/badge.svg)](https://github.com/astral-sh/ruff/actions)

#### Backbone for *HydroDL2.0* w/ hydrology models (1.0 [here](https://github.com/mhpi/hydroDL))

A domain-agnostic Python framework for developing trainable differentiable models.
Following as a generalization of *HydroDL*, *DeltaModel* (or `generic_deltaModel`) aims
to expand differentiable parameter learning capabilities beyond hydrology. 

Those wishing to run hydrologic tasks tied to MHPI's research can couple *DeltaModel*
with physics models from *HydroDL2.0* ([`hydroDL2`](https://github.com/mhpi/hydroDL2)) and data processors from *HydroData* 
([`hydro_data_dev`](https://github.com/mhpi/hydro_data_dev)). The *hydroDL2.0* package also
contains hydrology-oriented modules (e.g., variational data assimilation) as augmentations to *DeltaModel* differential
model capabilities.


---
See [here](https://github.com/orgs/mhpi/projects/4) for a roadmap of planned additions and improvements.

<!-- ### Maintainers:
See Pyproject.toml for information. -->

### Contributing:
We ask all changes to this repo be made through a fork and PR.


### Repository Structure:

├───deltaMod/
│   │   __main__.py
│   ├───conf/
│   │   │   config.py
│   │   │   config.yaml
│   │   ├───hydra/
│   │   └───observations/
│   ├───core/
│   │   ├───calc/
│   │   ├───data/
│   │   └───utils/
│   ├───models/
│   │   │   differentiable_model.py
│   │   │   model_handler.py
│   │   ├───loss_functions/
│   │   └───neural_networks/
│   └───trainers/
├───docs/
├───envs/
└───example/
