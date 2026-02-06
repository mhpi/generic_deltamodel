# Changelog

All notable changes to 𝛿MG are documented here.

---

## Unreleased

### Added
- Comprehensive testing suite with regression, forward pass, gradient, parameter, convergence, and model handler tests.
- Experiment loggers (TensorBoard, Weights & Biases).
- Persistent state caching — save and load hidden/cell states and physics model storages to disk.
- Pydantic-based configuration validation.
- Formalized HydroDL LSTM (`CudnnLstmModel`) with improved state management.
- GEFS bias correction warm-start example.
- CI/CD workflows for automated testing and wheel builds.

### Changed
- Path overhaul for cleaner module resolution.
- LSTM improvements for CPU/GPU parity.
- Removed Cartopy from core dependencies (moved to `[geo]` optional).

### Fixed
- Bug patches for daily dHBV 2.0 runs.
- Multi-timescale NextGen integration fixes.

---

## v1.3.1 — 2025-09-03

### Changed
- Updated license.
- Spatial testing improvements for PUB/PUR experiments.

---

## v1.3.0 — 2025-06-10

### Added
- Ray Tune hyperparameter tuning support (`[tune]` optional dependency).
- Spatial testing framework (PUB and PUR cross-validation).

### Changed
- Transition to lowercase package name standard (`dmg`).
- README and documentation updates.

---

## v1.2.1 — 2025-05-09

### Added
- CSDMS BMI compliance for NextGen National Water Modeling Framework integration.
- Liquid package backend — installable as a proper Python package via pip.

### Changed
- Overhauled import structure for cleaner subpackage management.

---

## v1.2.0 — 2025-02-13

### Added
- Complete tutorial overhaul: δHBV 1.0, δHBV 1.1p, and δHBV 2.0 example notebooks.
- Geo-plotting support for spatial metric visualization.
- Updated loss functions and post-processing utilities.

### Changed
- Multi-scale data loader and trainer improvements for dHBV 2.0.

---

## v1.1.0 — 2025-02-06

### Added
- δHBV 2.0 multi-scale, multi-timescale differentiable water model.
- Multi-scale data loaders, samplers, and trainers.
- Distributed Data Parallel (DDP) preparation.

### Changed
- Module loaders made more robust with universal loader support.

---

## v1.0.0 — 2024-12-04

Initial public release of 𝛿MG.

### Features
- Differentiable Parameter Learning (`DplModel`) coupling neural networks with physics models.
- `ModelHandler` for high-level model management and multimodel ensembles.
- Neural network architectures: LSTM, ANN, MLP, CuDNN LSTM.
- Loss functions: MSE, RMSE, NSE, KGE, and variants.
- Hydra-based configuration management.
- Support for δHBV 1.0 and δHBV 1.1p hydrological models via hydrodl2.
- Data loaders and samplers for CAMELS hydrological datasets.
- Example notebooks for hydrology use cases.
