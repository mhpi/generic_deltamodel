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
- `time_to_date()`: calling `.datetime()` on a `datetime.date` object raised `AttributeError` when `hr=True`; replaced with `datetime.combine(t, time())`.
- `select_subset()`: when `nt <= rho` and `warmup > 0`, time indices were clamped to 0 instead of `warmup`, producing negative NumPy indices that silently wrapped around and read wrong data.
- `select_subset()`: device placement was conditionally gated on `torch.cuda.is_available()`, causing tensors to stay on CPU even when `config['device']` specified otherwise. Now always honors `config['device']`.
- `ModelHandler._trim()`: warmup was stripped from `target` twice when output and target lengths differed, misaligning loss computation. The second removal now truncates to match lengths instead.
- `ModelHandler.load_model()`: checkpoints were loaded with `strict=False`, silently accepting incompatible architectures. Now uses `strict=True`.
- `ModelHandler.save_states()`: `torch.save` was called a second time after the `else: raise NotImplementedError` block, causing a `NameError` when more than one model was present.
- `DplModel._init_phy_model()`: used `/` (division) instead of `+` for string concatenation in a `raise ValueError`, causing a `TypeError` before the intended error could be raised.
- `BaseSampler.to_tensor()`: `self.dtype` and `self.device` were referenced but never initialized; set to `float32` / `cpu` defaults in `__init__`.
- `BaseLoader.to_tensor()`: same uninitialized `self.dtype`/`self.device` issue as `BaseSampler`; fixed with same defaults.
- `BaseTrainer.validate_config()`: checked for config keys `'rain'` and `'delta_model'` instead of `'train'` and `'model'`, causing the validator to reject all valid configs.

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
