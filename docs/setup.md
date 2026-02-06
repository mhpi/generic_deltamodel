# *𝛿MG* Setup

</br>

## 1. System Requirements

𝛿MG uses PyTorch models and supports both CPU and CUDA (GPU) execution. For large-scale training, an NVIDIA GPU with CUDA support is strongly recommended:

- Windows, Linux, or macOS
- NVIDIA GPU(s) supporting CUDA (>12.0 recommended) for GPU-accelerated training

</br>

## 2. Install from PyPI

The simplest way to install 𝛿MG:

```bash
pip install dmg
```

To install with optional dependencies:

```bash
pip install "dmg[hydrodl2]"    # MHPI hydrology models (δHBV, etc.)
pip install "dmg[geo]"         # Geographical plotting (Cartopy)
pip install "dmg[logging]"     # TensorBoard and W&B logging
pip install "dmg[tune]"        # Hyperparameter tuning (Optuna, Ray Tune)
pip install "dmg[dev]"         # Development tools (ruff, pytest, pre-commit)
```

</br>

## 3. Install from Source (Development)

To develop with or contribute to 𝛿MG, install from source in developer mode so that changes to the code are immediately reflected without reinstallation.

### Clone the Repository

Open a terminal, navigate to the directory where 𝛿MG will be stored, and clone:

```shell
git clone https://github.com/mhpi/generic_deltamodel.git
```

</br>

### Create a New Environment and Install

- **UV** (**Recommended** — UV runs [much faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md) than the alternatives)

  If not already installed, run `pip install uv` or [see here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

  Create a virtual environment:

  ```bash
  uv venv --python 3.12 ./generic_deltamodel/.venv
  ```

  Activate with `source .venv/bin/activate`, then install 𝛿MG:

  ```bash
  uv pip install -e ./generic_deltamodel
  ```

- **Pip**

  Either create a Conda environment or a virtual environment in the 𝛿MG directory. For a virtual environment:

  ```bash
  python3.12 -m venv ./generic_deltamodel/.venv
  ```

  Activate with `source .venv/bin/activate`. Then install 𝛿MG:

  ```bash
  pip install -e ./generic_deltamodel
  ```

  This will also install all dependencies (see `./generic_deltamodel/pyproject.toml`).

- **Conda**

  Create a base environment for Python versions 3.9–3.13:

  ```shell
  conda env create -n dmg python=3.x
  ```

  Activate the environment with `conda activate dmg`.

  To install 𝛿MG (developer mode), we recommend using pip inside the Conda environment:

  ```bash
  pip install -e ./generic_deltamodel
  ```

  Note: `conda develop` is deprecated and is not recommended.

  There is a known issue with CUDA failing on new Conda installations. To verify, open a Python instance and check that CUDA is available with PyTorch:

  ```python
  import torch
  print(torch.cuda.is_available())
  ```

  If CUDA is not available, uninstall PyTorch from the environment and reinstall according to your system [specification](https://pytorch.org/get-started/locally/).

</br>

### Install Optional Dependencies

- **hydrodl2**

  To work with hydrological models like δHBV, δHBV 2.0, etc. developed by MHPI, the [hydrodl2 repository](https://github.com/mhpi/hydrodl2) of physical hydrological models (to be coupled with NNs in 𝛿MG) will need to be installed alongside 𝛿MG. To simply use the models:

  ```bash
  pip install "./generic_deltamodel[hydrodl2]"

  # or

  uv pip install "./generic_deltamodel[hydrodl2]"
  ```

  If you would like to develop in or contribute to hydrodl2, clone the [hydrodl2 master branch](https://github.com/mhpi/hydrodl2) from GitHub and install in developer mode (similar to 𝛿MG):

  ```bash
  git clone git@github.com:mhpi/hydrodl2.git
  pip install -e ./hydrodl2

  # or

  uv pip install -e ./hydrodl2
  ```

  Note, developer mode will ensure hydrodl2 won't need to be reinstalled whenever you make changes.

- **Geo Plotting**

  For geographical plotting features (e.g., mapping model metrics spatially) available in `./examples/`, install dependencies with:

  ```bash
  pip install "./generic_deltamodel[geo]"

  # or

  uv pip install "./generic_deltamodel[geo]"
  ```

- **Development**

  For developing with and/or making contributions to 𝛿MG, some linting and test packages can be installed with:

  ```bash
  pip install "./generic_deltamodel[dev]"

  # or

  uv pip install "./generic_deltamodel[dev]"
  ```

</br>

---

*Please submit an [issue](https://github.com/mhpi/generic_deltamodel/issues) on GitHub to report any questions, concerns, bugs, etc.*
