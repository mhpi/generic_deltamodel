# *dMG* Setup

## 1. System Requirements

dMG uses PyTorch models requiring CUDA support only available with NVIDIA GPUs. Therefore, this package requires

- Windows or Linux
- NVIDIA GPU(s) supporting CUDA (>12.0 recommended)

## 2. Steps for Setup

To run dMG as a framework or package, it must be installed as a package. This can be done with Conda, Pip, or UV (**recommended**; UV runs [much faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md) than the alternatives), and we encourage using developer mode so that any changes to package code is immediately reflected without reinstallation.

### Clone the Repository

- Open a terminal on your system, navigate to the directory where dMG will be stored, and clone:
  
    ```shell
    git clone https://github.com/mhpi/generic_deltamodel.git
    ```

- Your install directory should now look like:

    .
    ├── generic_deltaModel/
    └── hydroDL2/   *(optional, see [here](#install-optional-dependencies))*

### Create a New ENV and Install

- Conda

  - A yaml setup file is included for Conda setups:

    ```shell
    conda env create --file ./generic_deltaModel/env/dmg_env.yaml
    ```

    or
  
    ```shell
    conda env create --prefix path/to/env --file ./generic_deltaModel/env/dmg_env.yaml
    ```

  - Then the ENV can be activated like `conda activate dmg`.

  - To install dMG (developer mode), you can run `conda develop ./generic_deltamodel`. However, this is depreciated and we would instead recommend `pip install -e ./generic_deltamodel`.

  - There is a known issue with CUDA failing on new Conda installations. To confirm this, open a python instance and check that CUDA is available with PyTorch:

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

  - If CUDA is not available, uninstall PyTorch from the env and reinstall according to your system specifications [here](https://pytorch.org/get-started/locally/).

- Pip

  - Either create a Conda ENV or virtual ENV in the dMG directory. For a virtual ENV installation,
  
    ```shell
    python3.12 -m venv ./generic_deltamodel/.venv
    ```

    and activate with `source .venv/bin/activate`. The Python version will need to be installed if not present.

  - In the ENV, install dMG like
  
    ```bash
    pip install -e ./generic_deltamodel
    ```

    This will also install all dependencies (see `./generic_deltamodel/pyproject.toml`).

- UV (**Recommended**)

  - If not already installed, run `pip install uv` or see [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

  - Create a virtual environment:

    ```bash
    uv venv --python 3.12 ./generic_deltamodel/.venv
    ```

    and activate with `source .venv/bin/activate`

  - Install dMG with Pip. Pip operates nearly 1:1 inside of UV. ~All pip actions can be reached by `uv pip`. Therfore, we install dMG like

    ```bash
    uv pip install -e ./generic_deltamodel
    ```

### Install Optional Dependencies

- HydroDL 2.0

  - To work with hydrological models like δHBV, δHBV 2.0, etc. developed by MHPI, the [HydroDL 2.0 repository](https://github.com/mhpi/hydroDL2) of physical hydrological models (to be coupled with NNs in dMG) will need to be installed alongside dMG (again, in your ENV of choice). There are two ways to do this depending on your intentions. To simply use the models, you can run

  ```bash
  pip install "./generic_deltamodel[hydrodl2]"
  ```

  or

  ```bash
  uv pip install "./generic_deltamodel[hydrodl2]"
  ```

  For conda installations, it is most straightforward to use pip as illustrated above.

  If you wish to develop in or contribute to HydroDL 2.0, clone the [HydroDL 2.0 master branch](https://github.com/mhpi/hydroDL2) from GitHub and install in developer mode (similar to dMG):

  ```bash
  git clone git@github.com:mhpi/hydroDL2.git
  pip install -e ./hydroDL2
  ```

  or

  ```bash
  uv pip install -e ./hydroDL2
  ```

  Note, developer mode will ensure hydroDL2 won't need to be reinstalled whenever you make changes.

- Development

  - For developing with and/or making contributions to dMG some linting and test packages can be installed wit

  ```bash
  pip install "./generic_deltamodel[dev]"
  ```

  or

  ```bash
  uv pip install "./generic_deltamodel[dev]"
  ```

  If you wish to install with conda, you will need to manually install 'dev' packages (e.g., ruff, pytest) listed in `./generic_deltamodel/pyproject.toml`.

---

*Please submit an [issue](https://github.com/mhpi/generic_deltaModel/issues) on GitHub to report any questions, concerns, bugs, etc.*
