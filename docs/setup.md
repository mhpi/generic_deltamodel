# *𝛿MG* Setup

</br>

## 1. System Requirements

𝛿MG uses PyTorch models requiring CUDA support only available with NVIDIA GPUs. Therefore, this package requires

- Windows or Linux
- NVIDIA GPU(s) supporting CUDA (>12.0 recommended)

</br>

## 2. Steps for Setup

To run 𝛿MG as a framework or package, it must be installed as a package. This can be done with Conda, Pip, or UV (**recommended**; UV runs [much faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md) than the alternatives), and we encourage using developer mode so that any changes to package code is immediately reflected without reinstallation.

### Clone the Repository

- Open a terminal on your system, navigate to the directory where 𝛿MG will be stored, and clone:

  ```shell
  git clone https://github.com/mhpi/generic_deltamodel.git
  ```

- Your install directory should now look like:

    ```text
    .
    ├── generic_deltamodel/
    └── hydrodl2/   *
    ```

  **(optional for development, [see here](#install-optional-dependencies))*

</br>

### Create a New ENV and Install

- Conda

  - To create a base ENV for Python versions 3.9-3.13 or higher,

    ```shell
    conda env create -n dmg python=3.x
    ```

  - Then the ENV can be activated like `conda activate dmg`.

  - To install 𝛿MG (developer mode), you can run `conda develop ./generic_deltamodel`. However, this is depreciated and we would instead recommend `pip install -e ./generic_deltamodel`. Note, required packages may need to be downloaded manually if choosing to use conda develop (see `dependencies` in `./generic_deltamodel/pyproject.toml`).

  - There is a known issue with CUDA failing on new Conda installations. To confirm this, open a python instance and check that CUDA is available with PyTorch:

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

  - If CUDA is not available, uninstall PyTorch from the env and reinstall according to your system [specification](https://pytorch.org/get-started/locally/).

- Pip

  - Either create a Conda ENV or virtual ENV in the 𝛿MG directory. For a virtual ENV installation,

    ```bash
    python3.12 -m venv ./generic_deltamodel/.venv
    ```

    and activate with `source .venv/bin/activate`. The Python version will need to be installed if not present.

  - In the ENV, install 𝛿MG like

    ```bash
    pip install -e ./generic_deltamodel
    ```

    This will also install all dependencies (see `./generic_deltamodel/pyproject.toml`).

- UV (**Recommended**)

  - If not already installed, run `pip install uv` or [see here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

  - Create a virtual environment:

    ```bash
    uv venv --python 3.12 ./generic_deltamodel/.venv
    ```

    and activate with `source .venv/bin/activate`

  - Install 𝛿MG with Pip. Pip operates nearly 1:1 inside of UV. ~All pip actions can be reached by `uv pip`. Therfore, we install 𝛿MG like

    ```bash
    uv pip install -e ./generic_deltamodel
    ```

### Install Optional Dependencies

- hydrodl2

  - To work with hydrological models like δHBV, δHBV 2.0, etc. developed by MHPI, the [hydrodl2 repository](https://github.com/mhpi/hydrodl2) of physical hydrological models (to be coupled with NNs in 𝛿MG) will need to be installed alongside 𝛿MG (again, in your ENV of choice). There are two ways to do this depending on your intentions. To simply use the models, you can run

  ```bash
  pip install ./generic_deltamodel[hydrodl2]

  # or

  uv pip install ./generic_deltamodel[hydrodl2]
  ```

  For conda installations, it is most straightforward to use pip as illustrated above.

  If you would like to develop in or contribute to hydrodl2, clone the [hydrodl2 master branch](https://github.com/mhpi/hydrodl2) from GitHub and install in developer mode (similar to 𝛿MG):

  ```bash
  git clone git@github.com:mhpi/hydrodl2.git
  pip install -e ./hydrodl2

  # or

  uv pip install -e ./hydrodl2
  ```

  Note, developer mode will ensure hydrodl2 won't need to be reinstalled whenever you make changes.

- Geo Plotting

  - For geographical plotting features (e.g., mapping model metrics spatially) available in `./examples/`, install dependencies with

  ```bash
  pip install ./generic_deltamodel[geo]

  # or

  uv pip install ./generic_deltamodel[geo]
  ```

- Development

  - For developing with and/or making contributions to 𝛿MG some linting and test packages can be installed with

  ```bash
  pip install ./generic_deltamodel[dev]

  # or

  uv pip install ./generic_deltamodel[dev]
  ```

  If you wish to install with conda, you will need to manually install 'dev' packages (e.g., ruff, pytest) listed in `./generic_deltamodel/pyproject.toml`.

</br>

---

*Please submit an [issue](https://github.com/mhpi/generic_deltamodel/issues) on GitHub to report any questions, concerns, bugs, etc.*
