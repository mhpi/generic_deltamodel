# Running 𝛿MG

This doc gives details on how to *run* 𝛿MG once [setup](./setup.md) steps are completed.

For a breakdown of how to *code* with 𝛿MG, we encourage you to check the [hydrology example](../example/hydrology/).

</br>

</br>

## 1. Command Line

Thanks to 𝛿MG's framework-package duality, there are two ways to run 𝛿MG from the command line. Assuming your ENV is active (see [setup](./setup.md)), these are

1. `python -m dmg`

2. `python ./generic_deltamodel/src/dmg/__main__.py`

Both of these options are equivalent and will default 𝛿MG to using the [`./generic_deltamodel/conf/default.yaml`](../conf/default.yaml) master configuration file. To be clear, this is a template and will run unless modified.

To use a different configuration file, we can add an option `--config-name` to either of the above. For example

```bash
python -m dmg --config-name <config_name>
```

where 'config_name' is the file name of your master configuration less the yaml extension. This option comes from Hydra, which is used for parsing the yaml configuration files in 𝛿MG. See [flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/) for more details and available options.

</br>

</br>

## 2. Usage

### 2.1 MHPI Hydrology Models

If you have installed 𝛿MG to use differentiable hydrology models developed by MHPI (δHBV 1.0, δHBV 1.1p, δHBV 2.0, etc.; see [hydrodl2](https://github.com/mhpi/hydrodl2) for current public offerings) you have two options:

1. Use the pre-built [example files](../example/hydrology/) to train or forward these models.

2. You can use the master + observation configuration files from [`./generic_deltamodel/example/conf/`](../example/conf/) to train or forward using the command line arguments in [Section 1](#1-command-line). Simply move these files to the [`./generic_deltamodel/conf/`](../conf/) directory, and then run, for example,

```bash
python -m dmg --config-name config_dhbv.yaml
```

</br>

### 2.2 Custom Model Development

To use 𝛿MG to build and experiment with your own differentiable model, a few things need to happen.

1. Build a master configuration file to encapsulate your model and experiment settings. This must minimally include settings in [`./generic_deltamodel/conf/default.yaml`](../conf/default.yaml). Any additional settings can be added as needed for your custom modules (see below).

2. Build an observations configuration file. This contains all settings/data paths needed to load data in your data loader. You can see the hydrology examples for inspiration.

3. Design or modify modules. If creating a new module, the class name must follow camel-case and the file name must be all lower-case with underscores to split camel-case. (See 𝛿MG source code for examples.)
    - Data loader: Loads full dataset
        - `../dmg/core/data/loaders/`
        - Specify in `data_loader` setting in master configuration.
    - Data sampler: Takes samples of full dataset for minibatching during model training and testing.
        - `../dmg/core/data/samplers/`
        - Specify in `data_sampler` setting in master configuration.
    - Trainer: Handles training and testing experiments, as well as batching data for model forward.
        - `../dmg/trainers/`
        - Specify in `trainer` setting in master configuration.
    - NN: Neural network
        - `../dmg/models/neural_networks/`
        - Specify as `delta_model: nn_model: model` setting in master configuration.
    - Physical model: Physical model written in a differentiable way with PyTorch. (See [hydrodl2/models](https://github.com/mhpi/hydrodl2/tree/master/src/hydrodl2/models/hbv) for examples from hydrology.)
        - `../dmg/models/phy_models/`
        - Specify as `delta_model: phy_model: model` setting in master configuration.

4. Run 𝛿MG from the command line. For example,

    ```bash
    python -m dmg --config-name config_dhbv.yaml
    ```

</br>

---

*Please submit an [issue](https://github.com/mhpi/generic_deltamodel/issues) on GitHub to report any questions, concerns, bugs, etc.*
