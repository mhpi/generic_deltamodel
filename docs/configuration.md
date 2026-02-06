# Configuration in *𝛿MG*

Every model built in 𝛿MG is designed to run on a pair of configuration files to isolate experiment, model, and data settings. These are handled by [Hydra config manager](https://hydra.cc/docs/intro/).

<br>

## Configuration Files

1. **(Master) Model/experiment**: `./conf/<config_name>.yaml`

    - This will govern model training/testing/prediction settings, in addition to differentiable model, neural network, and physical model-specific settings.

    - A minimal required implementation is given in `/config.yaml`; *all settings here are required by the framework*.

2. **Data**: `./conf/observations/<dataset_name>.yaml`

    - This contains settings specific to a dataset (multiple may be present for different datasets) you wish to use for your model, and includes directory paths, temporal ranges, and constituent variables.

    - A basic example is given in `/observations/none.yaml`.

    - These configs use a *name* attribute to link to the main config (Hydra effectively links this data config to the main as a subdictionary). The header of the main config contains this linkage:

      ```yaml
        defaults:
          - _self_
          - hydra: settings
          - observations: <observations_name>
        ```

    - There are **no** requirements for this except that the config have the *name* attribute. All settings here are intended to be minimally exposed within your data loader, so it's up to you what you want to include.

<br>

## Adding Configurations

If you wish to use additional configuration files to store distinguished settings not related to the above:

- Create a new directory for the config type like `./conf/<config_type>/` and place your configs within.

- Add to the header of your main config

  ```yaml
  defaults:
    - _self_
    - hydra: settings
    - observations: <observations_name>
    - <config_type>: <config_file_name>  # <-- Add here
  ```

  where *config_file_name* reflects the `name` attribute of the config file.

<br>

## Initializing Configuration Files in *𝛿MG*

Configuration file management is handled by the Hydra config manager (see above). Essentially, at the start of a model experiment, Hydra will load configs into a single Python dictionary object of all settings that can be accessed throughout the framework.

You can see this demonstrated in the main 𝛿MG run file, `./src/dmg/__main__.py`, at the start of the main function we call the decorator

```python
@hydra.main(
    version_base='1.3',
    config_path='conf/',
    config_name='config',
)
def main(config):
    config = initialize_config(config)
    ...
```

where *config* is the name of the main `config.yaml` file. Hydra builds and passes config as an Omegaconf DictConfig object *config* (see main definition) that we then parse into a Python dictionary with *initialize_config*.

This processing can be done without the decorator, but this is generally the most straightforward way to do it and *needs to be included* in any other scripts used to run your models.

<br>

## Accessing Settings in the Config Dictionary

After your configuration files are initialized as a dictionary:

- Any settings in the main config can be accessed like `config['mode']` or `config['train']['start_time']` for subsettings in the config.yaml (headers like *train* and *delta_model* create subdictionaries).

- Settings in your observations data config or other type (see [adding-configurations](#adding-configurations)) can be accessed as subdictionaries like `config['observations'][<setting_name>]` or `config['config_type'][<setting_name>]`.

<br>

---

<br>

# Configuration Glossary

This section defines all configuration options used in (1) model and (2) observation configuration files in 𝛿MG.

Universal options will be **bold-faced** to distinguish from options used specifically for MHPI hydrology models like δHBV (see [examples](../example/hydrology/)).

The settings are broken down as they appear in the YAML configuration files, with default/valid choices provided where appropriate.

---

<br>

<br>

## 1. Model/Experiment (Master) Configuration

### 1.1 Hydra (YAML Argument-parser)

**defaults**:

- self
- **hydra**: Name of the settings file in the `./hydra/` directory.
- **observations**: Name of observations configuration file (must match `name` key of said file).
- <**config_type**>: If you want to add supplemental configuration files, add name of configuration here just as for observations.

<br>

### 1.2 General

**mode**: [train, test, train_test, simulation] Experiment mode.

- `train`: Train the model.
- `test`: Test the model and report performance metrics.
- `train_test`: Run train and test in series.
- `simulation`: Perform batched model forward without performance metric calculations.

**multimodel_type**: [none, avg, nn_parallel, nn_sequential] Multimodel type.

- `none`: No multimodeling.
- `avg`: Average outputs from multiple models.
- `nn_parallel`: Dynamically weight model outputs using a neural network (NN) trained in parallel with models.
- `nn_sequential`: Dynamically weight model outputs using a NN trained after pre-training models to be ensembled.

**seed**: [111111] Seed to fix deterministic behavior in NumPy and PyTorch.

**logging**: [none, tensorboard, wandb] Experiment logger.

- `none`: No experiment logging.
- `tensorboard`: Use [Tensorboard](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) for logging.
- `wandb`: Use [Weight & Biases](https://wandb.ai/site/) for logging.

**cache_states**: [bool] If true, all physical and/or ML models in the runtime will use cached internal states as an initialization (this could be hidden + cell states for an LSTM, or the storages for a process-based model, for instance). This can be useful in instances where forwarding on a full temporal sequence in a single batch is forbidden, or where sequential forwarding is otherwise required.

**device**: [cpu, cuda] Device to run models on.

- `cpu`: Run on CPU.
- `cuda`: Run on GPU (must be available).

**gpu_id**: [0] If `device = cuda`, the index of the GPU in your system to run models on. Index 0 will always be available.

**data_loader**: Class name of the data loader to use. E.g., *HydroLoader* located in `./src/dmg/core/data/loaders/hydro_loader.py`. Note class name must be Camel-case w/o spaces corresponding to the file name.

**data_sampler**: Class name of the data sampler used in training/inference. E.g., *HydroSampler* located in `./src/dmg/core/data/samplers/hydro_sampler.py`. Follows same convention as data_loader.

**trainer**: Class name of the trainer used in training/inference. E.g., *Trainer* located in `./src/dmg/trainers/trainer.py`. Follows same convention as data_loader.

*model_dir*: Path to a directory containing trained model weights, may also include subdirectory of model outputs. Note this path must end with a forward slash ('/').

*load_state_dir*: Path to a PyTorch save file (`.pt`) containing cached model states for a neural network or differentiable model (e.g., hidden and cell states for an LSTM, buckets for a physical model). If this path is provided, the Model Handler will attempt to load these states into the current model.

<br>

### 1.3 Training

**train**:

- **start_time**: Start date of train period, format as YYYY/MM/DD.

- **end_time**: End date of train period, format as YYYY/MM/DD.

- **target**: Name(s) of target model output(s). Must match key names in model output dictionary, and must be provided as a list of strings.

- **optimizer**:
  - **name**: [Adadelta, Adam] Name of [PyTorch optimizer](https://docs.pytorch.org/docs/stable/optim.html).

- **lr_scheduler**:
  - **name**: [StepLR, ExponentialLR, CosineAnnealingLR] Name of [PyTorch learning rate scheduler](https://docs.pytorch.org/docs/stable/optim.html) for the optimizer.

- **loss_function**:
  - **name**: [KgeBatchLoss, KgeNormBatchLoss, MseLoss, NseBatchLoss, NseSqrtBatchLoss, RmseCombLoss, RmseLoss] Name of loss function for training. See `./src/dmg/models/criterion/` for all available loss functions. You can add custom criterion, but they must follow Class-File convention as illustrated for `data_loader`, etc.

- **batch_size**: Training batch size. Must be less than total number of samples.

- **epochs**: Number of training epochs.

- **start_epoch**: Epoch to resume training from. A checkpoint file for this epoch must be present if not 0.

- **save_epoch**: Save module weights after every epoch that is a multiple of this number.

<br>

### 1.4 Evaluation

**test**:

- **start_time**: Start date of test period, format as YYYY/MM/DD.

- **end_time**: End date of test period, format as YYYY/MM/DD.

- **batch_size**: Testing batch size. Must be less than total number of samples.

- **test_epoch**: Epoch to test model from. Model weights must be available for this epoch.

<br>

### 1.5 Inference

**sim**:

- **start_time**: Start date of simulation period, format as YYYY/MM/DD.

- **end_time**: End date of simulation period, format as YYYY/MM/DD.

- **batch_size**: Simulation batch size. Must be less than total number of samples.

<br>

### 1.6 Differentiable Model

**model**: Settings for the differentiable model.

- **rho**: Prediction window.

- *warm_up*: Number of timesteps to use as warmup for model states.

- *use_log_norm*: Apply log normalization to any input variables listed here.

- **phy**: Physical model settings.
  - **name**: [Hbv, Hbv_adj, Hbv_1_1p, Hbv_2_0, custom_model] name of physical model to use. Can be more than one if a list is passed.
    - `Hbv`: δHBV 1.0
    - `Hbv_adj`: δHBV with adjoint method.
    - `Hbv_1_1p`: δHBV 1.1p
    - `Hbv_2_0`: δHBV 2.0
    - `custom_model`: If you create and add a physical to [phy_model/](../src/dmg/models/phy_models/), this will be the class name. Note it must follow the Class-File convention as illustrated for `data_loader`, etc.

  - *nmul*: Number of parallel parameter sets to use. These will be averaged for single physical model output.

  - *warm_up_states*: [bool] If False, keep model gradients on during warmup. Increases computational time for training. Can slightly improve accuracy.

  - *dy_drop*: Factor to control the chance that some time-dynamic parameters are set to static.

  - *dynamic_params*: Names of parameters in the physical model(s) to learn as time-dynamic.
    - *custom_model*: [parBETA, parBETAET] List of parameters for the physical models named above. These names must be defined within the physical model.

  - *routing*: [bool] Use unit hydrograph river routing on hydrological model discharge predictions if True.

  - *ad_efficient*: [bool] Use efficient automatic differentiation method if True.

  - **nearzero**: [1e-5] Small perturbation value to avoid zero-devision or zeros in calculations.

  - **forcings**: List of time-dynamic input variables for the physical model. Used in data loader.

  - **attributes**: List of static input variables for the physical model. Used in data loader.

- **nn**: Neural network settings.
  - **name**: [CudnnLstmModel, LstmModel, AnnModel, AnnCloseModel, MlpModel, LstmMlpModel] Class name of the NN to use.
    - `CudnnLstmModel`: Custom LSTM built with PyTorch CUDA backend. LSTM of HydroDL. Only supports GPU.
    - `LstmModel`: LSTM equivalent to `CudnnLstmModel` with CPU and GPU support.
    - `AnnModel`: ANN.
    - `AnnCloseModel`: ANN with close observations.
    - `MlpModel`: MLP.
    - `LstmMlpModel`: Combined LSTM-MLP model for multiscale applications.

  - **dropout**: [0.5] Dropout rate applied to model layers for regularization, discourages overfitting.

  - **hidden_size**: [256] Number of hidden units in model layers, determines model complexity.

  - **lr**: [1.0] Initial learning rate used by the optimizer during training.

  - **lr_scheduler**: [None, StepLR, ExponentialLR, CosineAnnealingLR] Name of [Pytorch scheduler](https://docs.pytorch.org/docs/stable/optim.html) to adjust learning rate for the optimizer during training.
    - `None`: No scheduler.
    - `StepLR`: Decays the learning rate of each parameter group by gamma every step_size epochs.
    - `ExponentialLR`: Decays the learning rate of each parameter group by gamma every epoch.
    - `CosineAnnealingLR`: Set the learning rate of each parameter group using a cosine annealing schedule.

  - *lr_scheduler_params*: Settings for the scheduler
    - *step_size*: [10] Period of learning rate decay.

    - *gamma*: [0.1] Multiplicative factor of learning rate decay.

    - *t_max*: Maximum number of iterations.

    - *eta_min*: [0] Minimum learning rate.

  - **forcings**: List of time-dynamic NN input variables. Used in data loader.

  - **attributes**: List of static NN input variables. Used in data loader.

<br>

### 1.7 Multimodel (Optional)

*multimodel*: Settings for NN in NN-weighted multimodel ensembles (see [Section 1.2](#12-general) multimodel type). NN learns to weight a collection of models.

- *model*: [CudnnLstmModel, LstmModel, AnnModel, AnnCloseModel, MlpModel] Class name of the NN to use.
  - `CudnnLstmModel`: Custom LSTM built with PyTorch CUDA backend. LSTM of HydroDL. Only supports GPU.
  - `LstmModel`: LSTM equivalent to `CudnnLstmModel` with CPU and GPU support.
  - `AnnModel`: ANN.
  - `AnnCloseModel`: ANN with close observations.
  - `MlpModel`: MLP.

- *mosaic*: [bool] For gridded domains. If True, only use best (highest weighted) model at each site/gridpoint. If False, do linear combination of all *models* using learned weights at each site/gridpoint.

- *dropout*: [0.5] Dropout rate applied to model layers for regularization, discourages overfitting.

- *hidden_size*: [256] Number of hidden units in model layers, determines model complexity.

- *learning_rate*: [1.0] Initial learning rate used by the optimizer during training.

- *scaling_function*: [sigmoid, softmax] Method to use for scaling learned weights.

- *loss_function*: [KgeBatchLoss, KgeNormBatchLoss, MseLoss, NseBatchLoss, NseSqrtBatchLoss, RmseCombLoss, RmseLoss] Loss function for training. See `./src/dmg/models/criterion/` for all available loss functions. You can add custom criterion, but they must follow Class-File convention as illustrated for `data_loader`, etc.

- *use_rb_loss*: [bool] If True, include range-bound loss regularization. Penalize learned weights when their sum exceeds specific bounds.

- *loss_factor*: [0.10] Scaling factor for range-bound loss.

- *loss_lower_bound*: [0.7] Lower bound for range-bound loss.

- *loss_upper_bound*: [1.3] Upper bound for range-bound loss.

- *forcings*: List of time-dynamic NN input variables. Used in data loader.

- *attributes*: List of static NN input variables. Used in data loader.

---

<br>

<br>

## 2. Observations Configuration

*Note* observation settings are up to you to define and use in your data loader. The only requirement is the `name` parameter for the master configuration. We simply provide definitions here for the settings used for hydrological models presented in 𝛿MG as an example.

**name**: Name of the observations. This is the name to list in `observations: ...` in the [master configuration](#11-hydra-yaml-argument-parser) when this data is to be used.

*data_path*: Full path to the data file.

*gage_info*: Full path to streamflow gage (grid) IDs.

*subset_path*: Full path to a list of gage IDs to subset the dataset.

*subbasin_id_name*: Name of subbasin ID variable for multiscale δHBV 2.0.

*upstream_area_name*: Name of upstream area variable in the dataset to use for input normalization.

*subbasin_area_name*: Name of subbasin area variable in the dataset to use for input normalization.

*elevation_name*: Name of elevation variable in the dataset.

*area_name*: Name of the area variable in the dataset to use for input normalization.

*start_time*: Start date of available date range in data in format YYYY/MM/DD.

*end_time*: End date of available date range in data in format YYYY/MM/DD.

*all_forcings*: List of names of all time-dynamic variables available in the dataset.

*all_attributes*: List of names of all static variables available in the dataset.

<br>

---

*Please submit an [issue](https://github.com/mhpi/generic_deltamodel/issues) on GitHub to report any questions, concerns, bugs, etc.*
