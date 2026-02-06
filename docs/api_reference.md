# API Reference

This document catalogs the public API of the `dmg` package. All items listed here are importable from their respective modules.

---

## Delta Models

Core differentiable model classes that couple neural networks with physics-based models.

| Class | Module | Description |
|-------|--------|-------------|
| `DplModel` | `dmg.models.delta_models` | Differentiable Parameter Learning model. An `nn.Module` that composes a neural network with a physics model — the NN learns parameters that are passed to the physics model. |
| `MtsDplModel` | `dmg.models.delta_models` | Multi-timescale variant of `DplModel` for models like δHBV 2.0. |

**DplModel**

```python
from dmg.models.delta_models import DplModel

model = DplModel(
    phy_model=phy_model,       # Physics model (torch.nn.Module)
    nn_model=nn_model,         # Neural network (torch.nn.Module)
    phy_model_name='Hbv',      # Or specify name to auto-initialize
    config=config,             # Config dict (required if using name-based init)
    device='cpu',              # 'cpu' or 'cuda'
)

output = model(data_dict)  # Returns dict of physics model outputs
```

</br>

## Model Handlers

High-level managers that handle model initialization, multimodel ensembles, loss computation, and checkpointing.

| Class | Module | Description |
|-------|--------|-------------|
| `ModelHandler` | `dmg.models.model_handler` | Main handler for single and multimodel differentiable model experiments. |
| `MtsModelHandler` | `dmg.models.mts_model_handler` | Handler for multi-timescale δHBV 2.0 models. |

**ModelHandler**

```python
from dmg.models.model_handler import ModelHandler

handler = ModelHandler(config, device='cuda', verbose=True)
handler.load_model(epoch=0)      # Load from checkpoint or create new
handler.train(mode=True)         # Set training mode
output = handler.forward(data)   # Forward pass (returns dict)
loss = handler.calc_loss(data)   # Compute loss
handler.save_model(epoch=10)     # Save checkpoint
```

Key methods:
- `load_model(epoch)` — Load model weights from checkpoint or initialize new.
- `forward(dataset_dict, eval=False)` — Run forward pass.
- `calc_loss(dataset_dict, loss_func=None)` — Calculate loss.
- `train(mode=True)` — Set train/eval mode on all internal models.
- `get_parameters()` — Return all model parameters for optimizer.
- `save_model(epoch)` / `save_states()` / `load_states()` — Serialization.

</br>

## Neural Networks

Available neural network architectures. Specify by class name in the `model.nn.name` config field.

| Class | Module | Description |
|-------|--------|-------------|
| `LstmModel` | `dmg.models.neural_networks` | LSTM with linear input/output layers. Supports CPU and GPU. |
| `CudnnLstmModel` | `dmg.models.neural_networks` | HydroDL LSTM using cuDNN backend. **GPU only.** |
| `AnnModel` | `dmg.models.neural_networks` | Feed-forward ANN with 6 hidden layers and dropout. |
| `AnnCloseModel` | `dmg.models.neural_networks` | ANN with close-observation integration. |
| `MlpModel` | `dmg.models.neural_networks` | Multi-layer perceptron with 3 hidden layers. |
| `LstmMlpModel` | `dmg.models.neural_networks` | Combined LSTM + MLP for multi-scale parameter learning. |
| `StackLstmMlpModel` | `dmg.models.neural_networks` | Stacked LSTM-MLP for multi-timescale applications. |

**LstmModel**

```python
from dmg.models.neural_networks import LstmModel

lstm = LstmModel(
    nx=10,             # Input feature dimension
    ny=5,              # Output dimension
    hidden_size=256,   # Hidden state size
    dr=0.5,            # Dropout rate
    cache_states=False # Whether to cache hidden/cell states
)
```

</br>>

## Loss Functions (Criterion)

Loss functions for training. Specify by class name in the `train.loss_function.name` config field.

| Class | Module | Description |
|-------|--------|-------------|
| `MSELoss` | `dmg.models.criterion` | Mean squared error. |
| `RmseLoss` | `dmg.models.criterion` | Root mean squared error. |
| `RmseCombLoss` | `dmg.models.criterion` | Combined RMSE + log-sqrt RMSE (alpha-weighted). |
| `NseBatchLoss` | `dmg.models.criterion` | Nash-Sutcliffe Efficiency (NSE) batch loss. |
| `NseSqrtBatchLoss` | `dmg.models.criterion` | Square-root NSE batch loss. |
| `KgeBatchLoss` | `dmg.models.criterion` | Kling-Gupta Efficiency (KGE) loss. |
| `KgeNormBatchLoss` | `dmg.models.criterion` | Normalized KGE (N-KGE) loss. |
| `RangeBoundLoss` | `dmg.models.criterion` | Penalty loss for multimodel weight bounds. |

All loss functions inherit from `BaseCriterion` and implement:

```python
loss_fn = MSELoss(config, device='cpu')
loss = loss_fn(y_pred, y_obs)  # Returns scalar tensor
```

</br>>>

## Data Loaders

Data loaders preprocess and load full datasets. Specify by class name in the `data_loader` config field.

| Class | Module | Description |
|-------|--------|-------------|
| `HydroLoader` | `dmg.core.data.loaders` | CAMELS hydrological dataset loader. |
| `MsHydroLoader` | `dmg.core.data.loaders` | Multi-scale hydrological dataset loader. |
| `MtsHydroLoader` | `dmg.core.data.loaders` | Multi-timescale hydrological dataset loader. |

**HydroLoader**

```python
from dmg.core.data.loaders import HydroLoader

loader = HydroLoader(config, test_split=False)
train_data = loader.train_dataset   # dict of tensors
eval_data = loader.eval_dataset     # dict of tensors
full_data = loader.dataset          # dict of tensors (inference)
```

</br>

## Data Samplers

Samplers draw minibatches from loaded datasets during training. Specify by class name in the `data_sampler` config field.

| Class | Module | Description |
|-------|--------|-------------|
| `HydroSampler` | `dmg.core.data.samplers` | Hydrological data sampler for minibatch selection. |
| `MsHydroSampler` | `dmg.core.data.samplers` | Multi-scale hydrological sampler. |

</br>

## Trainers

Trainers orchestrate training loops, evaluation, and inference. Specify by class name in the `trainer` config field.

| Class | Module | Description |
|-------|--------|-------------|
| `Trainer` | `dmg.trainers` | Main trainer for NN and differentiable models. |
| `MsTrainer` | `dmg.trainers` | Trainer for multi-scale models. |

**Trainer**

```python
from dmg.trainers import Trainer

trainer = Trainer(
    config=config,
    model=model_handler,
    train_dataset=loader.train_dataset,
    eval_dataset=loader.eval_dataset,
)
trainer.train()       # Run training loop
trainer.evaluate()    # Run evaluation
trainer.inference()   # Run inference/simulation
```

</br>

## Factory Functions

Utility functions for dynamically initializing components from config.

| Function | Module | Description |
|----------|--------|-------------|
| `load_nn_model(config, phy_model, ...)` | `dmg.core.utils` | Initialize a neural network from config settings. |
| `load_criterion(y_obs, config, ...)` | `dmg.core.utils` | Initialize a loss function from config settings. |
| `import_phy_model(model, ...)` | `dmg.core.utils` | Dynamically import a physics model class from hydrodl2 or local modules. |
| `import_data_loader(name)` | `dmg.core.utils` | Dynamically import a data loader class by name. |
| `import_data_sampler(name)` | `dmg.core.utils` | Dynamically import a data sampler class by name. |
| `import_trainer(name)` | `dmg.core.utils` | Dynamically import a trainer class by name. |

```python
from dmg.core.utils import load_nn_model, load_criterion, import_phy_model

# Initialize NN from config
nn = load_nn_model(config['model'], phy_model=phy_model)

# Initialize loss function
loss_fn = load_criterion(y_obs, config, device='cuda')

# Import physics model class
PhyModel = import_phy_model('Hbv')
phy_model = PhyModel(config['model']['phy'])
```

</br>

## General Utilities

| Function | Module | Description |
|----------|--------|-------------|
| `initialize_config(config)` | `dmg.core.utils` | Parse Hydra/OmegaConf config into a Python dict. |
| `print_config(config)` | `dmg.core.utils` | Pretty-print configuration settings. |
| `set_randomseed(seed)` | `dmg.core.utils` | Set random seeds for NumPy and PyTorch reproducibility. |
| `save_model(path, model, name, epoch)` | `dmg.core.utils` | Save model state dict to disk. |
| `save_outputs(config, predictions, y_obs)` | `dmg.core.utils` | Save model predictions to disk. |

</br>

## Metrics

| Class | Module | Description |
|-------|--------|-------------|
| `Metrics` | `dmg.core.calc.metrics` | Pydantic model that computes evaluation metrics from prediction/target arrays. |

```python
from dmg.core.calc.metrics import Metrics

m = Metrics(pred=predictions, target=observations)
print(m.nse, m.kge, m.rmse, m.corr)
```

Computed metrics include: `nse`, `kge`, `rmse`, `r2`, `corr`, `corr_spearman`, `mae`, `pbias`, `flv`, `fhv`, and more.
