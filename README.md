# 𝛿MG: The Generic, Scalable Differentiable Modeling Framework on PyTorch

[![image](https://img.shields.io/github/license/saltstack/salt)](https://github.com/mhpi/generic_deltaModel/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14868671.svg)](https://doi.org/10.5281/zenodo.14868671)



A domain-agnostic, PyTorch-based framework for developing trainable [differentiable models](https://www.nature.com/articles/s43017-023-00450-9) that merge neural networks
with process-based equations. "Differentiable" means that gradient calculations can be achieved efficiently at a large
scale throughout the model, so process-based equations can be trained together with NNs on big data, on GPU. 
Following as a generalization of `HydroDL`, 𝛿MG (`generic_deltaModel`) aims to expand differentiable modeling and
learning capabilities to a wide variety of domains where prior equations can bring in benefits. 

𝛿MG is not a particular model. Rather, it is a generic framework that supports many models across various domains (some from HydroDL2.0) in a uniform way, while integrating lots of ecosystem tools. Although the packages contain some basic examples for the learner's convenience, the deployment models are supposed to exist in separate repositories and couple to the 𝛿MG framework. 𝛿MG has been generalized and formalized after years of experience in working with various differentiable models across domains.
Most of the differentiable modeling efforts in our research group will be using 𝛿MG. 𝛿MG can be configured to run through a configuration file and it should be easy and clear for beginners to learn. We even include a Graphical User Interface that allows easy job customization. The framework will closely synergize with advanced deep learning tools like foundation models and the scale advantage of PyTorch. According to our peer-reviewed, published benchmarks, well-tuned differentiable models can match deep networks in performance while extrapolating better in data-scarce regions or extreme scenarios and outputting untrained variables with causal, physical interpretability.

While differentiable models are powerful and have many desirable characteristics, they come with a larger decision space than purely data-driven neural networks since physical processes are involved, and can thus feel "trickier" to work with. Hence, we recommend first reproducing our results and then systematically making changes, one at a time. Furthermore, another recommendation is to focus on the multifaceted outputs, diverse causal analyses, and predictions of untrained variables permitted by differentiable models, rather than purely trying to outperform other models' metrics. 

This package is maintained by the [MHPI group](http://water.engr.psu.edu/shen/) advised by Dr. Chaopeng Shen. If this work is of use to you, please cite the following paper for now (we will have more dedicated citations later):
- Shen, C., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. Nature Reviews Earth & Environment, 4(8), 552–567. https://doi.org/10.1038/s43017-023-00450-9

<br>


## Ecosystem Integration
𝛿MG seamlessly integrates with:

- **HydroDL2.0 ([`hydroDL2`](https://github.com/mhpi/hydroDL2))**: Home to MHPI's suite of process-based hydrology models and differentiable model augmentations (think variational data assimilation, model coupling, and other tools designed for hydrology).
- **HydroData ([`hydro_data_dev`](https://github.com/mhpi/hydro_data_dev))**: Data extraction, processing, and management tools optimized for geospatial datasets. (In development)
- **Config GUI ([`GUI-Config-builder`](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/config_builder_gui/Config+Builder+GUI.zip))([Source](https://github.com/mhpi/GUI-Config-builder))**: An intuitive, user-friendly tool designed to simplify the creation and editing of configuration files for model setup and development.
- **Differentiable Ecosystem modeling ([`diffEcosys (dev version only)`](https://github.com/hydroPKDN/diffEcosys/))**: A physics-informed machine learning system for ecosystem modeling, demonstrated using the photosynthesis process representation within the Functionally Assembled Terrestrial Ecosystem Simulator (FATES) model. This model is coupled to neural networks that learn parameters from observations of photosynthesis rates.
- **Concurrent development activities**: We are working on these efforts connected to 𝛿MG: (i) numerical PDE solvers on Torch; (ii) [adjoint](https://doi.org/10.5194/hess-28-3051-2024) sensitivity; (iii) extremely efficient and highly accurate surrogate models; (iv) data assimilation; (v) downscaled and bias corrected climate data; (vi) mysteriously powerful neural networks, and more ...

<br>


## Key Features
- **Hybrid Modeling**: Combines neural networks with physical process equations for enhanced interpretability and generalizability. For example, instead of manually tuning (calibrating) model parameters, use neural networks to directly learn robust and interpretable parameters ([Tsai et al., 2021](https://doi.org/10.1038/s41467-021-26107-z)).

- **PyTorch Integration**: Easily scales with PyTorch, enabling efficient training and compatibility with modern deep learning tools, trained foundation models, and differentiable numerical solvers.

- **Domain-agnostic and Flexible**: Extends differentiable modeling to any field where physics-guided learning can add value, with modularity to meet the diversity of needs along the way.

- **Benchmarking**: All in one place. 𝛿MG + hydroDL2 will enable rapid deployment and replication of key published MHPI results.

- **NextGen-ready**: 𝛿MG is designed to be [CSDMS BMI](https://csdms.colorado.edu/wiki/BMI)-compliant, and our differentiable hydrology models in hydroDL2 come with a prebuilt BMI allowing seamless compatibility with [NOAA-OWP](https://water.noaa.gov/about/owp)'s [NextGen National Water Modelling Framework](https://github.com/NOAA-OWP/ngen). Incidentally, this capability also enables 𝛿MG to be easily interfaced with other applications.

<br>


## Use Cases

### Hydrologic modeling
This package includes the lumped differentiable rainfall-runoff model [𝛿HBV1.0](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2022WR032404), improved [𝛿HBV1.1p](https://essopenarchive.org/doi/full/10.22541/essoar.172304428.82707157), 𝛿PRMS, and 𝛿SAC-SMA. 
This package powers the global- and  [national-scale water model](https://doi.org/10.22541/essoar.172736277.74497104/v1) that provide high-quality seamless hydrologic [simulations](https://mhpi.github.io/datasets/CONUS/) across US and the world. 
It also hosts [global-scale ecosystem](https://doi.org/10.22541/au.173101418.87755465/v1) learning and simulations.
Many other use cases are being developed concurrently.

#### 1. Unseen extreme event testing using 𝛿HBV1.1p
In the unseen extreme event spatial test, we used water years with a 5-year or lower return period peak flow from 1990/10/01 to 2014/09/30 for training, and held out the water years with greater than a 5-year return period peak flow for testing. The spatial test was conducted using a 5-fold cross-validation approach for basins in the [CAMELS dataset](https://gdex.ucar.edu/dataset/camels.html). This application has been benchmarked against LSTM and demonstrates better extrapolation abilities. More details and results can be found in [Song, Sawadekar, et al. (2024)](https://essopenarchive.org/doi/full/10.22541/essoar.172304428.82707157). 

![Alt text](./docs/images/extreme_temporal.png)

#### 2. National-scale water modeling using 𝛿HBV2.0
This is a national-scale water modeling study on approximately 180,000 river reaches (with a median length of 7 km) across the contiguous US, using a high-resolution, differentiable, national-scale water model. More details and results can be found in [Song, Bindas, et al. (2024)](https://doi.org/10.22541/essoar.172736277.74497104/v1) and 𝛿HBV2.0.

![Alt text](./docs/images/CONUS_dataset.jpg)


### 3. Global-scale photosynthesis modeling

More details and results can be found in [Aboelyazeed et al. (2024)](https://doi.org/10.22541/au.173101418.87755465/v1). 

![Alt text](./docs/images/Vcmax25_learnt_global_combined_2011_2020.png)

<br>


## The Overall Idea
Characterized by the combination of process-based equations and neural networks (NNs), differentiable models train these components together, enabling parameter inputs for the equations to be effectively and efficiently learned at scale by the NNs. Alternatively, one can also have NNs learning the residuals of the physical models. There are many possibilities for how such models are built.

In 𝛿MG, we define a differentiable model with the class *DeltaModel* that can couple one or more NNs with a process-based model (itself potentially a collection of models). This class holds `nn` and `phy_model` objects as internal attributes, and describes how they interface with each other. The *DeltaModel* object can be trained and forwarded just like any other PyTorch model (nn.Module).

We also define *DataLoader* and *DataSampler* classes to handle datasets, a *Trainer* class for running training/testing experiments, and a *ModelHandler* class for multimodel handling, multi-GPU training, data assimilation, and streaming in a uniform and modular way. All model, training, and simulation settings are collected in a configuration file that can be adapted to custom applications. 
According to this schema, we define these core classes from the bottom up:

- **nn**: PyTorch neural networks that can learn and provide either parameters, missing process representations, corrections, or other forms of enhancements to physical models.
- **phy_model**: A physical (process-based) model written in PyTorch (or potentially another interoperable differentiable platform) that takes learnable outputs from the `nn` model(s) and returns a prediction of some target variable(s). This can also be a wrapper holding several physical models.
- **DeltaModel**: Holds (one or multiple) `nn` objects and a `phy_model` object, and describes how they are coupled; connection to ODE packages.
- **ModelHandler**: Manages multimodeling, multi-GPU computation, and data assimilation or streaming. Can contain its own optimizers. Acts as an interface to CSDMS BMI or other interfaces.
- **DataSampler**: Samples data according to data format and training/testing requirements.
- **Trainer**: Manages model training and testing, and connects data to models.
- **DataLoader**: Preprocesses data to be used in training, testing, and simulations.

<br>


## Repository Structure:
    .
    ├── deltaModel/
    │   ├── __main__.py                 # Runs the framework; model experiments
    │   ├── conf/                       # Configuration repository
    │   │   ├── config.py
    │   │   ├── config.yaml             # Main configuration file
    │   │   ├── hydra/                  
    │   │   └── observations/           # Data configuration files
    │   ├── core/                       
    │   │   ├── calc/                   # Calculation utilities
    │   │   ├── data/                   # Data loaders and samplers
    │   │   └── utils/                  # Helper functions
    │   ├── models/                     
    │   │   ├── differentiable_model.py # Differentiable model (dPL modality)
    │   │   ├── model_handler.py        # High-level model manager
    │   │   ├── loss_functions/         # Custom loss functions
    │   │   └── neural_networks/        # Neural network architectures
    │   └── trainers/                   # Training routines
    ├── docs/                           
    ├── envs/                           # Environment configuration files
    └── example/                        # Example and tutorial scripts

<br>


## Quick Start: Building a Differentiable HBV (𝛿HBV) Model
Here’s an example of how you can build a differentiable model, coupling a physical model with a neural network to intelligently learn parameters. In this instance, we use an
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) neural network to learn parameters for the [HBV](https://en.wikipedia.org/wiki/HBV_hydrology_model) physics-based hydrological model.
```python
from example import load_config 
from hydroDL2.models.hbv.hbv import HBV as hbv
from deltaModel.models.neural_networks import init_nn_model
from deltaModel.models.differentiable_model import DeltaModel
from deltaModel.core.data.data_loaders.hydro_loader import HydroDataLoader
from deltaModel.core.data.data_samplers.hydro_sampler import take_sample


CONFIG_PATH = '../example/conf/config_dhbv1_1p.yaml'


# 1. Load configuration dictionary of model parameters and options.
config = load_config(CONFIG_PATH)

# 2. Set up a dataset dictionary of NN and physics model inputs.
dataset = HydroDataLoader(config, test_split=True).eval_dataset
dataset_sample = take_sample(config, dataset, days=730, basins=100)

# 3. Initialize physical model and NN.
phy_model = hbv(config['dpl_model']['phy_model'])
nn = init_nn_model(phy_model, config['dpl_model'])

# 4. Create the differentiable model dHBV: a torch.nn.Module that describes how 
# the NN is linked to the physical model HBV.
dpl_model = DeltaModel(phy_model=phy_model, nn_model=nn)


## From here, forward (run) or train dpl_model just as any torch.nn.Module model.

# 5. For example, to forward:
output = dpl_model.forward(dataset_sample)
```

In the above, we illustrate a critical behavior of the differentiable model object `DeltaModel`, which is composed of the physical model, `phy_model=hbv`, and a neural network, `nn`. 

When we forward `DeltaModel`, we feed scaled inputs (stored within the dataset dictionary) to `nn`, which returns a set of predicted parameters. These parameters then pass with the dataset dictionary to forward `phy_model` and output final model predictions. Internally, these steps are represented within the DeltaModel forward method as

```python
# Parameterization
parameters = self.nn_model(dataset_sample['xc_nn_norm'])        

# Physics model forward
predictions = self.phy_model(
    dataset_sample,
    parameters,
)
```

See [examples](https://github.com/mhpi/generic_deltaModel/blob/master/example/differentiable_hydrology/dhbv_tutorial.ipynb) in the `generic_deltaModel` repository for this and other tutorials.

Note, the [Config GUI](https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/config_builder_gui/Config+Builder+GUI.zip) can be used to create/edit additional config files for use with these examples (see [usage instructions](https://github.com/mhpi/GUI-Config-builder/blob/main/README.md))

<br>

### Contributing:
We welcome contributions! Please submit changes via a fork and pull requests. For more details, refer to docs/CONTRIBUTING.md.

Explore the [roadmap](https://github.com/orgs/mhpi/projects/4) for planned features and improvements. Differentiable numerical packages like torchode and torchdiffeq will be coming in the near future!
