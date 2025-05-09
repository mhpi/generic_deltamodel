# Configuration Files in *dMG*

Every model built in dMG is designed to run on a pair of configuration files to isolate experiment, model, and data settings. These are handled by [Hydra config manager](https://hydra.cc/docs/intro/).

</br>

## Configurations

1. **(Master) Model/experiment**: `./generic_deltamodel/conf/<config_name>.yaml`

    - This will govern model training/testing/prediction settings, in addition to differentiable model, neural network, and physical model-specific settings.

    - A minimal required implementation is given in `/config.yaml`; *all settings here are required by the framework*.

2. **Data**: `./generic_deltamodel/conf/observations/<dataset_name>.yaml`

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

See the [configuration glossary](./configuration_glossary.md) for definitions of standard and hydrology-specific keys used in dMG.

</br>

## Adding Configurations

If you wish to use additional configuration files to store distinguished settings not related to the above:

- Create a new directory for the config type like `./generic_deltamodel/conf/<config_type>/` and place your configs within.

- Add to the header of your main config

  ```yaml
  defaults:
    - _self_
    - hydra: settings
    - observations: <observations_name>
    - <config_type>: <config_file_name>  # <-- Add here
  ```

  where *config_file_name* reflects the `name` attribute of the config file.

</br>

## Initializing Configuration Files in *dMG*

Configuration file managment is handled by the Hydra config manager (see above). Essentially, at the start of a model experiment, Hydra will load configs into a single Python dictionary object of all settings that can be accessed throughout the framework.

You can see this demonstrated in the main dMG run file, `./generic_deltamodel/src/dMG/__main__.py`, at the start of the main function we call the decorator

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

</br>

## Accessing Settings in the Config Dictionary

After your configuration files are initialized as a dictionary:

- Any settings in the main config can be accessed like `config['mode']` or `config['train']['start_time']` for subsettings in the config.yaml (headers like *train* and *delta_model* create subdictionaries).

- Settings in your observations data config or other type (see [here](#adding-configurations)) can be accessed as subdictionaries like `config['observations'][<setting_name>]` or `config['config_type'][<setting_name>]`.

</br>

---

*Please submit an [issue](https://github.com/mhpi/generic_deltamodel/issues) on GitHub to report questions, concerns, bugs, etc.*
