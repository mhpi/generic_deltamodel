from typing import Any

from ray import tune  # also requires manual install: pyarrow, optuna
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from dMG.core.tune.tune import RayTrainable


def run_tuning(config: dict[str, Any]):
    """Generic implementation of RayTune for model/hyperparameter.
    
    See details here: https://docs.ray.io/en/latest/tune/index.html

    Parameters
    ----------
    config
        Configuration dictionary.
    """
    search_space = {
        "trainer.lr": tune.loguniform(1e-5, 1e-2),
        "trainer.batch_size": tune.choice([16, 32, 64]),
    }

    tuner = tune.Tuner(
        RayTrainable,
        param_space=search_space,
        tune_config= tune.TuneConfig(
            metric=config['tune']['metric'],
            mode=config['tune']['mode'],
            scheduler= ASHAScheduler(),
            search_alg=OptunaSearch(),
            num_samples=config['tune']['num_samples'],
        ),
        run_config=tune.RunConfig(name="tune_dMG"),
    )
    tuner.fit()
