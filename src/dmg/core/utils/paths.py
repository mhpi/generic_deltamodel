# src/dmg/core/utils/paths.py
"""Path resolvers for Hydra experiment management.

Provides OmegaConf resolvers that enable named experiments via the
`exp_name` config field. When exp_name is set, runs are placed in
`output/{name}/experiments/{exp_name}/`; otherwise they fall back to
timestamped `output/{name}/runs/run-{timestamp}/` directories.

Directory Structure:
    output/{model_name}/
    ├── experiments/{exp_name}/      # Named experiments (preferred)
    └── runs/run-{timestamp}/        # Timestamped runs (default)

@leoglonz
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

log = logging.getLogger('dmg.paths')

_PROJECT_ROOT: Optional[str] = None
_resolvers_registered = False


def get_project_root() -> str:
    """Get the project root directory.

    Detects the project root by looking for marker files (.git, pyproject.toml)
    or by extracting from Hydra run directories.
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    cwd = os.getcwd()

    # If we're in a Hydra run directory, extract project root
    # Pattern: {project}/output/{model}/runs/run-{timestamp} or experiments/{name}
    if '/output/' in cwd:
        parts = cwd.split('/output/')
        if len(parts) >= 2:
            _PROJECT_ROOT = parts[0]
            return _PROJECT_ROOT

    # Walk up looking for project markers
    current = Path(cwd)
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists() or (parent / 'pyproject.toml').exists():
            _PROJECT_ROOT = str(parent)
            return _PROJECT_ROOT

    _PROJECT_ROOT = cwd
    return _PROJECT_ROOT


def get_output_root() -> str:
    """Get the output root directory ({project_root}/output)."""
    return os.path.join(get_project_root(), 'output')


def _exp_dir_resolver(exp_name: str | None) -> str:
    """Resolve experiment directory based on exp_name.

    Returns "experiments/{exp_name}" if exp_name is set,
    otherwise "runs/run-{timestamp}".
    """
    if exp_name and exp_name != 'null':
        return f"experiments/{exp_name}"
    return f"runs/run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def register_resolvers() -> None:
    """Register custom OmegaConf resolvers for path management.

    Resolvers:
        - exp_dir: Resolves experiment directory from exp_name
        - project_root: Returns project root directory
        - output_root: Returns output root directory

    Safe to call multiple times.
    """
    global _resolvers_registered
    if _resolvers_registered:
        return

    if not OmegaConf.has_resolver("exp_dir"):
        OmegaConf.register_new_resolver("exp_dir", _exp_dir_resolver)

    if not OmegaConf.has_resolver("project_root"):
        OmegaConf.register_new_resolver("project_root", get_project_root)

    if not OmegaConf.has_resolver("output_root"):
        OmegaConf.register_new_resolver("output_root", get_output_root)

    _resolvers_registered = True


def check_experiment_exists(exp_name: str | None) -> None:
    """Check if a named experiment already exists and would be overwritten.

    Skipped for timestamped runs (exp_name is None or 'null').

    Raises
    ------
    FileExistsError
        If exp_name is set and the experiment directory already contains data.
    """
    if not exp_name or exp_name == 'null':
        return

    run_dir = os.getcwd()
    hydra_items = {'configs', '.hydra'}

    for f in os.listdir(run_dir):
        if f in hydra_items or f.startswith('.') or f.endswith('.log'):
            continue
        raise FileExistsError(
            f"\n\nExperiment '{exp_name}' already exists @ {run_dir}\n"
            f"Use a different exp_name or delete the existing experiment.\n"
        )


# Register resolvers on module import
register_resolvers()
