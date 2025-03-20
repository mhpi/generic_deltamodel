"""Test data samplers in dMG/core/data/samplers/."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dMG.core.data.samplers.base import BaseSampler
from tests import get_available_classes

# Path to module directory
PATH = Path(__file__).parent.parent / 'src' / 'dMG' / 'core' / 'data' / 'samplers'
PKG_PATH = 'dMG.core.data.samplers'


loaders = get_available_classes(PATH, PKG_PATH, BaseSampler)
