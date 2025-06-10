"""Test data samplers in dmg/core/data/samplers/."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dmg.core.data.samplers.base import BaseSampler
from tests import get_available_classes

# Path to module directory
PATH = Path(__file__).parent.parent / 'src' / 'dmg' / 'core' / 'data' / 'samplers'
PKG_PATH = 'dmg.core.data.samplers'


loaders = get_available_classes(PATH, PKG_PATH, BaseSampler)
