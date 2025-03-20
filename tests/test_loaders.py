"""Test data loaders in dMG/core/data/loaders/."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


from dMG.core.data.loaders.base import BaseLoader
from tests import get_available_classes

# Path to module directory
PATH = Path(__file__).parent.parent / 'src' / 'dMG' / 'core' / 'data' / 'loaders'
PKG_PATH = 'dMG.core.data.loaders'


loaders = get_available_classes(PATH, PKG_PATH, BaseLoader)
