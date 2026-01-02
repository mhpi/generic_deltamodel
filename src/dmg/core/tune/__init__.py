# src/dmg/core/tune/__init__.py
try:
    from .tune import RayTrainable

    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    RayTrainable = None

__all__ = ['RayTrainable', 'HAS_RAY']
