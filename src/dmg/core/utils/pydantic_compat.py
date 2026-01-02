"""
Hopefully temporary compatibility layer to enable Pydantic v1/v2 support.
Necessary for operational environs (NOAA DMOD, T-Route) still on Pydantic v1.
"""

from pydantic import (
    VERSION,
    root_validator as v1_root_validator,
    validator as v1_validator,
)


PYDANTIC_V2 = VERSION.startswith("2.")


# --- Validator Helpers ---


def universal_model_validator(func):
    """Replacement for @model_validator(mode='after')."""
    if PYDANTIC_V2:
        from pydantic import model_validator

        return model_validator(mode='after')(func)

    from functools import wraps

    # We MUST use (cls, values) here to satisfy Pydantic v1's internal check
    @v1_root_validator(pre=False)
    @wraps(func)
    def wrapper(cls, values):
        # Create the 'MockSelf' object so the user's logic can still use 'self'
        class MockSelf:
            def __init__(self, d):
                self.__dict__.update(d)

        # Call the user's function passing our fake self
        # This allows 'def validate_dates(self):' to work
        func(MockSelf(values))
        return values

    return wrapper


def universal_field_validator(*fields):
    """Replacement for @field_validator('field1', 'field2')."""
    if PYDANTIC_V2:
        from pydantic import field_validator

        return field_validator(*fields)

    # V1 @validator uses 'pre=False' by default which matches V2 default
    return v1_validator(*fields)


# --- Type Helpers ---

if PYDANTIC_V2:
    from pydantic import ConfigDict
else:
    # For v1, return empty dict because we use 'class Config' anyway.
    def ConfigDict(**kwargs):
        """Stand-in for Pydantic v2 ConfigDict to prevent NameErrors in v1."""
        return kwargs
