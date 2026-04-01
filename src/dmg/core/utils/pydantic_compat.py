"""
Hopefully temporary compatibility support for Pydantic v1/v2.
Necessary for operational environs (NOAA DMOD, T-Route) still on Pydantic v1.
"""

from pydantic import VERSION

PYDANTIC_V2 = VERSION.startswith("2.")


def v1_mock_self(cls, values):
    """Create a mock self object for Pydantic v1 validators."""

    class Mock:
        pass

    obj = Mock()
    obj.__dict__.update(values)
    # Add defaults for missing optional fields
    for k, v in cls.__fields__.items():
        if k not in values:
            setattr(obj, k, v.default)
    return obj
