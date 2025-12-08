import re


def convert_nested(obj):
    """Recursively traverse nested dicts/lists and convert all values."""

    def convert_value(v):
        """Convert a single value from string to its proper dtype if possible."""
        if isinstance(v, str):
            # Handle special cases
            if v == "None":
                return None
            if v == "True":
                return True
            if v == "False":
                return False

            # Handle numeric forms (integer, float, scientific notation)
            if re.fullmatch(r"[-+]?\d+", v):
                return int(v)
            if re.fullmatch(
                r"[-+]?\d*\.\d+(e[-+]?\d+)?", v, re.IGNORECASE
            ) or re.fullmatch(r"[-+]?\d+e[-+]?\d+", v, re.IGNORECASE):
                try:
                    return float(v)
                except ValueError:
                    return v

            # Keep as string otherwise
            return v
        return v

    if isinstance(obj, dict):
        return {k: convert_nested(convert_value(v)) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nested(convert_value(i)) for i in obj]
    else:
        return convert_value(obj)
