## $\delta$MG Style Guide

The following is a shortlist of guidelines for the Python coding formalisms used throughout the `$\delta$MG` package,
in addition to `hydroDL2` and `hydro_data`. These are for development reference, and are organized in no
particular order.

We suggest any additions or modifications be made in accordance with this document to minimize code discord.

Recommendations for this list are welcome.

---

1. Configuration files should be handled as a dict with named keys like so:
    - For a config with key name `random_seed`, the key value should be called as `config['random_seed']`.

2. Docstrings should use the NumPy docstring standard (see [here](https://numpydoc.readthedocs.io/en/latest/format.html)
and extended [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)) with form 

                ```python
                def function_name(param1, param2, *args, **kwargs):
                    """Brief summary of function's purpose.

                    Extended description of function.

                    Parameters
                    ----------
                    param1 : data_type
                        Description of the first parameter. Include details about its purpose,
                        constraints, or special conditions.
                    param2 : data_type
                        Description of the second parameter. Specify expected input type and
                        any relevant details.
                    *args : tuple, optional
                        Description of additional positional arguments.
                    **kwargs : dict, optional
                        Description of additional keyword arguments.
                    
                    Returns
                    -------
                    return_type
                        Description of the return value(s). For instance, specify the type and
                        what the return value represents in context of the function.

                    Raises
                    ------
                    ExceptionType
                        Description of conditions under which the exception is raised.

                    Notes
                    -----
                    Additional notes about the function, such as implementation details or
                    mathematical formulas.

                    Examples
                    --------
                    >>> result = function_name(param1_value, param2_value)
                    >>> print(result)
                    """
                    pass
                ```
    
    3. Type-hinting should be used whenever writing definitions or classes: e.g. 
        - `def initialize_config(cfg: DictConfig) -> Dict[str, Any]`
    