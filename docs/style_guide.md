# *dMG* Style Guide

This is a shortlist of guidelines and code formalisms used in this package. These are for development reference, and organized in no
particular order. Recommendations are welcome.

---

1. Configuration files should be handled as a dict with named keys like so:
    - For a config with key name `random_seed`, the key value should be called as `config['random_seed']`.

2. Docstrings should use the NumPy docstring standard (see [here](https://numpydoc.readthedocs.io/en/latest/format.html)
and extended [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)) with form

    ```python
    def function_name(param1: type, param2: type, *args: Optional[tuple], **kwargs: Optional[dict]) -> type:
        """Brief summary of function's purpose.

        Extended description of function.

        Parameters
        ----------
        param1
            Description of the first parameter. Include details about its purpose,
            constraints, or special conditions.
        param2
            Description of the second parameter. Specify expected input type and
            any relevant details.
        *args
            Description of additional positional arguments.
        **kwargs
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

3. Type hints should be used whenever writing definitions or classes: e.g.
    - `def initialize_config(cfg: DictConfig) -> Dict[str, Any]`
