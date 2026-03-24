"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Deserialises criticality function definitions from JSON into instantiated
    objects from functions.py. Each JSON entry specifies a function type
    (Gaussian, StaticHypercube, Ramp, Noise, Sinus) and its parameters;
    active_dims optionally restricts evaluation to a subset of space dimensions.

    - key role: Bridges JSON-based space specifications and the Space constructor.
    - dependency: functions.py (Gaussian, StaticHypercube, Ramp, Noise, Sinus)
    - output: List of function instances passed to Space(functions=...)

Usage:
    from functionloader import load_functions_from_json
    fns = load_functions_from_json("Spaces/examples/example2dwnoise.json")
    # or pass a pre-loaded dict:
    fns = load_functions_from_json(config_dict)
"""

import json
import functions
from pathlib import Path
from typing import Union, List


def load_functions_from_json(source: Union[str, Path, dict]) -> List:
    """
    Parse a criticality-space JSON specification and return instantiated
    function objects.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a JSON file or a pre-loaded configuration dict.
        Expected top-level key: ``"functions"`` — a list of function specs,
        each with at minimum a ``"type"`` field.
        Supported types: Gaussian, StaticHypercube, Ramp, Noise, Sinus.

    Returns
    -------
    list
        Ordered list of function instances (from functions.py) ready to be
        passed to ``Space(functions=...)``.

    Raises
    ------
    FileNotFoundError
        If a file path is given but does not exist.
    TypeError
        If source is neither a path nor a dict.
    ValueError
        If a function spec contains inconsistent dimensions or an unknown type.
    """
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(
                f"The specified configuration file does not exist: {file_path}"
            )
        with file_path.open("r") as file:
            config = json.load(file)
    elif isinstance(source, dict):
        config = source
    else:
        raise TypeError("source must be a file path (str or Path) or a dictionary.")

    function_instances = []

    for func in config["functions"]:
        kwargs = {}
        func_type = func["type"]
        active_dims = func.get("active_dims")  # optional

        if func_type == "Gaussian":
            kwargs["mean"] = func["mean"]
            kwargs["std_dev"] = func["std_dev"]
            if "max_amplitude" in func:
                kwargs["max_amplitude"] = func["max_amplitude"]
            if "range_factor" in func:
                kwargs["range_factor"] = func["range_factor"]
            kwargs["active_dims"] = (
                active_dims
                if active_dims is not None
                else list(range(len(func["mean"])))
            )
            # Consistency check
            if len(kwargs["active_dims"]) != len(kwargs["mean"]):
                raise ValueError(
                    f"Inconsistent dimensions in Gaussian: len(active_dims) != len(mean)"
                )
            function_instances.append(functions.Gaussian(**kwargs))

        elif func_type == "StaticHypercube":
            dimensions = [tuple(d) for d in func["dimensions"]]
            kwargs["dimensions"] = dimensions
            if "static_value" in func:
                kwargs["static_value"] = func["static_value"]
            function_instances.append(functions.StaticHypercube(**kwargs))

        elif func_type == "Ramp":
            kwargs["coeff"] = func["coeff"]
            kwargs["min_value"] = func["min_value"]
            kwargs["max_value"] = func["max_value"]
            kwargs["bounds"] = func["bounds"]
            kwargs["active_dims"] = (
                active_dims
                if active_dims is not None
                else list(range(len(func["coeff"])))
            )
            if len(kwargs["active_dims"]) != len(kwargs["coeff"]):
                raise ValueError(
                    f"Inconsistent dimensions in Ramp: len(active_dims) != len(coeff)"
                )
            function_instances.append(functions.Ramp(**kwargs))

        elif func_type == "Noise":
            kwargs["amplitude"] = func["amplitude"]
            if "std_dev" in func:
                kwargs["std_dev"] = func["std_dev"]
            if "seed" in func:
                if func["seed"] == "None":
                    kwargs["seed"] = None
                else:
                    kwargs["seed"] = func["seed"]
            function_instances.append(functions.Noise(**kwargs))

        elif func_type == "Sinus":
            kwargs["amplitudes"] = func["amplitudes"]
            kwargs["frequencies"] = func["frequencies"]
            kwargs["weight_vectors"] = func["weight_vectors"]
            if "phases" in func:
                kwargs["phases"] = func["phases"]
            if "offset" in func:
                kwargs["offset"] = func["offset"]
            if "bounds" in func:
                kwargs["bounds"] = [tuple(b) for b in func["bounds"]]
            if "minimum_value" in func:
                kwargs["minimum_value"] = func["minimum_value"]
            kwargs["active_dims"] = (
                active_dims
                if active_dims is not None
                else list(range(len(func["weight_vectors"][0])))
            )
            # Check that each weight vector is consistent with active_dims
            for i, w in enumerate(kwargs["weight_vectors"]):
                if len(w) != len(kwargs["active_dims"]):
                    raise ValueError(
                        f"Inconsistent dimensions in Sinus wave {i}: weight vector length does not match active_dims"
                    )
            function_instances.append(functions.Sinus(**kwargs))

        else:
            raise ValueError(f"Unsupported function type: {func_type}")

    return function_instances
