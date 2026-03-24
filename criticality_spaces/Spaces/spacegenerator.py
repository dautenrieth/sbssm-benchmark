"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Procedurally generates randomised criticality-space JSON files by composing
    primitive functions (Gaussian, StaticHypercube, Ramp, Sinus) with randomly
    sampled parameters drawn from a JSON configuration file.  Each generated
    space is peak-normalised so that its maximum value equals 1.0.

    - key role: Offline data-generation script that populates the
                ``generated_spaces/`` directory with diverse benchmark spaces
                used for reproducible evaluation of selection methods.
    - dependency: numpy, scipy.stats.qmc, tqdm, space.Space,
                  functionloader.load_functions_from_json
    - output: One JSON file per space written to
              ``generated_spaces/<output_directory>/``.

Usage:
    python spacegenerator.py
    # Reads generation_config_files/config6D.json and writes NUM_FILES
    # normalised space JSON files to the configured output directory.
"""

import os, sys
import json, math
import numpy as np
from pathlib import Path
import random
import itertools
from tqdm import tqdm
from scipy.stats import qmc

from concurrent.futures import ProcessPoolExecutor
base_path = Path(__file__).parent
cs_path = (base_path.parent).resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))
from space import Space
from functionloader import load_functions_from_json

# -------------------------------
# Load Configuration
# -------------------------------
CONFIG_PATH = Path(__file__).parent / 'generation_config_files' / 'config6D.json'

def load_config():
    """
    Load the space-generation configuration from ``CONFIG_PATH``.

    Returns
    -------
    dict
        Parsed JSON configuration dictionary.
    """
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config

config = load_config()

# Configuration Variables
NUM_DIMENSIONS= config["num_dimensions"]
BOUNDS_DIMENSIONS = config["bounds_dimensions"]
NUM_FILES = config["num_files"]
MAX_VALUE = config["max_value"]
SEED = config.get("seed", None)

def generate_active_dims_per_function():
    """
    Sample a random subset of active dimensions for a single primitive function.

    Each dimension ``i`` is included independently with probability
    ``config["active_dims_probability"][str(i)]``.  At least one dimension
    is always returned.

    Returns
    -------
    list of int
        Indices of the dimensions that are active for the primitive function.
    """
    if "0" not in config["active_dims_probability"]:
        print("Warning: Dimension 0 is not explicitly defined in 'active_dims_probability'. This may lead to unexpected omissions.")
    active_dims = [
        i for i in range(NUM_DIMENSIONS)
        if random.random() < config["active_dims_probability"].get(str(i), 0.0)
    ]
    if not active_dims:
        active_dims = [random.randint(0, NUM_DIMENSIONS- 1)]  # Guarantees that one dimension is active
    return active_dims

# -------------------------------
# Function Generators
# -------------------------------
def generate_random_gaussian():
    """
    Generate a random Gaussian primitive function descriptor.

    Parameters are sampled from the ``gaussian_params`` section of the
    configuration: mean and standard deviation per active dimension, and a
    non-negative max amplitude drawn from a normal distribution.

    Returns
    -------
    dict
        Gaussian function descriptor with keys ``type``, ``mean``,
        ``std_dev``, ``max_amplitude``, ``active_dims``.
    """
    active_dims = generate_active_dims_per_function()
    mean = [random.uniform(*config["gaussian_params"]["mean_range"]) for _ in active_dims]
    std_dev = [random.uniform(*config["gaussian_params"]["std_range"]) for _ in active_dims]
    max_amplitude = max(0.01, np.random.normal(
        loc=config["gaussian_params"]["max_amplitude_mean"],
        scale=config["gaussian_params"]["max_amplitude_std_dev"]
    ))

    return {
        "type": "Gaussian",
        "mean": mean,
        "std_dev": std_dev,
        "max_amplitude": max_amplitude,
        "active_dims": active_dims
    }

def generate_random_statichypercube():
    """
    Generate a random StaticHypercube primitive function descriptor.

    For each active dimension a width is drawn from a Beta(2, 5) distribution
    and a compatible lower bound is sampled uniformly, giving a sub-interval
    of the configured bounds range.  The constant value inside the hypercube
    is also Beta(2, 5)-distributed within ``value_range``.

    Returns
    -------
    dict
        StaticHypercube function descriptor with keys ``type``,
        ``dimensions``, ``static_value``, ``active_dims``.
    """
    active_dims = generate_active_dims_per_function()
    cube_dimensions = []

    for _ in active_dims:
        width = np.random.beta(2, 5) * (config["static_hypercube_params"]["bounds_range"][1] - config["static_hypercube_params"]["bounds_range"][0])
        lower_bound = random.uniform(config["static_hypercube_params"]["bounds_range"][0],
                                     config["static_hypercube_params"]["bounds_range"][1] - width)
        upper_bound = lower_bound + width
        cube_dimensions.append([lower_bound, upper_bound])

    static_value = np.random.beta(2, 5) * (config["static_hypercube_params"]["value_range"][1] - config["static_hypercube_params"]["value_range"][0])
    static_value += config["static_hypercube_params"]["value_range"][0]

    return {
        "type": "StaticHypercube",
        "dimensions": cube_dimensions,
        "static_value": static_value,
        "active_dims": active_dims
    }

def generate_random_ramp():
    """
    Generate a random Ramp primitive function descriptor.

    For each active dimension a length is Beta(2, 5)-distributed and a
    coefficient is drawn uniformly from ``coeff_range``.  The minimum and
    maximum output values are sampled from their respective ranges.

    Returns
    -------
    dict
        Ramp function descriptor with keys ``type``, ``coeff``,
        ``min_value``, ``max_value``, ``bounds``, ``active_dims``.
    """
    active_dims = generate_active_dims_per_function()
    bounds = []
    coeff = []

    for _ in active_dims:
        length = np.random.beta(2, 5) * (config["ramp_params"]["bounds_range"][1] - config["ramp_params"]["bounds_range"][0])
        lower_bound = random.uniform(config["ramp_params"]["bounds_range"][0],
                                     config["ramp_params"]["bounds_range"][1] - length)
        upper_bound = lower_bound + length
        bounds.append([lower_bound, upper_bound])
        coeff.append(random.uniform(*config["ramp_params"]["coeff_range"]))

    min_value = random.uniform(*config["ramp_params"]["min_value_range"])
    max_value = random.uniform(min_value + config["ramp_params"]["max_addition_range"][0],
                               min_value + config["ramp_params"]["max_addition_range"][1])

    return {
        "type": "Ramp",
        "coeff": coeff,
        "min_value": min_value,
        "max_value": max_value,
        "bounds": bounds,
        "active_dims": active_dims
    }

def generate_random_sinus():
    """
    Generate a random Sinus primitive function descriptor.

    A random number of sinusoidal waves is sampled; each wave has an
    amplitude (Beta-distributed), frequency, phase (both uniform), and a
    random weight vector over the active dimensions.  An additive offset
    and a minimum-value floor are also included.

    Returns
    -------
    dict
        Sinus function descriptor with keys ``type``, ``amplitudes``,
        ``frequencies``, ``weight_vectors``, ``phases``, ``offset``,
        ``bounds``, ``minimum_value``, ``active_dims``.
    """
    active_dims = generate_active_dims_per_function()
    bounds = []

    for _ in active_dims:
        length = np.random.beta(2, 5) * (config["sinus_params"]["bounds_range"][1] - config["sinus_params"]["bounds_range"][0])
        lower_bound = random.uniform(config["sinus_params"]["bounds_range"][0],
                                     config["sinus_params"]["bounds_range"][1] - length)
        upper_bound = lower_bound + length
        bounds.append([lower_bound, upper_bound])

    num_waves = random.randint(*config["sinus_params"]["num_waves"])
    amplitudes = [
        np.random.beta(2, 5) * (config["sinus_params"]["amplitude_range"][1] - config["sinus_params"]["amplitude_range"][0])
        + config["sinus_params"]["amplitude_range"][0]
        for _ in range(num_waves)
    ]
    frequencies = [random.uniform(*config["sinus_params"]["frequency_range"]) for _ in range(num_waves)]
    weight_vectors = np.random.uniform(0, 1, (num_waves, len(active_dims))).tolist()
    phases = [random.uniform(*config["sinus_params"]["phase_range"]) for _ in range(num_waves)]
    offset = random.uniform(*config["sinus_params"]["offset_range"])

    return {
        "type": "Sinus",
        "amplitudes": amplitudes,
        "frequencies": frequencies,
        "weight_vectors": weight_vectors,
        "phases": phases,
        "offset": offset,
        "bounds": bounds,
        "minimum_value": config["sinus_params"]["minimum_value"],
        "active_dims": active_dims
    }

# -------------------------------
# Sampling & Scaling
# -------------------------------
def batched(iterable, batch_size):
    """
    Yield successive fixed-size chunks from *iterable*.

    Parameters
    ----------
    iterable : iterable
        Source sequence to partition.
    batch_size : int
        Maximum number of elements per chunk.

    Yields
    ------
    list
        Successive slices of *iterable* each containing at most
        *batch_size* elements.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            break
        yield chunk

def sample_scale(functions, samples=None, seed=None):
    """
    Normalise a list of primitive function descriptors so that the space
    maximum equals 1.0.

    Evaluates the composed space over *samples* (or a fresh 1 M-point LHS
    grid when *samples* is ``None``), finds the global maximum, and scales
    all amplitude/value parameters of every function in-place so that the
    peak value becomes 1.0.

    Parameters
    ----------
    functions : list of dict
        Primitive function descriptors (modified in-place and returned).
    samples : list of list or None, optional
        Pre-generated evaluation points.  When ``None`` a 1 M-point LHS
        grid is generated internally (default ``None``).
    seed : int or None, optional
        Random seed forwarded to the internal LHS sampler when *samples*
        is ``None`` (default ``None``).

    Returns
    -------
    list of dict
        The same *functions* list with scaled amplitude parameters.
    """
    dimensions = BOUNDS_DIMENSIONS

    if samples is None: # Only if not samples are given
        n_samples = 1_000_000
        sampler = qmc.LatinHypercube(d=NUM_DIMENSIONS, seed=seed)
        samples = qmc.scale(sampler.random(n=n_samples), [dim[0] for dim in dimensions], [dim[1] for dim in dimensions]).tolist()

    function_instances = load_functions_from_json({"functions": functions})

    space = Space(dimensions=dimensions, functions=function_instances)

    max_value = -np.inf
    for chunk in batched(samples, 50_000):   # Adjust batch size to available RAM
        vals = space.get_values_for_points(chunk, save_points=False)
        mv = np.max(vals)
        if mv > max_value:
            max_value = mv

    if max_value == 0:
        scale_factor = 1.0
    else:
        scale_factor = 1.0 / max_value

    for func in functions:
        if func['type'] == 'Gaussian':
            func['max_amplitude'] *= scale_factor
        elif func['type'] == 'StaticHypercube':
            func['static_value'] *= scale_factor
        elif func['type'] == 'Ramp':
            func['min_value'] *= scale_factor
            func['max_value'] *= scale_factor
        elif func['type'] == 'Sinus':
            func['amplitudes'] = [a * scale_factor for a in func['amplitudes']]
            func['offset'] *= scale_factor
        elif func['type'] == 'Noise':
            func['amplitude'] *= scale_factor

        else:
            print(f"Function type not implemented: {func['type']}")

    return functions

# -------------------------------
# Main File Generation
# -------------------------------
def generate_single_file(i, OUTPUT_DIR, seed, samples):
    """
    Generate and write a single normalised criticality-space JSON file.

    Samples a random mixture of primitive functions (weighted by type),
    normalises the composed space via :func:`sample_scale`, and serialises
    the result to ``<OUTPUT_DIR>/<i>_<NUM_DIMENSIONS>D.json``.

    Parameters
    ----------
    i : int
        File index used in the output filename.
    OUTPUT_DIR : pathlib.Path
        Directory to write the JSON file into.
    seed : int or None
        RNG seed for reproducibility; forwarded to numpy and random.
    samples : list of list
        Pre-generated evaluation points passed to :func:`sample_scale`.
    """
    # One-time initialisation
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_functions = round(NUM_DIMENSIONS * math.log(NUM_DIMENSIONS + 1) + 5)

    generators_with_weights = [
        (generate_random_gaussian, 0.4),
        (generate_random_ramp, 0.2),
        (generate_random_statichypercube, 0.2),
        (generate_random_sinus, 0.2),
    ]

    functions = [gen() for gen in random.choices(
        [g for g, _ in generators_with_weights],
        weights=[w for _, w in generators_with_weights],
        k=num_functions
    )]

    functions = sample_scale(functions.copy(), samples=samples, seed=seed)

    file_name = f"{i}_{NUM_DIMENSIONS}D.json"

    file_path = OUTPUT_DIR / file_name
    with open(file_path, 'w') as f:
        json.dump({"functions": functions}, f, indent=4)
    return

def get_scaled_samples(NUM_DIMENSIONS, MAX_SAMPLES=1_000_000, k=50, p=4.3):
    """
    Compute the number of LHS evaluation points for normalisation.

    Returns the larger of ``k * NUM_DIMENSIONS^p`` and *MAX_SAMPLES* to
    ensure adequate coverage in high-dimensional spaces.

    Parameters
    ----------
    NUM_DIMENSIONS : int
        Number of input dimensions.
    MAX_SAMPLES : int, optional
        Minimum sample count floor (default 1 000 000).
    k : float, optional
        Scaling coefficient (default 50).
    p : float, optional
        Dimension exponent (default 4.3).

    Returns
    -------
    int
        Number of evaluation samples to use for normalisation.
    """
    return max(int(k * NUM_DIMENSIONS ** p), MAX_SAMPLES)


# -------------------------------
# Run
# -------------------------------

def run_generation(i, OUTPUT_DIR, base_seed, samples):
    """
    Worker entry point for parallel space generation.

    Derives a per-file seed from *base_seed* + *i* (or ``None`` when no
    seed is configured) and delegates to :func:`generate_single_file`.

    Parameters
    ----------
    i : int
        File index.
    OUTPUT_DIR : pathlib.Path
        Target output directory.
    base_seed : int or None
        Base random seed; each file receives ``base_seed + i``.
    samples : list of list
        Shared evaluation points passed to :func:`generate_single_file`.
    """
    local_seed = base_seed + i if base_seed is not None else None
    generate_single_file(i, OUTPUT_DIR, local_seed, samples)


def main():
    """
    Entry point: generate ``NUM_FILES`` normalised criticality-space files.

    Draws a shared LHS sample grid, clears the output directory, then
    dispatches :func:`run_generation` calls via a ``ProcessPoolExecutor``
    (``max_workers=1`` for reproducibility) with a ``tqdm`` progress bar.
    """
    sampler = qmc.LatinHypercube(d=NUM_DIMENSIONS, seed=SEED)
    n = get_scaled_samples(NUM_DIMENSIONS, 10_000_000)
    n = 1_000_000
    samples = sampler.random(n=n)
    samples = qmc.scale(samples, [dim[0] for dim in BOUNDS_DIMENSIONS], [dim[1] for dim in BOUNDS_DIMENSIONS])
    samples = samples.tolist()

    OUTPUT_DIR = Path(__file__).parent / 'generated_spaces' / config["output_directory"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file in OUTPUT_DIR.glob("*.json"):
        file.unlink()

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(run_generation, i, OUTPUT_DIR, SEED, samples)
            for i in range(NUM_FILES)
        ]
        for f in tqdm(futures):
            f.result()

if __name__ == "__main__":
    main()
