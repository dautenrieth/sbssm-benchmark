"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Analyses a folder of generated criticality-space JSON files, computes
    Fourier-based spectral features and pairwise cosine-similarity distances,
    and selects a diverse representative subset via KMedoids clustering.
    Each selected space is written to ``test_cases/`` in two variants:
    with and without an additive Noise primitive.

    - key role: Offline post-processing script that curates a diverse set of
                benchmark test-case spaces from a larger generated pool.
    - dependency: numpy, scipy, sklearn, sklearn_extra, tqdm, space.Space,
                  functionloader.load_functions_from_json
    - output: test_{idx}.json files written to
              test_cases/<output_directory>/with_noise/ and
              test_cases/<output_directory>/without_noise/.

Usage:
    python space_comparison.py
    # Reads generation_config_files/config10D.json and processes the
    # corresponding generated_spaces/ subdirectory.
"""

from sklearn_extra.cluster import KMedoids
import numpy as np
import random, sys
from scipy.signal import correlate
from pathlib import Path
base_path = Path(__file__).parent
cs_path = (base_path.parent).resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))
from space import Space
from functionloader import load_functions_from_json
from scipy.spatial.distance import squareform, pdist
import json
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations

POINTS_PER_DIM = 40 # Nyquist condition: Δx ≤ 1 / (2 * f_max) - Change based on used functions
MAX_ALLOWED_SAMPLES = 10_000_000 # Change based on available computing power

config_path = Path(__file__).parent / "generation_config_files" / "config10D.json"

# -------------------------------
# Fourier feature calculation
# -------------------------------
def fourier_features_nd(values_nd: np.ndarray) -> tuple[list, np.ndarray]:
    """
    Compute n-dimensional FFT spectral features of a criticality-space grid.

    Applies an n-D FFT to *values_nd*, shifts the zero-frequency component to
    the centre, and extracts two summary statistics from the magnitude spectrum:
    total energy and the number of coefficients required to capture 95 % of
    that energy (spectral bandwidth).

    Parameters
    ----------
    values_nd : np.ndarray
        N-dimensional grid of criticality values (e.g. ``space.values``).

    Returns
    -------
    features : list
        ``[energy, bandwidth]`` — total spectral energy and 95 %-energy
        bandwidth (number of coefficients).
    magnitude_spectrum : np.ndarray
        Magnitude spectrum after ``fftshift``, same shape as *values_nd*.
    """
    fft_result = np.fft.fftn(values_nd)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)

    energy = np.sum(magnitude**2)

    sorted_energy = np.sort(magnitude.ravel()**2)[::-1]
    cumulative_energy = np.cumsum(sorted_energy)
    bandwidth = np.argmax(cumulative_energy >= 0.95 * energy)

    return [energy, bandwidth], magnitude

# -------------------------------
# Correlation between spectra
# -------------------------------
def max_cosine_similarity_permutation(space1: np.ndarray, space2: np.ndarray, normalize=True):
    """
    Compute the maximum cosine similarity between two spectra over all axis permutations.

    Searches all permutations of the array axes of *space2* and returns the
    highest cosine similarity with *space1*, accounting for axis-order
    invariance in the frequency domain.

    Parameters
    ----------
    space1 : np.ndarray
        Reference spectrum (magnitude array).
    space2 : np.ndarray
        Query spectrum; must have the same shape as *space1*.
    normalize : bool, optional
        If True, zero-mean and unit-std normalise each flattened spectrum
        before computing similarity (default True).

    Returns
    -------
    max_similarity : float
        Highest cosine similarity found across all axis permutations.
    best_permutation : tuple of int
        The axis permutation of *space2* that achieved *max_similarity*.
    """
    assert space1.shape == space2.shape, "Spectra must have the same shape."
    flat1 = space1.flatten()
    if normalize:
        flat1 = (flat1 - np.mean(flat1)) / (np.std(flat1) + 1e-8)

    max_similarity = -np.inf
    best_permutation = None

    for perm in permutations(range(space1.ndim)):
        permuted = np.transpose(space2, perm).flatten()
        if normalize:
            permuted = (permuted - np.mean(permuted)) / (np.std(permuted) + 1e-8)

        sim = cosine_similarity(flat1.reshape(1, -1), permuted.reshape(1, -1))[0, 0]
        if sim > max_similarity:
            max_similarity = sim
            best_permutation = perm

    return max_similarity, best_permutation

# -------------------------------
# Feature extraction from space
# -------------------------------
def get_feature_vector(values):
    """
    Extract a fixed-length feature vector from a criticality-space value grid.

    Combines Fourier spectral features (energy, bandwidth) with first-order
    statistics (mean, std) and two threshold exceedance fractions.

    Parameters
    ----------
    values : np.ndarray
        N-dimensional grid of criticality values.

    Returns
    -------
    feature_vector : list of float
        ``[energy, bandwidth, mean, std, pct_over_0.6, pct_over_0.8]``.
    magnitude : np.ndarray
        Magnitude spectrum returned by :func:`fourier_features_nd`.
    """
    (energy, bandwidth), magnitude = fourier_features_nd(values)
    mean = np.mean(values)
    std = np.std(values)
    pct_over_06 = np.sum(values > 0.6) / values.size
    pct_over_08 = np.sum(values > 0.8) / values.size

    feature_vector = [energy, bandwidth, mean, std, pct_over_06, pct_over_08]
    return feature_vector, magnitude

# -------------------------------
# Process a single file
# -------------------------------
def process_single_file(file, dimensions):
    """
    Load a space JSON file, compute its feature vector, and persist stats.

    Instantiates a :class:`Space` from *file*, evaluates it on a
    ``POINTS_PER_DIM``-resolution grid, extracts features via
    :func:`get_feature_vector`, and writes the resulting stats back into the
    JSON file via :func:`update_json_with_stats`.

    Parameters
    ----------
    file : pathlib.Path
        Path to a criticality-space JSON file.
    dimensions : list of tuple
        Per-dimension ``(lower, upper)`` bounds.

    Returns
    -------
    features : list of float or None
        Feature vector, or ``None`` if the file produced invalid data.
    magnitudes : np.ndarray or None
        Magnitude spectrum, or ``None`` on failure.
    """
    function_instances = load_functions_from_json(file)
    space = Space(dimensions=dimensions, functions=function_instances, n_points=POINTS_PER_DIM)

    features, magnitudes = get_feature_vector(space.values)

    if features[0] == -1:
        print(f"Skipping {file} due to invalid data.")
        return None, None

    # Stats dictionary
    stats = {
        "energy": features[0],
        "bandwidth": features[1],
        "mean": features[2],
        "std": features[3],
        "pct_over_06": features[4],
        "pct_over_08": features[5]
    }

    # Update stats to file
    update_json_with_stats(file, stats)

    return features, magnitudes

def wrapper_process(file_path, config, seed=None):
    """
    Multiprocessing wrapper around :func:`process_single_file`.

    Sets the per-worker RNG state when a *seed* is provided, then delegates
    to :func:`process_single_file` with bounds parsed from *config*.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the space JSON file to process.
    config : dict
        Configuration dict; must contain ``"bounds_dimensions"``.
    seed : int or None, optional
        Optional RNG seed for reproducibility (default None).

    Returns
    -------
    tuple
        ``(features, magnitudes)`` forwarded from :func:`process_single_file`.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dimensions = [tuple(dim) for dim in config["bounds_dimensions"]]
    return process_single_file(file_path, dimensions)

# -------------------------------
# Update JSON with stats
# -------------------------------
def update_json_with_stats(file_path, stats):
    """
    Write a ``"stats"`` key into an existing space JSON file in-place.

    Serialises numpy scalars and arrays to native Python types before
    writing, ensuring the resulting JSON is valid.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the JSON file to update.
    stats : dict
        Dictionary of scalar statistics to store under the ``"stats"`` key.
    """
    def sanitize_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(sanitize_json(x) for x in obj)
        elif isinstance(obj, list):
            return [sanitize_json(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: sanitize_json(v) for k, v in obj.items()}
        else:
            return obj

    clean_stats = {k: sanitize_json(v) for k, v in stats.items()}

    with open(file_path, 'r') as f:
        data = json.load(f)

    data["stats"] = clean_stats

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    return

def generate_random_noise_from_config(config):
    """
    Generate a random Noise primitive function descriptor from config.

    Parameters
    ----------
    config : dict
        Must contain ``"noise_params"`` with ``"amplitude_range"`` and
        ``"std_dev_range"`` keys.

    Returns
    -------
    dict
        Noise function descriptor with keys ``type``, ``amplitude``,
        ``std_dev``.
    """
    amplitude = random.uniform(*config["noise_params"]["amplitude_range"])
    std_dev = random.uniform(*config["noise_params"]["std_dev_range"])
    return {
        "type": "Noise",
        "amplitude": amplitude,
        "std_dev": std_dev
    }

def skip_and_copy_if_oversized(config, base_path, json_files) -> bool:
    """
    Early-exit check to skip expensive processing if the required grid size is too large.

    - Computes required_samples = points_per_dim ** n_dim
    - If required_samples > max_allowed_samples:
        * creates/cleans target dirs 'with_noise' and 'without_noise',
        * selects up to n_cluster files (randomly if there are more),
        * copies them via `write_to_json` (with and without noise),
        * returns True to signal that heavy processing should be skipped.
    - Otherwise returns False (continue with clustering pipeline).

    Parameters
    ----------
    config : dict
        Needs keys:
          - "bounds_dimensions": list of [min, max] per dimension
          - "n_cluster": int
          - "output_directory": str
          - "noise_params": dict with amplitude/std_dev ranges
          - "random_seed": int (optional, default=42)
    base_path : pathlib.Path
        Base path of the project.
    json_files : list[pathlib.Path]
        Candidate JSON files.

    Returns
    -------
    bool
        True if heavy processing was skipped (copy-only performed).
        False otherwise (safe to proceed).
    """
    n_dim = config["num_dimensions"]
    required_samples = POINTS_PER_DIM ** n_dim
    max_allowed = MAX_ALLOWED_SAMPLES

    if required_samples <= max_allowed:
        return False  # proceed with full processing

    # --- Copy-only mode ---
    target_base = base_path / 'test_cases' / config["output_directory"]
    target_dir_with_noise = target_base / 'with_noise'
    target_dir_without_noise = target_base / 'without_noise'

    for path in (target_dir_with_noise, target_dir_without_noise):
        path.mkdir(parents=True, exist_ok=True)
        for f in path.glob("*.json"):
            f.unlink()

    n_clusters = int(config["n_cluster"])
    rng = random.Random(config.get("random_seed", 42))
    if len(json_files) > n_clusters:
        selected_files = rng.sample(list(json_files), n_clusters)
    else:
        selected_files = list(json_files)

    for idx, src_file in enumerate(sorted(selected_files), 1):
        write_to_json(
            src_file=src_file,
            idx=idx,
            target_dir_with_noise=target_dir_with_noise,
            target_dir_without_noise=target_dir_without_noise,
            config=config,
            echo=True,
        )

    print(
        f"Required samples {required_samples:,} exceed max_allowed_samples {max_allowed:,}. "
        f"Skipped heavy processing and only copied {len(selected_files)} files."
    )
    return True

def write_to_json(src_file, idx, target_dir_with_noise, target_dir_without_noise, config, *, echo=True):
    """
    Read functions from `src_file` and write two test-case JSONs:
    - without noise:  test_{idx}.json in `target_dir_without_noise`
    - with noise:     test_{idx}.json in `target_dir_with_noise` (adds one noise function)

    Parameters
    ----------
    src_file : pathlib.Path
        Source JSON file containing {"functions": [...] }.
    idx : int
        1-based index used for output filenames (test_{idx}.json).
    target_dir_with_noise : pathlib.Path
    target_dir_without_noise : pathlib.Path
    config : dict
        Must contain "noise_params" used by `generate_random_noise_from_config`.
    echo : bool, optional
        If True, prints a short copy message.

    Returns
    -------
    (path_without, path_with) : tuple[pathlib.Path, pathlib.Path]
        Paths of the written files.
    """
    with open(src_file, "r") as f:
        data = json.load(f)
    functions = data.get("functions", [])

    # without noise
    target_file_without_noise = target_dir_without_noise / f"test_{idx}.json"
    with open(target_file_without_noise, "w") as f:
        json.dump({"functions": functions}, f, indent=4)

    # with noise
    noise_func = generate_random_noise_from_config(config)
    functions_with_noise = functions + [noise_func]
    target_file_with_noise = target_dir_with_noise / f"test_{idx}.json"
    with open(target_file_with_noise, "w") as f:
        json.dump({"functions": functions_with_noise}, f, indent=4)

    if echo:
        print(f"Copied {src_file.name} → {target_file_with_noise.name} (with/without noise)")

    return target_file_without_noise, target_file_with_noise

# -------------------------------
# Main processing function
# -------------------------------
def process_folder(config, base_path):
    """
    Analyse all spaces in a generated folder and write clustered test cases.

    Parallelises feature extraction via a ``ProcessPoolExecutor``, builds a
    combined spectral + feature-distance matrix, clusters with KMedoids, and
    writes the medoid spaces to the test-case directories via
    :func:`write_to_json`.  Falls back to a copy-only path for high-dimensional
    spaces that exceed ``MAX_ALLOWED_SAMPLES``.

    Parameters
    ----------
    config : dict
        Configuration dict with keys ``"n_cluster"``, ``"output_directory"``,
        ``"bounds_dimensions"``, ``"noise_params"``, etc.
    base_path : pathlib.Path
        Repository root used to construct all relative paths.
    """
    n_clusters = config["n_cluster"]
    folder_name = config["output_directory"]

    folder_path = base_path / 'generated_spaces' / folder_name
    json_files = sorted(folder_path.glob("*.json"))

    if len(json_files) < n_clusters:
        print(f"Skipping '{folder_name}': not enough files for clustering.")
        return

    print(f"\n=== Processing folder {folder_name} ===")

    if skip_and_copy_if_oversized(config, base_path, json_files):
        return

    results = [None] * len(json_files)
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(wrapper_process, file, config, 42 + i): i
            for i, file in enumerate(json_files)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Analyzing ({folder_name})"):
            i = futures[future]
            results[i] = future.result()

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print(f"No valid data found in '{folder_name}' - skipped.")
        return

    feature_vectors, magnitudes = zip(*valid_results)
    feature_vectors = np.array(feature_vectors)

    # Compute correlation matrix
    n = len(feature_vectors)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim, _ = max_cosine_similarity_permutation(magnitudes[i], magnitudes[j], normalize=False)
            corr_matrix[i, j] = corr_matrix[j, i] = 1 - sim

    # Compute feature distance matrix
    dist_matrix = squareform(pdist(feature_vectors, metric='euclidean'))

    # Normalize and combine
    scaler = MinMaxScaler()
    corr_matrix_norm = scaler.fit_transform(corr_matrix)
    dist_matrix_norm = scaler.fit_transform(dist_matrix)
    combined_matrix = 0.7 * corr_matrix_norm + 0.3 * dist_matrix_norm

    # Clustering
    print(f"Clustering {folder_name} using KMedoids...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', random_state=42)
    kmedoids.fit(combined_matrix)

    # Prepare output directories
    target_base = base_path / 'test_cases' / config["output_directory"]
    target_dir_with_noise = target_base / 'with_noise'
    target_dir_without_noise = target_base / 'without_noise'

    for path in [target_dir_with_noise, target_dir_without_noise]:
        path.mkdir(parents=True, exist_ok=True)
        for f in path.glob("*.json"):
            f.unlink()

    for idx, medoid_idx in enumerate(kmedoids.medoid_indices_, 1):
        src_file = json_files[medoid_idx]

        write_to_json(src_file, idx, target_dir_with_noise /
                      f"test_{idx}.json", target_dir_without_noise / f"test_{idx}.json", config)

    print(f"Finished processing {folder_name}.\n")


# -------------------------------
# Entry point
# -------------------------------
def main():
    """
    Entry point: load config and run the clustering pipeline for one folder.

    Reads ``config_path`` (``config10D.json`` by default) and calls
    :func:`process_folder` to analyse, cluster, and export test-case spaces.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    process_folder(config, base_path)

    print("All folder analyses complete.")


if __name__ == "__main__":
    main()
