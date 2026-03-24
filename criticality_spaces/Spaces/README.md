# Spaces

Pre-generated synthetic criticality spaces used as benchmark test inputs, together
with the scripts that generate and curate them.

---

## Directory Structure

```
Spaces/
├── spacegenerator.py             — Generates and normalises criticality-space JSON files
├── space_comparison.py           — Clusters spaces and selects representative test cases
├── generation_config_files/      — JSON configuration files for each dimensionality
├── generated_spaces/             — Pre-generated spaces
└── test_cases/
    └── 2D/
        ├── without_noise/        — Curated 2-D test cases (no noise primitive)
        │   └── vis.py            — Interactive 3-D surface viewer
        └── with_noise/           — Same test cases with an additive Noise primitive
            └── vis.py            — Interactive 3-D surface viewer
```

---

## JSON Space Format

Each file contains a single JSON object with a `"functions"` array. Every element
describes one primitive function that contributes additively to the criticality value:

```json
{
    "functions": [
        {
            "type": "Gaussian",
            "mean": [2.1, 5.4],
            "std_dev": [1.2, 0.8],
            "max_amplitude": 0.74,
            "active_dims": [0, 1]
        },
        {
            "type": "StaticHypercube",
            "dimensions": [[0.5, 3.2], [1.0, 4.5]],
            "static_value": 0.31,
            "active_dims": [0, 1]
        }
    ]
}
```

Supported primitive types: `Gaussian`, `StaticHypercube`, `Ramp`, `Sinus`, `Noise`.
The `active_dims` field lists which input dimensions each primitive acts on.
All spaces are normalised so that their maximum value equals 1.0.

---

## Generating Spaces

Edit the desired configuration file in `generation_config_files/` (e.g. `config6D.json`),
update `CONFIG_PATH` in `spacegenerator.py`, then run:

```bash
cd criticality_spaces/Spaces
python spacegenerator.py
```

Output files are written to `generated_spaces/<output_directory>/`.

---

## Curating Test Cases

After generation, `space_comparison.py` selects a diverse representative subset via
KMedoids clustering on combined spectral (FFT) and feature-distance matrices:

```bash
cd criticality_spaces/Spaces
python space_comparison.py
```

Selected medoid spaces are written to `test_cases/<dim>/without_noise/` and
`test_cases/<dim>/with_noise/` (an additive Noise primitive is appended for the
noisy variant).

---

## Visualising Test Cases

```bash
cd criticality_spaces/Spaces/test_cases/2D/without_noise
python vis.py
```

Renders an interactive 3-D surface plot for every JSON file in the current directory.
