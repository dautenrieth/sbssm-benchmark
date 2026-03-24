# criticality_spaces

Implements the synthetic criticality space framework used for standardised benchmarking
of Search-Based Scenario Selection Methods (SBSSMs). A `Space` represents a
d-dimensional scenario parameter domain whose criticality surface is defined by summing
analytic primitive functions loaded from JSON.

---

## Module Overview

| Module | Role |
|--------|------|
| `space.py` | Central `Space` class — meshgrid evaluation, caching, discretisation, boundary detection |
| `functions.py` | Primitive criticality shapes: `Gaussian`, `Ramp`, `StaticHypercube`, `Noise`, `Sinus` |
| `functionloader.py` | `load_functions_from_json()` — deserialises JSON space specs into function instances |
| `distributions.py` | `DistributionHandler` — per-axis input distributions for importance-weighted sampling |
| `samplingstrategies.py` | Space-filling strategies: random, grid-LHS, continuous LHS |
| `metrics.py` | Full metric suite across four evaluation categories |
| `metric_categories.py` | Maps category keys to metric sets consumed by `Metrics.run_metrics_suite()` |
| `visualizations.py` | `SpaceVisualizer` — Plotly and Matplotlib plots (surface, heatmap, boundary) |
| `run_demo.py` | End-to-end demo: load space → sample → evaluate → visualise |

---

## Quick Start

```python
from space import Space
from functionloader import load_functions_from_json
import samplingstrategies as ss

# 1. Load a pre-generated 2-D criticality space
fns = load_functions_from_json("Spaces/generated_spaces/2D/6_2D.json")

# 2. Construct the Space (evaluates the full 200x200 meshgrid on construction)
space = Space(
    dimensions=2,                      # 2 dimensions with default bounds (0, 10)
    functions=fns,
    n_points=200,                      # grid resolution per dimension
    criticality_thresholds=[0.3, 0.9], # partition into 3 criticality classes
)

# 3. Sample and evaluate
samples = ss.latin_hypercube_sampling([(0, 10), (0, 10)], n_samples=50)
space.get_values_for_points(samples)

# 4. Compute metrics
results = space.metrics.run_metrics_suite(method_categories=["general"])

# 5. Visualise
space.visualizer.plot_3d_two_varied(show_critplane=True)
```

Run the demo directly:
```bash
cd criticality_spaces
python run_demo.py
```

---

## Space Definition (JSON)

Spaces are defined as JSON files with a `"functions"` list. Each entry specifies a
primitive type and its parameters. Multiple primitives are summed to form the composite
criticality surface.

```json
{
  "functions": [
    {
      "type": "Gaussian",
      "mean": [5.0, 5.0],
      "std_dev": [1.5, 1.5],
      "max_amplitude": 1.0
    },
    {
      "type": "Noise",
      "amplitude": 0.05,
      "seed": 42
    }
  ]
}
```

Supported types: `Gaussian`, `StaticHypercube`, `Ramp`, `Noise`, `Sinus`.

See `Spaces/examples/` for annotated templates and `Spaces/generation_config_files/`
for the configurations used to generate the benchmark spaces.

---

## Pre-generated Spaces

The benchmark suite includes 1,230 pre-generated spaces:

| Folder | Dimensionality | Count |
|--------|---------------|-------|
| `Spaces/generated_spaces/2D/` | 2-D | 100 |
| `Spaces/generated_spaces/3D/` | 3-D | 1000 |
| `Spaces/generated_spaces/4D/` | 4-D | 100 |
| `Spaces/generated_spaces/6D/` | 6-D | 10 |
| `Spaces/generated_spaces/8D/` | 8-D | 10 |
| `Spaces/generated_spaces/10D/` | 10-D | 10 |

Filtered spaces (as described in the paper) can be found in the `Spaces/test_cases/`-folder.

---

## Metric Categories

| Category key | Metrics |
|---|---|
| `"general"` | average/min/max criticality (full & selected), spatial entropy, discrepancy |
| `"extremum_search"` | convergence rate, extremum gap |
| `"model_reconstruction"` | MSE, R2, F1 coverage (requires a fitted surrogate model) |
| `"boundary_detection"` | boundary precision, effectiveness, coverage |

Pass one or more keys to `run_metrics_suite`:
```python
space.metrics.run_metrics_suite(
    method_categories=["general", "boundary_detection"]
)
```

---

## Input Distributions (Optional, Future Work)

> **Note:** This feature is not actively used in the benchmark paper and is
> intended for future work. Only a single example spec is provided
> (`Distributions/dist3d.json`).

To model non-uniform scenario distributions, supply a JSON distribution spec
(see `Distributions/dist3d.json` for an example):

```json
{
  "distributions": [
    {"type": "norm", "loc": 5.0, "scale": 2.0},
    {"type": "uniform", "loc": 0.0, "scale": 10.0}
  ]
}
```

Pass the loaded spec to `Space(distribution_specs=...)`.
Query density at a point via `space.get_density_for_point(point)`.

---

## Standalone Use vs. Integration

**Standalone:** Load any JSON space, apply a sampling strategy from
`samplingstrategies.py`, and evaluate with `Metrics`. No external dependencies
beyond the modules listed above.

**Integrated with `scenario_selection_methods/`:** A `Space` instance is passed
directly to a selector. The selector calls `space.get_values_for_points()` internally;
metrics are evaluated on the same `Space` object after the run.
