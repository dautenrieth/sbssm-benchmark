# Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation

This repository contains the full implementation accompanying the benchmark paper.
It is organised into three independent but interoperable components:

| Folder | Purpose |
|--------|---------|
| [`criticality_spaces/`](criticality_spaces/) | Synthetic criticality spaces and evaluation metrics |
| [`scenario_selection_methods/`](scenario_selection_methods/) | Search-based scenario selection algorithms |
| [`data/`](data/) | Simulationrun datasets and plot generation |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    criticality_spaces/                      │
│   JSON space definitions  →  Space  →  Metrics & Plots      │
│                                ↑                            │
└────────────────────────────────┼────────────────────────────┘
                                 │  Space instance shared via API
┌────────────────────────────────┼────────────────────────────┐
│              scenario_selection_methods/                    │
│   Selector (e.g. GPR, IPSO, ANN) calls                      │
│   space.get_values_for_points(samples)                      │
│   space.metrics.run_metrics_suite(...)                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         data/                               │
│   Simulation CSV datasets  →  Standalone plot scripts       │
│   (DLR, TKMS)                                               │
└─────────────────────────────────────────────────────────────┘
```

The typical evaluation workflow is:

1. **Load a space** from a JSON file in `criticality_spaces/Spaces/generated_spaces/`.
2. **Instantiate a selector** from `scenario_selection_methods/` and pass it the `Space` object.
3. **Run the selector**, which calls `space.get_values_for_points()` to query the criticality surface.
4. **Evaluate** using `space.metrics.run_metrics_suite()` across one or more metric categories.
5. **Visualise** results via `space.visualizer` or the standalone scripts in `data/`.

---

## Component Summaries

### `criticality_spaces/`
Implements the `Space` class, which represents a d-dimensional scenario parameter space
composed of analytic criticality functions (Gaussian bumps, ramps, sinusoids, etc.).
Spaces are defined in JSON, loaded via `functionloader`, and evaluated over a meshgrid.
The module provides a full metric suite (general statistics, extremum search, surrogate
model fidelity, boundary detection) and interactive Plotly/Matplotlib visualisations.

→ See [`criticality_spaces/README.md`](criticality_spaces/README.md) for full details.

### `scenario_selection_methods/`
Implements nine Search-Based Scenario Selection Methods (SBSSMs) sharing a common
`BaseSelector` interface: Random, Sobol, Latin Hypercube, IDW, GPR, ANN, IPSO, NNDV,
and GDNNAS. Each selector is instantiated with a `Space` object and a target sample
budget, then run to produce a set of selected scenario points for metric evaluation.

→ See [`scenario_selection_methods/README.md`](scenario_selection_methods/README.md) for full details.

### `data/`
Contains two datasets with the simulation results (DLR, TKMS) in CSV format and the Python scripts
that generate publication-quality plots from them. These are standalone and do not
depend on the other modules.

→ See [`data/README.md`](data/README.md) for full details.

---

## Reproducibility

All synthetic spaces used in the paper are pre-generated and stored in
`criticality_spaces/Spaces/generated_spaces/`.
Generation configurations are in `criticality_spaces/Spaces/generation_config_files/`.
Some evaluation results are stored as XLSX files in
`scenario_selection_methods/evaluations/`. Summaries of different runs are provided on the second sheet.

If you use this benchmark in your research, please cite the benchmark paper accordingly.

---

## Dependencies

The codebase requires Python ≥ 3.9. Key dependencies:

```
numpy  scipy  scikit-learn  scikit-learn-extra  matplotlib  plotly  tqdm  torch  pandas  openpyxl
```

Install via:
```bash
pip install numpy scipy scikit-learn scikit-learn-extra matplotlib plotly tqdm torch pandas openpyxl
```
