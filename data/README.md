# data

Contains two simulation space datasets and the Python scripts that generate
publication-quality plots from them. These scripts are standalone and do not depend
on `criticality_spaces/` or `scenario_selection_methods/`.

---

## Contents

```
data/
├── DLR/
│   ├── parameters_and_results.csv    — Simulation results from DLR test scenarios
│   └── generate_plots_from_csv.py    — Generates 3-D surface and analysis plots
│
└── TKMS/
    ├── dataframettc.csv              — Time-to-collision metrics from TKMS scenarios
    └── generate_plots_from_runs.py   — Generates plots from simulation run data
```

---

## Datasets

### DLR (`DLR/parameters_and_results.csv`)
Scenario parameter sweeps and associated simulation results from DLR (Deutsches Zentrum
für Luft- und Raumfahrt). Each row corresponds to one scenario instance with parameter
values and outcome metrics. Used to produce 3-D criticality surface plots for the paper.

### TKMS (`TKMS/dataframettc.csv`)
Criticality and parameter data from TKMS maritime scenario simulations. Each row
contains scenario parameters and different outcome metrics, which serves as the
criticality proxy.

---

## Plot Generation

Each subfolder contains a self-contained script. Run from the respective subfolder:

```bash
# DLR plots
cd data/DLR
python generate_plots_from_csv.py

# TKMS plots
cd data/TKMS
python generate_plots_from_runs.py
```

Both scripts use IEEE-style Matplotlib configuration and export figures as PNG or PDF.
No arguments are required; input paths are resolved relative to the script location.

---

## Relation to the Benchmark

These datasets represent simulation-based validation use cases that complement the synthetic
`criticality_spaces/` benchmark. They are not used directly by
`scenario_selection_methods/` but serve as qualitative reference points for the
criticality surfaces produced by the framework.
