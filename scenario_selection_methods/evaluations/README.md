# evaluations

Stores raw benchmark result files produced by the evaluation pipeline. Results are
organised by metric category, matching the four evaluation dimensions defined in the
benchmark framework.

---

## Structure

```
evaluations/
├── Boundary Identification/      — Decision-boundary detection metrics
├── Extremum Search/              — Extremum-search metrics
├── Global Coverage/              — Space-filling / coverage metrics
└── Landscape Reconstruction/     — Surrogate model reconstruction metrics
```

---

## File Naming Convention

Individual run files follow the pattern:

```
evaluation_results_<YYYYMMDD>_<HHMMSS>_<budget>_<methods>.xlsx
```

`CombinedResults_*.xlsx` files are manually merged summaries that aggregate results
across multiple runs and are used directly to produce figures in the paper.

---

## Producing Results

Run the evaluation pipeline from the `scenario_selection_methods/` directory:

```bash
cd scenario_selection_methods
python run_evaluation.py
```

Output XLSX files are written automatically to the appropriate subfolder here based
on the metric category selected in the run configuration.
