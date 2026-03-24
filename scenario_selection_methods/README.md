# scenario_selection_methods

Implements nine Search-Based Scenario Selection Methods (SBSSMs) that operate on a
`criticality_spaces.Space` instance. All methods share a common `BaseSelector`
interface, making them interchangeable within the evaluation pipeline.

---

## Implemented Methods

| File | Class(es) | Method |
|------|-----------|--------|
| `selector_random.py` | `RandomSelector`, `SobolSelector`, `LatinHypercubeSelector` | Uniform random, Sobol sequence, Latin Hypercube Sampling |
| `selector_idw.py` | `IDWSelector` | Inverse Distance Weighting surrogate (k-NN via KD-Tree) |
| `selector_gpr.py` | `GPSTLSelector` | Gaussian Process Regression (Torben et al., 2023) |
| `selector_ann.py` | `ANNSelector` | Artificial Neural Network surrogate (PyTorch MLP) |
| `selector_ipso.py` | `IPSOSearchSelector` | Improved Particle Swarm Optimisation |
| `selector_nndv.py` | `NNDVSelector` | Nearest-Neighbour Diversity / Voronoi |
| `selector_gdnnas.py` | `GDNNASSelector` | Global Diversity with Nearest-Neighbour Adaptive Sampling |

---

## Quick Start

### Run a single selector manually

```python
from space import Space
from functionloader import load_functions_from_json
from factory import get_selector

# Load space (from criticality_spaces/)
import sys
sys.path.insert(0, "../criticality_spaces")
fns = load_functions_from_json("../criticality_spaces/Spaces/generated_spaces/2D/6_2D.json")
space = Space(dimensions=2, functions=fns, n_points=200, criticality_thresholds=[0.3, 0.9])

# Instantiate and run a selector
selector = get_selector("GPR", space=space, n_select=50)
selector.run()

# Evaluate
results = space.metrics.run_metrics_suite(method_categories=["general", "boundary_detection"])
print(results)
```

### Run the full benchmark

```bash
cd scenario_selection_methods
python main.py
```

`main.py` runs all selectors across all configured spaces, logs results, and
writes XLSX output to `evaluations/`.

---

## Selector Interface

All selectors implement the `BaseSelector` interface from `base.py`:

```python
class BaseSelector:
    def __init__(self, space: Space, n_select: int): ...
    def run(self) -> None: ...              # executes the selection
    def get_selected_points(self): ...      # returns selected scenario points
```

Model-based selectors (GPR, ANN, IDW) additionally extend `BaseModelSelector`,
which adds `fit()` and `predict()` methods for the surrogate model.

---

## Factory

Use `factory.get_selector(name, space, n_select)` to instantiate any selector by name:

```python
from factory import get_selector
selector = get_selector("IPSO", space=space, n_select=100)
```

Valid names: `"Random"`, `"Sobol"`, `"LHS"`, `"IDW"`, `"GPR"`, `"ANN"`,
`"IPSO"`, `"NNDV"`, `"GDNNAS"`.

---

## Evaluation Output

Results are stored in `evaluations/` organised by metric category:

```
evaluations/
├── Boundary Identification/
├── Extremum Search/
├── Global Coverage/
├── Landscape Reconstruction/
├── evaluation.log
└── evaluation_results_<timestamp>.xlsx
```

Each XLSX file contains one sheet per metric category with rows for each
(space, selector, run) combination.

---

## Integration with criticality_spaces

Each selector receives a `Space` object and queries it via
`space.get_values_for_points(points)`. The space's internal point-value cache
makes repeated queries efficient. After `selector.run()`, the same `Space`
object holds the `selected_points` list, which is consumed directly by
`space.metrics.run_metrics_suite()`.

The `criticality_spaces/` module must be importable (add it to `sys.path` or
install it) for the selectors to function.
