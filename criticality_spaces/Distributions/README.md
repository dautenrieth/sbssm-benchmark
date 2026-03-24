# Distributions

This folder contains input distribution specifications for the `Space` class.

> **Note:** Input distribution support is a planned feature for future work and
> is **not actively used** in the benchmark paper. Accordingly, only a single
> example file (`dist3d.json`) is provided here.

---

## Purpose

A distribution spec allows each axis of the scenario parameter space to be
assigned an independent probability distribution (e.g. normal, uniform,
Weibull). When supplied to `Space(distribution_specs=...)`, the
`DistributionHandler` enables importance-weighted queries via
`space.get_density_for_point(point)`.

## File Format

Each JSON file contains a `"distributions"` list with one entry per dimension.
Supported types are any univariate distribution available in `scipy.stats`:

```json
{
  "distributions": [
    {"type": "norm",        "loc": 5.0, "scale": 2.0},
    {"type": "uniform",     "loc": 0.0, "scale": 10.0},
    {"type": "weibull_min", "c": 1.5,   "scale": 3.0}
  ]
}
```

Pass the parsed JSON to the `Space` constructor:

```python
import json
from space import Space
from functionloader import load_functions_from_json

with open("Distributions/dist3d.json") as f:
    dist_specs = json.load(f)

fns = load_functions_from_json("Spaces/generated_spaces/3D/1_3D.json")
space = Space(dimensions=3, functions=fns, n_points=50,
              distribution_specs=dist_specs)

joint_density, axis_densities = space.get_density_for_point([5.0, 3.0, 2.0])
```
