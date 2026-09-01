# Baseline controllers

Controller implementations, parameter optimization and analysis scripts for the
pystorms version 2 scenarios and difficulty levels.

## Layout

Each scenario directory holds the same four scripts:

| Script | Purpose |
| --- | --- |
| `optimize_parameters.py` | Tunes controller parameters by Bayesian optimization and writes `v<N>/optimal_*.txt` |
| `evaluate_baseline_controllers.py` | Runs the tuned controllers and writes result logs |
| `compare_levels.py` | Plots performance across difficulty levels |
| `compare_timeseries.py` | Plots state and action trajectories |

At the top level, `controller_performance_summary.py` and `pareto_front.py`
aggregate results across scenarios, and `bouc_example.py` produces the
Bayesian optimization under unknown constraints example.

The tuned parameters in `<scenario>/v<N>/optimal_*.txt` are the output of
optimization runs and are the inputs the evaluation scripts read. Result logs
and figures are not tracked here; run the scripts to regenerate them, or see
the `dev` branch for the set used in the manuscript.

## Running

The scripts resolve paths relative to their own directory, so run them from
inside the scenario directory:

```bash
cd theta
python evaluate_baseline_controllers.py
```

## Requirements

Beyond pystorms itself:

```
matplotlib
pandas
scipy
networkx
dill
```

`optimize_parameters.py` additionally needs `trieste` (which pulls in
TensorFlow) and `scikit-optimize`. `theta/benchmarking_dev.py` uses `modpods`.

## Reproducibility

Level 1 runs are deterministic and reproduce exactly. Levels 2 and 3 draw
sensor drift, bias and the stuck sensor and actuator schedules from an unseeded
global RNG, so repeated runs of the same scenario produce different faults and
different numbers. Seed `numpy.random` before constructing a scenario if you
need repeatable faults.
