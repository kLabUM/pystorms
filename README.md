# pystorms: simulation sandbox for the evaluation and design of stormwater control algorithms
[![pystorms](https://github.com/kLabUM/pystorms/actions/workflows/python-package.yml/badge.svg?branch=master&event=push)](https://github.com/kLabUM/pystorms/actions/workflows/python-package.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Overview 

This library has been developed in an effort to systematize quantitative analysis of stormwater control algorithms.
It is a natural extension of the Open-Storm's mission to open up and ease access into the technical world of smart stormwater systems.
 Our initial efforts allowed us to develop open source and free tools for anyone to be able to deploy flood sensors, measure green infrastructure, or even control storm or sewer systems.
 Now we have developed a tool to be able to test the performance of algorithms used to coordinate these different sensing and control technologies that have been deployed throughout urban water systems.    

For the motivation behind this effort, we refer the reader to our manuscript [*pystorms*](https://dl.acm.org/citation.cfm?id=3313336). In general, this repo provides a library of `scenarios` that are built to allow for systematic quantitative evaluation of stormwater control algorithms.

The project website and detailed documentation are at [pystorms.netlify.app](https://pystorms.netlify.app).


## Getting Started 

### Installation 

**Requirements**

- Python >= 3.9
- PyYAML >= 5.3
- numpy >= 1.18.4
- pyswmm >= 1.0.1
- pandas >= 1.0
- swmmio >= 0.6.11


```bash 
pip install pystorms
```

Please raise an issue on the repository or reach out if you run into any issues installing the package. 

> On macOS 26 the `swmm-toolkit` wheel ships ad hoc signed libraries that the
> hardened runtime refuses to load, and the interpreter is killed with no
> traceback. This affects Apple's Python and uv managed interpreters alike. If
> `import pystorms` dies silently, re-sign them:
>
> ```bash
> codesign --force --sign - "$(python -c 'import swmm.toolkit, pathlib; print(pathlib.Path(swmm.toolkit.__file__).parent)')"/*.dylib
> ```

### Example 

Here is an example implementation on how you would use this library for evaluating the ability of a rule based control in maintaining the flows in a network below a desired threshold. 

```python 
import pystorms 
import numpy as np

# Define your awesome controller.
# Hold water back until a basin is more than half full, then release.
def controller(state, trigger=0.5, opening=0.5):
    actions = np.zeros(len(state))
    actions[state > trigger] = opening
    return actions


env = pystorms.scenarios.theta() # Initialize scenario 

done = False
while not done:
    state = env.state()
    actions = controller(state)
    done = env.step(actions)

performance = env.performance()

```

`step` closes the SWMM simulation when the event ends. SWMM allows one open
simulation per process, so if you stop an episode early call `env.terminate()`
before building the next scenario, or use the scenario as a context manager:

```python
with pystorms.scenarios.theta() as env:
    for _ in range(100):
        env.step(controller(env.state()))
# closed here, even if the loop raised
```

### Scenario versions

Harder variants of _theta_, _alpha_, _gamma_, _delta_ and _epsilon_ are selected with the `version` keyword. `"1"` is the default and matches the originally published scenarios; `"2"` tightens the objective and modifies the network. _beta_ and _zeta_ only have version `"1"`, and asking any scenario for a version it does not define raises `ValueError`.

```python
env = pystorms.scenarios.theta(version="2")
```

Version 2 networks are built by rewriting the shipped SWMM input file. The copy, together with the `.rpt` and `.out` files SWMM writes for every run, goes to a per user cache directory: `$PYSTORMS_CACHE` if set, otherwise `$XDG_CACHE_HOME/pystorms/networks` or `~/.cache/pystorms/networks`. Processes that run the same scenario at the same time write the same files there, so give each worker its own `PYSTORMS_CACHE` when running scenarios in parallel.

### Difficulty levels

The `level` keyword decides how trustworthy the instrumentation is. `"1"` is the default and gives perfect readings. `"2"` adds sensor noise, drift and calibration bias, and lets valves stick. `"3"` intensifies all of that and adds sensors that drop out entirely. Version and level are independent, so you can hold the network fixed and vary only the measurement quality.

```python
env = pystorms.scenarios.theta(level="2")   # draws the fault schedule
done = False
while not done:
    state = env.state()                      # noisy, drifting, biased readings
    done = env.step(controller(state))       # stuck valves ignore the command
```

The level is remembered by the scenario. `state()` and `step()` also accept a `level` argument; pass `level="1"` to read the true state or apply every command on a degraded scenario, for instance to log the ground truth next to what the controller saw. Asking for a higher level than the scenario was built with raises `ValueError`.

What the levels do, in detail:

- Noise, drift and bias are drawn when the scenario is built. Noise is Gaussian with a standard deviation of 2.5 cm at level 2 and 15 cm at level 3. Drift affects a random subset of the sensors, at a fraction of a millimetre per day at level 2 and one to two centimetres per day at level 3. Bias is a multiplicative factor within one percent at level 2 and ten percent at level 3.
- Those magnitudes are lengths, and they are applied to every state the same way, including flow and pollutant concentration states. Readings can be negative.
- Each valve has a 60 percent chance at level 2 and an 80 percent chance at level 3 of sticking once, for a random window of 10 to 30 percent (level 2) or 20 to 50 percent (level 3) of the event. A stuck valve keeps the position it had when it stuck.
- At level 3 each sensor has an 80 percent chance of dropping out once, for 5 to 20 percent of the event. A dropped out sensor reports exactly zero.

The fault schedule is available on `env.env` as `drift_rates`, `bias`, `actuator_schedule` and `sensor_schedule`. It is drawn from numpy's global random state, so seed it before building a scenario if you need a level 2 or level 3 comparison to be repeatable:

```python
np.random.seed(42)
env = pystorms.scenarios.theta(level="3")
```

The version 2 scenarios, the difficulty levels and a set of baseline controllers are described in the accompanying manuscript; see [`baseline_controllers`](baseline_controllers) for the controller implementations and tuned parameters.

## Tutorials

The [`tutorials`](tutorials) directory has runnable notebooks for each scenario, plus:

| Notebook | Topic |
| --- | --- |
| [`Versions_and_Levels.ipynb`](tutorials/Versions_and_Levels.ipynb) | The `version` and `level` keywords, end to end |
| [`RuleBasedControl.ipynb`](tutorials/RuleBasedControl.ipynb) | A threshold controller, and a sweep over its parameters |
| [`BayesianOptimization.ipynb`](tutorials/BayesianOptimization.ipynb) | Tuning a controller with Bayesian optimization |
| [`ReinforcementLearning.ipynb`](tutorials/ReinforcementLearning.ipynb) | Training a deep Q network on _theta_ |

[`baseline_controllers`](baseline_controllers) holds the controller implementations, tuned parameters and analysis scripts behind the accompanying manuscript.

Detailed documentation can be found on the [webpage](https://pystorms.netlify.app).

## Changes

See [`CHANGELOG.md`](CHANGELOG.md). Version 2.0 changes some reported numbers: the _epsilon_ objective now reads the pollutant concentration in the outlet conduit rather than the node upstream of it, and level 3 dropouts report zero rather than noise around zero.
