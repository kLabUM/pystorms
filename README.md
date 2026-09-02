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


## Getting Started 

### Installation 

**Requirements**

- Python >= 3.9
- PyYAML >= 5.3
- numpy >= 18.4
- pyswmm >= 1.0.1
- pandas >= 1.0
- swmmio >= 0.6.11


```bash 
pip install pystorms
```

Please raise an issue on the repository or reach out if you run into any issues installing the package. 

> On macOS 26 the `swmm-toolkit` wheel ships ad hoc signed libraries that a
> hardened Python refuses to load, and the interpreter is killed with no
> traceback. If `import pystorms` dies silently, re-sign them:
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

### Scenario versions

Harder variants of _theta_, _alpha_, _gamma_, _delta_ and _epsilon_ are selected with the `version` keyword. `"1"` is the default and matches the originally published scenarios; `"2"` tightens the objective and modifies the network.

```python
env = pystorms.scenarios.theta(version="2")
```

### Difficulty levels

The `level` keyword decides how trustworthy the instrumentation is. `"1"` is the default and gives perfect readings. `"2"` adds sensor noise, drift and calibration bias, and lets valves stick. `"3"` intensifies all of that and adds sensors that drop out entirely. Version and level are independent, so you can hold the network fixed and vary only the measurement quality.

`level` has to be passed in **two** places: once when building the scenario, which draws the fault schedule, and again on every `state()` and `step()` call, which applies it.

```python
level = "2"

env = pystorms.scenarios.theta(level=level)   # draws the faults
done = False
while not done:
    state = env.state(level=level)            # applies noise, drift and dropouts
    done = env.step(controller(state), level=level)  # applies stuck valves
```

Passing it in only one place will either silently give you clean readings or raise an `AttributeError`.

Fault schedules are drawn from numpy's global random state, so seed it before building a scenario if you need a level 2 or level 3 comparison to be repeatable:

```python
np.random.seed(42)
env = pystorms.scenarios.theta(level="3")
```

More details on the updates are accessible at (preprint link).

## Tutorials

The [`tutorials`](tutorials) directory has runnable notebooks for each scenario, plus:

| Notebook | Topic |
| --- | --- |
| [`Versions_and_Levels.ipynb`](tutorials/Versions_and_Levels.ipynb) | The `version` and `level` keywords, end to end |
| [`RuleBasedControl.ipynb`](tutorials/RuleBasedControl.ipynb) | A threshold controller, and a sweep over its parameters |
| [`BayesianOptimization.ipynb`](tutorials/BayesianOptimization.ipynb) | Tuning a controller with Bayesian optimization |
| [`ReinforcementLearning.ipynb`](tutorials/ReinforcementLearning.ipynb) | Training a deep Q network on _theta_ |

[`baseline_controllers`](baseline_controllers) holds the controller implementations, tuned parameters and analysis scripts behind the accompanying manuscript.

Detailed documentation can be found on the [webpage](https://www.pystorms.org)
