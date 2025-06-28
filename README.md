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

- PyYAML >= 5.3
- numpy >= 18.4
- pyswmm < 2.0.0


```bash 
pip install pystorms
```

Please raise an issue on the repository or reach out if you run into any issues installing the package. 

### Example 

Here is an example implementation on how you would use this library for evaluating the ability of a rule based control in maintaining the flows in a network below a desired threshold. 

```python 
import pystorms 
import numpy as np

# Define your awesome controller 
def controller(state):
	actions = np.ones(len(state))
	for i in range(0, len(state)):
		if state[i] > 0.5:
			actions[i] = 1.0
	return actions 
	

env = pystorms.scenarios.theta() # Initialize scenario 

done = False
while not done:
	state = env.state()
	actions = controller(state)
	done = env.step(actions)

performance = env.performance()

```

Updated versions of _theta_, _alpha_, _gamma_, _delta_, and _epsilon_ are accessible via a version keyword in the initialization command.

```python
env = pystorms.scenarios.theta(version=version) # "1" is the default and original, "2" are the updated versions.
```

Sensor noise and actuator faults can also be enabled via the level keyword. The options are 1, 2, and 3 in ascending order of difficulty. Version or level or both can be specified. 

```python
env = pystorms.scenarios.theta(version=version, level=level) # "1" is the ideal, original, and default case. "2" is realistic and "3" is adverse.
env = pystorms.scenarios.theta(level=level) # also valid. This would load version 1 of the model.
```
More details on the updates are accessible at (preprint link). As of June 2025, these updates are only in the "dev" branch and have not yet been merged in "master."

Detailed documentation can be found on the [webpage](https://www.pystorms.org)
