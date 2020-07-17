from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold, exponentialpenalty
import yaml


class delta(scenario):
    r"""Delta Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    -----
    Objective is the following:
    1. To maintain levels of three detention ponds within a range of depths, and
    2. To maintain the network outflow below a threshold.

    Performance is measured as the following:
    1. First, deviation of depth above/below the "desired" depth range for the three detention ponds,
    2. Second, deviation of depth above/below "maximum/minimum" depth ranges for the three detention ponds
        and one other infiltration pond,
    3. Any flooding through the network, and
    4. Any deviation above the threshold of the outflow.

    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(load_config("delta"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        self.threshold = 12.0

        self.depth_thresholds = {
            "basin_C": (5.7, 2.21, 3.8, 3.28),
            "basin_S": (6.55, 9.5),
            "basin_N1": (5.92, 2.11, 5.8, 5.2),
            "basin_N2": (6.59, 4.04, 5.04, 4.44),
            "basin_N3": (11.99, 5.28, 5.92, 5.32),
        }

        # Additional penalty definition
        self.max_penalty = 10 ** 6

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "depthN": {},
            "flow": {},
            "flooding": {},
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True):
        # Implement the actions and take a step forward
        done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the performance
        __performance = 0.0

        for ID, attribute in self.config["performance_targets"]:
            # compute penalty for flooding
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += 10 ** 6
            # compute penalty for flow out of network above threshold
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __performance += threshold(
                    value=__flow, target=self.threshold, scaling=10.0
                )
            # compute penalty for depth at basins above/below predefined ranges
            if attribute == "depthN":
                depth = self.env.methods[attribute](ID)
                temp = self.depth_thresholds[ID]
                if ID == "basin_S":
                    if depth > temp[1]:  # flooding value
                        __performance += 10 ** 6
                    elif depth > temp[0]:
                        __temp = (depth - temp[0]) / (temp[1] - temp[0])
                        __performance += exponentialpenalty(
                            value=__temp, max_penalty=self.max_penalty, scaling=1.0
                        )
                    else:
                        __performance += 0.0
                else:
                    if depth > temp[0] or depth < temp[1]:  # flooding value + fish dead
                        __performance += 10 ** 6
                    elif depth > temp[2]:
                        __temp = (depth - temp[2]) / (temp[0] - temp[2])
                        __performance += exponentialpenalty(
                            value=__temp, max_penalty=self.max_penalty, scaling=1.0
                        )
                    elif depth < temp[3]:
                        __temp = (temp[3] - depth) / (temp[3] - temp[1])
                        __performance += exponentialpenalty(
                            value=__temp, max_penalty=self.max_penalty, scaling=1.0
                        )
                    else:
                        __performance += 0.0

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
