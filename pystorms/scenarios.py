from pyswmm_lite import environment
from pystorms.utilities import threshold, perf_metrics
from pystorms.networks import load_network
import numpy as np
import abc


# Create a abstract class to force the class definition
class scenario(abc.ABC):
    @abc.abstractmethod
    # Specific to the scenario
    def step(self, actions, log=True):
        pass

    @abc.abstractmethod
    # Specific to the scenario
    def _logger(self):
        pass

    def state(self):
        return self.env._state()

    def performance(self, metric="mean"):
        return perf_metrics(self.data_log["performance_measure"], metric)


class gamma(scenario):
    r"""Gamma Benchmarking Scenario

    Separated stormwater network driven by a __ __ event.

    Parameters
    ----------
    config : dict
        physical attributes of the network.

    Methods
    ----------

    Notes
    -----
    """

    def __init__(self):
        # Network configuration
        self.config = {
            "swmm_input": load_network("gamma"),
            "states": [
                ("1", "depthN"),
                ("2", "depthN"),
                ("3", "depthN"),
                ("4", "depthN"),
                ("5", "depthN"),
                ("6", "depthN"),
                ("7", "depthN"),
                ("8", "depthN"),
                ("9", "depthN"),
                ("10", "depthN"),
                ("11", "depthN"),
            ],
            "action_space": [
                "O1",
                "O2",
                "O3",
                "O4",
                "O5",
                "O6",
                "O7",
                "O8",
                "O9",
                "O10",
                "O11",
            ],
            "performance_targets": [
                ("O1", "flow"),
                ("O2", "flow"),
                ("O3", "flow"),
                ("O4", "flow"),
                ("O5", "flow"),
                ("O6", "flow"),
                ("O7", "flow"),
                ("O8", "flow"),
                ("O9", "flow"),
                ("O10", "flow"),
                ("O11", "flow"),
                ("1", "flooding"),
                ("2", "flooding"),
                ("3", "flooding"),
                ("4", "flooding"),
                ("5", "flooding"),
                ("6", "flooding"),
                ("7", "flooding"),
                ("8", "flooding"),
                ("9", "flooding"),
                ("10", "flooding"),
                ("11", "flooding"),
            ],
        }

        # Common threhold for the network, can be done independently
        self._performormance_threshold = 4.0

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {"performance_measure": [], "flow": {}, "flooding": {}}

        # Data logger for storing _performormance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions, log=True):
        # Implement the actions and take a step forward
        _, done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the _performormance
        __performorm = 0.0  # temp variable

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                flood = self.env.methods[attribute](ID)
                if flood > 0.0:
                    __performorm += 10 ** 6
            else:
                _target = self._performormance_threshold
                __performorm += threshold(
                    self.env.methods[attribute](ID), _target, scaling=1.0
                )

        # Record the _performormance
        self.data_log["performance_measure"].append(__performorm)

        # Terminate the simulation
        if done:
            self.env._terminate()

        return done

    def _logger(self):
        # Log all the _performormance values
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID].append(self.env.methods[attribute](ID))


class theta(scenario):
    r"""Theta Benchmarking Scenario

    Separated stormwater network driven by a __ __ event.

    Parameters
    ----------
    config : dict
        physical attributes of the network.

    Methods
    ----------


    Notes
    -----
    Notes about the peformance metric.

    """

    def __init__(self):
        # Network configuration
        self.config = {
            "swmm_input": load_network("theta"),
            "states": [("P1", "depthN"), ("P2", "depthN")],
            "action_space": ["1", "2"],
            "performance_targets": [
                ("8", "flow"),
                ("P1", "flooding"),
                ("P2", "flooding"),
            ],
        }

        self.threshold = 0.5

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {"performance_measure": [], "flow": {}, "flooding": {}}

        # Data logger for storing _performormance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions, log=True):
        # Implement the actions and take a step forward
        _, done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the performormance
        _perform = 0.0

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                flood = self.env.methods[attribute](ID)
                if flood > 0.0:
                    _perform += 10 ** 6
            if attribute == "flow":
                flow = self.env.methods[attribute](ID)
                _perform = threshold(value=flow, target=self.threshold, scaling=10.0)

        # Record the _performormance
        self.data_log["performance_measure"].append(_perform)

        # Terminate the simulation
        if done:
            self.env._terminate()

        return done

    def _logger(self):
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID].append(self.env.methods[attribute](ID))
