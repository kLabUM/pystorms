from pyswmm_lite import environment
from pystorms.utilities import threshold, perf_metrics, load_network
import yaml
import abc
import os

# Absolute path
PATH = os.path.abspath(os.path.dirname(__file__))


# Create a abstract class to force the class definition
class scenario(abc.ABC):
    @abc.abstractmethod
    # Specific to the scenario
    def step(self, actions, log=True):
        pass

    def _logger(self):
        for attribute in self.data_log.keys():
            if attribute != "performance_measure":
                for element in self.data_log[attribute].keys():
                    self.data_log[attribute][element].append(
                        self.env.methods[attribute](element)
                    )

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
        self.config = yaml.load(open(PATH + "/config/gamma.yaml", "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["swmm_input"])

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


class theta(scenario):
    r"""Theta Benchmarking Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------


    Notes
    -----
    Performance is measured as the deviation from the threshold.

    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(PATH + "/config/theta.yaml", "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["swmm_input"])

        self.threshold = 0.5

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {"performance_measure": [], "flow": {}, "flooding": {}}

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions, log=True):
        # Implement the actions and take a step forward
        _, done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the performance
        _perform = 0.0

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                flood = self.env.methods[attribute](ID)
                if flood > 0.0:
                    _perform += 10 ** 6
            if attribute == "flow":
                flow = self.env.methods[attribute](ID)
                _perform = threshold(value=flow, target=self.threshold, scaling=10.0)

        # Record the _performance
        self.data_log["performance_measure"].append(_perform)

        # Terminate the simulation
        if done:
            self.env._terminate()

        return done
