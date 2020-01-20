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

    Separated stormwater network driven by a 25 year 6 hour event.

    Parameters
    ----------
    config : yaml file

    Methods
    ----------
    step :

    Notes
    -----
    Objective : Route flows though the network such that they are below a threshold.
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
        self.data_log = {
            "performance_measure": [],
            "flow": {},
            "flooding": {},
            "depthN": {},
        }

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
        __performance = 0.0  # temp variable

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += 10 ** 6
            elif attribute == "flow":
                __target = self._performormance_threshold
                __performance += threshold(
                    self.env.methods[attribute](ID), __target, scaling=1.0
                )
            # Check for water in the last timestep
            elif done and attribute == "depthN":
                __depth = self.env.methods[attribute](ID)
                if __depth > 0.10:
                    __performance += 7 * 10 ** 5

        # Record the _performormance
        self.data_log["performance_measure"].append(__performance)

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
        __performance = 0.0

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += 10 ** 6
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __performance = threshold(
                    value=__flow, target=self.threshold, scaling=10.0
                )

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env._terminate()

        return done
