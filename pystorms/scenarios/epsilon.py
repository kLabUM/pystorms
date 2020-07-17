from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold
from pystorms.binaries import load_binary
import yaml


class epsilon(scenario):
    r"""Epsilon Scenario

    Stormwater network with control structures in pipes

    Parameters
    ----------
    config : yaml file

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    ----------
    Objective : Route flows to maintain constant outflow at the outlet

    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(load_config("epsilon"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])
        self.config["binary"] = load_binary(self.config["binary"])

        # Dry weather TSS loading, measured at the outlet of the network
        self._performormance_threshold = 1.075  # Kg/sec

        # Create the env based on the config file
        self.env = environment(self.config, ctrl=True, binary=self.config["binary"])

        # Create an object for storing data
        self.data_log = {
            "performance_measure": [],
            "loading": {},
            "pollutantL": {},
            "flow": {},
            "flooding": {},
        }

        # Data logger for storing _performormance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True):
        # Implement the action and take a step forward
        done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the _performormance
        __performance = 0.0  # temporary variable

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                flood = self.env.methods[attribute](ID)
                if flood > 0.0:
                    __performance += 10 ** 9
            elif attribute == "loading":
                pollutantLoading = (
                    self.env.methods["pollutantL"](ID, 0)
                    * self.env.methods["flow"](ID)
                    * 28.3168
                    / (10 ** 6)
                )
                __performance += threshold(pollutantLoading, self._performormance_threshold)

        # Record the _performormance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done

    def _logger(self):
        # Log all the _performormance values;
        # additionally, other components can be added here
        for ID, attribute in self.config["performance_targets"]:
            if attribute == "loading":
                pollutantLoading = (
                    self.env.methods["pollutantL"](ID, 0)
                    * self.env.methods["flow"](ID)
                    * 28.3168
                    / (10 ** 6)
                )
                self.data_log[attribute][ID].append(pollutantLoading)
            else:
                self.data_log[attribute][ID].append(self.env.methods[attribute](ID))
