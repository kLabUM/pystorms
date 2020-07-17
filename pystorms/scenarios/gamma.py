from pystorms.environment import environment
from pystorms.utilities import threshold
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
import yaml


class gamma(scenario):
    r"""Gamma Scenario

    Separated stormwater network driven by a 25 year 6 hour event.

    Parameters
    ----------
    config : yaml file

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    -----
    Objective : Route flows though the network such that they are below a threshold.
    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(load_config("gamma"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

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

    def step(self, actions=None, log=True):
        # Implement the actions and take a step forward
        done = self.env.step(actions)

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
            self.env.terminate()

        return done
