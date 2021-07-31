from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold, exponentialpenalty
import yaml


class alpha(scenario):
    r"""Alpha Scenario

    Separated stormwater network driven by a 2-year storm event.
    Scenario is adapted from SWMM Apps tutorial Example 8.

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
    1. To minimize the combined sewer overflow of the system
    2. To avoid flooding

    Performance is measured as the following:
    1. First, any combined sewer overflow,
    2. Second, any flooding in the network

    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(load_config("alpha"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "flow": {},
            "volume": {},
            "flooding": {},
            "simulation_time": [],
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True):
        # Implement the actions and take a step forward
        done = self.env.step(actions)

        # Temp variables
        __performance = 0.0

        # Determine current timestep in simulation by
        # obtaining the differeence between the current time
        # and previous time, and converting to seconds
        __currentsimtime = self.env._getCurrentSimulationDateTime()

        if len(self.data_log["simulation_time"]) > 1:
            __prevsimtime = self.data_log["simulation_time"][-1]
        else:
            __prevsimtime = self.env._getInitialSimulationDateTime()

        __timestep = (__currentsimtime - __prevsimtime).total_seconds()

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the performance
        for ID, attribute in self.config["performance_targets"]:
            # compute penalty for flooding
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += __flood * (10 ** 6)
            # compute penalty for CSO overflow
            # TODO - decide if we want to have a penalty for negative flow
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __volume = __timestep * __flow
                __performance += threshold(value=__volume, target=0.0, scaling=1.0)

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
