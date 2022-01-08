from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold
import yaml


class beta(scenario):
    r"""beta Scenario

    Stormwater network with tidal downstream fluctuations

    Parameters
    ----------
    config : yaml file

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    ----------
    Objective : Minimize flooding across most sensitive locations

    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(load_config("beta"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Create the environment based on the physical parameters defined in the config file
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing data
        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "flooding": {},
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True):
        # Implement the action and take a step forward
        done = self.env.step(actions)

        # Initialize the time step temporary performance value
        __performance = 0.0  # temporary variable

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

        # cycle through performance targets
        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                __floodrate = self.env.methods[attribute](ID)
                if __floodrate > 0.0:
                    __floodvol = (
                        __floodrate * __timestep * 7.48
                    )  # convert the flooding rate to flooding volume in gallons (1 ft^3 is 7.48 gallons)
                else:
                    __floodvol = 0.0
            __performance += __floodvol

        # Record the _performormance
        self.data_log["performance_measure"].append(__performance)

        # # Log the simulation time
        # self.data_log["simulation_time"].append(__currentsimtime)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
