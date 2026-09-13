from pystorms.environment import environment, validate_level
from pystorms.utilities import threshold
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.scenarios.scenario import validate_version
import yaml


class gamma(scenario):
    r"""Gamma Scenario

    Separated stormwater network driven by a 25 year 6 hour event.

    Parameters
    ----------
    version : str
        ``"1"`` is the scenario as published. ``"2"`` lowers the flow
        threshold and drops basins 5 and 9 from the state space, the action
        space and the performance targets.
    level : str
        difficulty level of the instrumentation, see
        :class:`pystorms.environment.environment`

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    -----
    Objective : Route flows though the network such that they are below a threshold.
    """

    def __init__(self, version="1", level="1"):
        self.version = validate_version(version, ("1", "2"), "gamma")
        self.level = validate_level(level)

        # Network configuration
        with open(load_config("gamma"), "r") as fh:
            self.config = yaml.load(fh, yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Common threhold for the network, can be done independently
        self._performormance_threshold = 4.0

        if self.version == "2":
            self._performormance_threshold = 3.0

            # remove basins 5 and 9 from the scenario
            def excluded(ID):
                return "5" in ID or "9" in ID

            self.config["states"] = [
                state for state in self.config["states"] if not excluded(state[0])
            ]
            self.config["action_space"] = [
                action for action in self.config["action_space"] if not excluded(action)
            ]
            self.config["performance_targets"] = [
                target
                for target in self.config["performance_targets"]
                if not excluded(target[0])
            ]

        # Create the environment based on the physical parameters
        self.env = environment(
            self.config, ctrl=True, version=self.version, level=self.level
        )

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "flow": {},
            "flooding": {},
            "depthN": {},
            "simulation_time": [],
        }

        # Data logger for storing _performormance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True, level=None, version=None):
        # version is accepted for backwards compatibility and ignored
        # Implement the actions and take a step forward
        done = self.env.step(actions, level=level)

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
