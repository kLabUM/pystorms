from pystorms.environment import environment, validate_level
from pystorms.networks import load_network, derived_network_path
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.scenarios.scenario import validate_version
from pystorms.utilities import threshold
import yaml
import swmmio


class theta(scenario):
    r"""Theta Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    version : str
        ``"1"`` is the scenario as published. ``"2"`` halves the flow
        threshold, halves the maximum depth of the first basin and ends the
        event a day and a half earlier.
    level : str
        difficulty level of the instrumentation, see
        :class:`pystorms.environment.environment`

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    -----
    Performance is measured as the deviation from the threshold.

    """

    def __init__(self, version="1", level="1"):
        self.version = validate_version(version, ("1", "2"), "theta")
        self.level = validate_level(level)

        # Network configuration
        with open(load_config("theta"), "r") as fh:
            self.config = yaml.load(fh, yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        self.threshold = 0.5

        if self.version == "2":
            # make the threshold more stringent
            self.threshold = self.threshold * (1.0 / 2.0)

            # shrink the first basin and end the event sooner. The shipped
            # network is left alone; the rewritten copy goes to the cache
            model = swmmio.Model(self.config["swmm_input"])
            basin = self.config["states"][0][0]
            model.inp.storage.loc[basin, "MaxD"] = (
                model.inp.storage.loc[basin, "MaxD"] / 2.0
            )
            model.inp.options.loc["END_DATE", "Value"] = "2/26/2018"
            model.inp.options.loc["END_TIME", "Value"] = "12:00:00"

            derived = derived_network_path(self.config["swmm_input"], "v2")
            model.inp.save(derived)
            self.config["swmm_input"] = derived

        # Create the environment based on the physical parameters
        self.env = environment(
            self.config, ctrl=True, version=self.version, level=self.level
        )

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "flow": {},
            "flooding": {},
            "simulation_time": [],
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True, level=None, version=None):
        # version is accepted for backwards compatibility and ignored; the
        # network was chosen when the scenario was built
        # Implement the actions and take a step forward
        done = self.env.step(actions, level=level)

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
            self.env.terminate()

        return done
