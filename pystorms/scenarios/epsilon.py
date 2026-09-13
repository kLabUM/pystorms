from pystorms.environment import environment, validate_level
from pystorms.networks import load_network, derived_network_path
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.scenarios.scenario import validate_version
from pystorms.utilities import threshold
import yaml
import swmmio


class epsilon(scenario):
    r"""Epsilon Scenario

    Stormwater network with control structures in pipes

    Parameters
    ----------
    version : str
        ``"1"`` is the scenario as published. ``"2"`` lowers the TSS loading
        threshold to 70 percent of the original, extends the event to the
        middle of February and scales the rainfall up by 10 percent.
    level : str
        difficulty level of the instrumentation, see
        :class:`pystorms.environment.environment`

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    ----------
    Objective : Route flows to maintain constant outflow at the outlet

    """

    def __init__(self, version="1", level="1"):
        self.version = validate_version(version, ("1", "2"), "epsilon")
        self.level = validate_level(level)

        # Network configuration
        with open(load_config("epsilon"), "r") as fh:
            self.config = yaml.load(fh, yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Dry weather TSS loading, measured at the outlet of the network
        self._performormance_threshold = 1.075  # Kg/sec

        if self.version == "2":
            # make the threshold more stringent
            self._performormance_threshold = self._performormance_threshold * (7.0 / 10.0)

            model = swmmio.Model(self.config["swmm_input"])
            # extend the event to the middle of February
            model.inp.options.loc["END_DATE", "Value"] = "02/15/2017"

            # increase the rainfall intensity by 10% throughout
            # the Value column is stored as text, so scale in float and write back as text
            model.inp.timeseries.loc[:, "Value"] = (
                1.1 * model.inp.timeseries["Value"].astype(float)
            ).astype(str)

            derived = derived_network_path(self.config["swmm_input"], "v2")
            model.inp.save(derived)
            self.config["swmm_input"] = derived

        # Create the env based on the config file
        self.env = environment(
            self.config, ctrl=True, version=self.version, level=self.level
        )

        # Create an object for storing data
        self.data_log = {
            "performance_measure": [],
            "loading": {},
            "pollutantL": {},
            "flow": {},
            "flooding": {},
            "simulation_time": [],
        }

        # Data logger for storing _performormance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True, level=None, version=None):
        # version is accepted for backwards compatibility and ignored
        # Implement the action and take a step forward
        done = self.env.step(actions, level=level)

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
                    self.env.methods["pollutantL"](ID, "TSS")
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
                    self.env.methods["pollutantL"](ID, "TSS")
                    * self.env.methods["flow"](ID)
                    * 28.3168
                    / (10 ** 6)
                )
                self.data_log[attribute][ID].append(pollutantLoading)
            else:
                self.data_log[attribute][ID].append(self.env.methods[attribute](ID))
