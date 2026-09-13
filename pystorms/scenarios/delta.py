from pystorms.environment import environment, validate_level
from pystorms.networks import load_network, derived_network_path
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.scenarios.scenario import validate_version
from pystorms.utilities import threshold, exponentialpenalty
import yaml
import swmmio


class delta(scenario):
    r"""Delta Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    version : str
        ``"1"`` is the scenario as published. ``"2"`` tightens the outflow
        threshold, extends the event by three days, scales the rainfall up by
        30 percent, removes the downstream conduit restrictions and the
        uncontrollable subcatchment flooding, and fixes the routing step at
        five seconds.
    level : str
        difficulty level of the instrumentation, see
        :class:`pystorms.environment.environment`

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    -----
    Objective is the following:
    1. To maintain levels of three detention ponds within a range of depths, and
    2. To maintain the network outflow below a threshold.

    Performance is measured as the following:
    1. First, deviation of depth above/below the "desired" depth range for the three detention ponds,
    2. Second, deviation of depth above/below "maximum/minimum" depth ranges for the three detention ponds
        and one other infiltration pond,
    3. Any flooding through the network, and
    4. Any deviation above the threshold of the outflow.

    """

    def __init__(self, version="1", level="1"):
        self.version = validate_version(version, ("1", "2"), "delta")
        self.level = validate_level(level)

        # Network configuration
        with open(load_config("delta"), "r") as fh:
            self.config = yaml.load(fh, yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        self.threshold = 12.0

        self.depth_thresholds = {
            "basin_C": (5.7, 2.21, 3.8, 3.28),
            "basin_S": (6.55, 9.5),
            "basin_N1": (5.92, 2.11, 5.8, 5.2),
            "basin_N2": (6.59, 4.04, 5.04, 4.44),
            "basin_N3": (11.99, 5.28, 5.92, 5.32),
        }

        # Additional penalty definition
        self.max_penalty = 10 ** 6

        if self.version == "2":
            # threshold more stringent
            self.threshold = 0.5

            model = swmmio.Model(self.config["swmm_input"])
            # extend end date
            model.inp.options.loc["END_DATE", "Value"] = "4/27/2016"
            # remove downstream structural flow limitation
            model.inp.xsections.loc["conduit_Eup", "Geom1"] = 5.0
            model.inp.xsections.loc["conduit_Edown", "Geom1"] = 5.0
            # eliminate uncontrollable subcatchment flooding
            model.inp.xsections.loc["conduit_Csc", "Geom1"] = 3.0
            model.inp.xsections.loc["conduit_N1sc", "Geom1"] = 3.0

            # fixed five second routing step, so every run has the same number of steps
            model.inp.options.loc["VARIABLE_STEP", "Value"] = "0.00"
            model.inp.options.loc["ROUTING_STEP", "Value"] = "0:00:05"

            # the initial moisture deficit is given as 4.0 for most subcatchments,
            # which is out of range. Assume the decimal point slipped.
            for subcatch in model.inp.infiltration.index:
                if model.inp.infiltration.loc[subcatch, "IMDmax"] > 1.0:
                    model.inp.infiltration.loc[subcatch, "IMDmax"] = 0.4

            # increase all rainfall intensities
            # the Value column is stored as text, so scale in float and write back as text
            model.inp.timeseries.loc[:, "Value"] = (
                1.3 * model.inp.timeseries["Value"].astype(float)
            ).astype(str)

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
            "depthN": {},
            "flow": {},
            "flooding": {},
            "simulation_time": [],
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True, level=None, version=None):
        # version is accepted for backwards compatibility and ignored
        # Implement the actions and take a step forward
        done = self.env.step(actions, level=level)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the performance
        __performance = 0.0

        for ID, attribute in self.config["performance_targets"]:
            # compute penalty for flooding
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += 10 ** 6
            # compute penalty for flow out of network above threshold
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __performance += threshold(
                    value=__flow, target=self.threshold, scaling=10.0
                )
            # compute penalty for depth at basins above/below predefined ranges
            if attribute == "depthN":
                depth = self.env.methods[attribute](ID)
                temp = self.depth_thresholds[ID]
                if ID == "basin_S":
                    if depth > temp[1]:  # flooding value
                        __performance += 10 ** 6
                    elif depth > temp[0]:
                        __temp = (depth - temp[0]) / (temp[1] - temp[0])
                        __performance += exponentialpenalty(
                            value=__temp, max_penalty=self.max_penalty, scaling=1.0
                        )
                    else:
                        __performance += 0.0
                else:
                    if depth > temp[0] or depth < temp[1]:  # flooding value + fish dead
                        __performance += 10 ** 6
                    elif depth > temp[2]:
                        __temp = (depth - temp[2]) / (temp[0] - temp[2])
                        __performance += exponentialpenalty(
                            value=__temp, max_penalty=self.max_penalty, scaling=1.0
                        )
                    elif depth < temp[3]:
                        __temp = (temp[3] - depth) / (temp[3] - temp[1])
                        __performance += exponentialpenalty(
                            value=__temp, max_penalty=self.max_penalty, scaling=1.0
                        )
                    else:
                        __performance += 0.0

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
