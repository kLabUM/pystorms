from pystorms.environment import environment, validate_level
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.scenarios.scenario import validate_version

import yaml


class zeta(scenario):
    r"""Zeta Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    version : str
        only ``"1"`` is defined for this scenario
    level : str
        difficulty level of the instrumentation, see
        :class:`pystorms.environment.environment`

    Methods
    ----------
    step:

    Notes
    -----
    Objectives are the following:
    1. Minimization of accumulated CSO volume
    2. Minimization of CSO to the creek more than the river
    3. Maximizing flow to the WWTP
    4. Minimizing roughness of control.

    Performance is measured as the following:
    1. *2 for CSO volume (doubled for flow into the creek)
    2. *(−1) for the flow to the WWTP
    3. *(0.01) for the roughness of the control

    """

    def __init__(self, version="1", level="1"):
        self.version = validate_version(version, ("1",), "zeta")
        self.level = validate_level(level)

        # Network configuration
        with open(load_config("zeta"), "r") as fh:
            self.config = yaml.load(fh, yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Create the environment based on the physical parameters
        self.env = environment(
            self.config, ctrl=True, version=self.version, level=self.level
        )

        self.penalty_weight = {
            "T1": 1,
            "T2": 1,
            "T3": 1,
            "T4": 1,
            "T5": 1,
            "T6": 2,  # creek
            "CSO7": 2,  # creek
            "CSO8": 1,
            "CSO9": 2,  # creek
            "CSO10": 1,
        }

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "flow": {},
            "flooding": {},
            "depthN": {},
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True, level=None, version=None):
        # version is accepted for backwards compatibility and ignored
        # Implement the actions and take a step forward
        done = self.env.step(actions, level=level)

        # Initialize temporary variables
        __performance = 0.0  #

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

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":  # compute penalty for CSO overflow
                __floodrate = self.env.methods[attribute](ID)  # flooding rate
                __volume = __floodrate * __timestep  # flooding volume
                __weight = self.penalty_weight[ID]
            else:  # compute reward for flow to WWTP, and penalty for change in flow from previous timestep due to control (i.e. throttle flow)
                __flowrate = self.env.methods[attribute](ID)
                if ID == "C14":  # conduit connected to "Out_to_WWTP" node
                    __volume = __flowrate * __timestep
                    __weight = -0.1
                else:
                    if len(self.data_log[attribute][ID]) > 1:
                        __prevflowrate = self.data_log[attribute][ID][-2]
                    else:
                        __prevflowrate = __flowrate
                    __throttleflowrate = abs(__prevflowrate - __flowrate)
                    __volume = __throttleflowrate * __timestep
                    __weight = 0.01
            __performance += __volume * __weight

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
