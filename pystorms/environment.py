"""
Environment abstraction for SWMM.
"""
import warnings

import numpy as np
import pyswmm.toolkitapi as tkai
from pyswmm.simulation import Simulation

from pystorms.networks import run_file_paths

try:
    from pyswmm.warnings import SimulationContextWarning
except ImportError:
    # pyswmm 1.x does not warn about the context manager
    SimulationContextWarning = None


LEVELS = ("1", "2", "3")

# Instrumentation faults drawn for the difficulty levels. Lengths are metres
# and are converted for networks that run in US units.
#
#   drift_rate         sensor drift per day
#   drift_chance       probability that a given sensor drifts at all
#   drift_spread       range of the multiplier applied to the drift rate
#   bias               range of the multiplicative calibration bias
#   noise              reading noise, as a multiple of NOISE_SIGMA
#   actuator_threshold an actuator sticks once if rand() exceeds this
#   actuator_duration  range of the stuck duration, as a fraction of the event
#   sensor_threshold   a sensor drops out once if rand() exceeds this
#   sensor_duration    range of the dropout duration, as a fraction of the event
FAULTS = {
    "2": dict(
        drift_rate=0.03 / 100,
        drift_chance=0.15,
        drift_spread=(0.5, 1.5),
        bias=(0.99, 1.01),
        noise=1.0,
        actuator_threshold=0.4,
        actuator_duration=(0.1, 0.3),
        sensor_threshold=None,
        sensor_duration=None,
    ),
    "3": dict(
        drift_rate=1.0 / 100,
        drift_chance=0.50,
        drift_spread=(1.0, 2.0),
        bias=(0.9, 1.1),
        noise=6.0,
        actuator_threshold=0.2,
        actuator_duration=(0.2, 0.5),
        sensor_threshold=0.2,
        sensor_duration=(0.05, 0.2),
    ),
}

# standard deviation of the level 2 reading noise, in metres
NOISE_SIGMA = 0.025

# metres to feet
FEET_PER_METRE = 3.28084


def validate_level(level):
    r"""Normalise *level* to a string and check it is a defined level.

    Parameters
    ----------
    level : str or int
        requested difficulty level

    Returns
    -------
    level : str
    """
    level = str(level)
    if level not in LEVELS:
        raise ValueError(
            "level must be one of {0}; got {1!r}".format(", ".join(LEVELS), level)
        )
    return level


def _mark_simulation_running(running):
    r"""Tell pyswmm whether a simulation currently holds the engine.

    pyswmm >= 2.0 allows a single Simulation per interpreter and tracks it on a
    module level flag. pystorms drives the engine through the lower level
    toolkit calls, so it has to keep that flag in step itself. Without this,
    building a second scenario in one process raises MultiSimulationError even
    though the first one has been terminated.
    """
    try:
        from pyswmm.simulation import _sim_state_instance
    except ImportError:
        # pyswmm 1.x tracks no such state
        return

    _sim_state_instance.sim_is_instantiated = running


class environment:
    r"""Environment for controlling the swmm simulation

    This class acts as an interface between swmm's simulation
    engine and computational components. This class's methods are defined
    as getters and setters for generic stormwater attributes. So that, if need be, this
    class can be updated with a different simulation engine, keeping rest of the
    workflow stable.

    Parameters
    ----------
    config : dict or str
        with ``ctrl=True``, a dict holding ``swmm_input`` (path to the input
        file) and the ``states``, ``action_space`` and ``performance_targets``
        of the scenario; with ``ctrl=False``, the path to a swmm input file
    ctrl : bool
        whether a state and action space are defined. Querying the state and
        setting control actions require ``ctrl=True``
    binary : str, optional
        ignored. Kept for backwards compatibility; the engine always comes
        from the installed pyswmm
    version : str
        scenario version the network was built for. Stored for reference, the
        network itself is selected by the scenario
    level : str
        difficulty level. ``"1"`` reports the true state. ``"2"`` and ``"3"``
        draw sensor drift, calibration bias, reading noise and stuck actuators
        when the environment is built; ``"3"`` also draws sensor dropouts

    Attributes
    ----------
    level : str
        the level the environment was built with
    drift_rates, bias : ndarray
        per sensor drift rate and calibration bias, for levels 2 and 3
    actuator_schedule, sensor_schedule : dict or None
        ``{ID: [(stuck_time, fix_time), ...]}`` for every asset that faults
        during the event, or None when nothing faults

    Methods
    ----------
    step
        implements the actions and steps the simulation forward
    initial_state
        returns the initial state in the stormwater network
    terminate
        closes the swmm simulation
    reset
        closes the swmm simulaton and start a new one with the predefined config file.
    """

    def __init__(self, config, ctrl=True, binary=None, version="1", level="1"):
        self.version = str(version)
        self.level = validate_level(level)

        # control expects users to define the state and action space
        # this is required for querying state and setting control actions
        self.ctrl = ctrl
        if self.ctrl:
            # read config from dictionary;
            # example configs are the yaml files in pystorms/config
            self.config = config

            # swmm writes a report and a binary output file for every run.
            # Keep them out of the installed package.
            report, output = run_file_paths(self.config["swmm_input"])
            self.sim = Simulation(self.config["swmm_input"], report, output)
        else:
            # load the swmm object based on the inp file path
            if isinstance(config, str):
                self.sim = Simulation(config)
            else:
                raise ValueError(f"Given input file path is not valid {config}")

        # start the swmm simulation
        # this reads the inp file and initializes elements in the model
        with warnings.catch_warnings():
            if SimulationContextWarning is not None:
                # pystorms manages the simulation lifetime itself
                warnings.simplefilter("ignore", SimulationContextWarning)
            self.sim.start()
        self._running = True

        # for levels 2 and 3, schedule random faults in sensors and actuators
        self.drift_rates = None
        self.bias = None
        self.actuator_schedule = None
        self.sensor_schedule = None
        self._draw_faults()

        # map class methods to individual class function calls
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "pollutantN": self._getNodePollutant,
            "pollutantL": self._getLinkPollutant,
            "simulation_time": self._getCurrentSimulationDateTime,
        }

    # ------ Difficulty levels ---------------------------------------------
    def _length_scale(self):
        r"""Factor converting metres into the network's length unit."""
        return FEET_PER_METRE if self.sim.system_units == "US" else 1.0

    def _draw_faults(self):
        r"""Draw the fault schedule for levels 2 and 3 from numpy's global RNG.

        The order of the draws is part of the interface: seeding numpy before
        building a scenario reproduces the same faults.
        """
        if self.level == "1":
            return

        faults = FAULTS[self.level]
        n_states = len(self.config["states"])

        # drift: a subset of the sensors drift, all at the same rate
        drifting = np.random.choice(
            [0, 1],
            size=n_states,
            p=[1 - faults["drift_chance"], faults["drift_chance"]],
        )
        rate = faults["drift_rate"] * self._length_scale()
        self.drift_rates = drifting * np.random.uniform(*faults["drift_spread"]) * rate

        # calibration bias, multiplicative, one per sensor
        self.bias = np.random.uniform(*faults["bias"], size=n_states)

        # sensor dropouts (level 3 only), then stuck actuators
        if faults["sensor_threshold"] is not None:
            sensor_ids = [entry[0] for entry in self.config["states"]]
            self.sensor_schedule = self._draw_schedule(
                sensor_ids, faults["sensor_threshold"], faults["sensor_duration"]
            )

        self.actuator_schedule = self._draw_schedule(
            self.config["action_space"],
            faults["actuator_threshold"],
            faults["actuator_duration"],
        )

    def _draw_schedule(self, ids, threshold, duration):
        r"""Draw at most one fault window per entry of *ids*.

        Returns ``{ID: [(stuck_time, fix_time), ...]}`` or None when no fault
        was drawn. An ID that appears twice in *ids* can draw two windows.
        """
        start = self.sim.start_time
        span = self.sim.end_time - start

        schedule = {}
        for ID in ids:
            if np.random.rand() > threshold:
                fault_duration = np.random.uniform(*duration)
                fault_time = np.random.uniform(0.0, 1.0 - fault_duration)
                stuck = start + span * fault_time
                fix = stuck + span * fault_duration
                schedule.setdefault(ID, []).append((stuck, fix))

        return schedule if schedule else None

    @staticmethod
    def _is_stuck(schedule, ID, now):
        r"""Whether *ID* is inside one of its fault windows at *now*."""
        windows = schedule.get(ID) if schedule else None
        if not windows:
            return False

        started = [stuck for stuck, _ in windows if stuck < now]
        if not started:
            return False

        latest = max(started)
        fixes = [fix for _, fix in windows if fix > latest]
        return min(fixes) > now

    def _resolve_level(self, level):
        r"""Pick the level to apply, defaulting to the one the environment was built with."""
        if level is None:
            return self.level

        level = validate_level(level)
        if level not in ("1", self.level):
            raise ValueError(
                "this scenario was built with level {0}, so its readings cannot "
                "be degraded to level {1}. Pass level={1!r} when building the "
                "scenario instead.".format(self.level, level)
            )
        return level

    # ------ State and actions ---------------------------------------------
    def _read(self, entry):
        r"""Read one ``(ID, attribute[, pollutant])`` entry of the state space."""
        ID, attribute = entry[0], entry[1]
        if attribute in ("pollutantN", "pollutantL"):
            return self.methods[attribute](ID, entry[2])
        return self.methods[attribute](ID)

    def _state(self, level=None):
        r"""
        Query the stormwater network states based on the config file.

        Parameters
        ----------
        level : str, optional
            difficulty level to apply to the readings. Defaults to the level
            the environment was built with. ``"1"`` always returns the true
            state.
        """
        if not self.ctrl:
            print("State config not defined! \n ctrl is defined as False")
            return np.array([])

        level = self._resolve_level(level)
        state = np.asarray([self._read(entry) for entry in self.config["states"]], dtype=float)

        if level == "1":
            return state

        faults = FAULTS[level]
        now = self._getCurrentSimulationDateTime()

        # noise, drift and bias. Magnitudes are lengths and are applied to every
        # state, whatever its quantity.
        elapsed_days = (now - self.sim.start_time).total_seconds() / 86400.0
        drift = self.drift_rates * elapsed_days
        sigma = faults["noise"] * NOISE_SIGMA * self._length_scale()
        state = self.bias * state + drift + np.random.normal(0.0, sigma, state.shape)

        # a sensor that has dropped out reports zero
        if self.sensor_schedule is not None:
            for i, entry in enumerate(self.config["states"]):
                if self._is_stuck(self.sensor_schedule, entry[0], now):
                    state[i] = 0.0

        return state

    def step(self, actions=None, level=None):
        r"""
        Implements the control action and forwards
        the simulation by a step.

        Parameters:
        ----------
        actions : list, array or dict
            valve settings, either in the order of the action space or keyed
            by asset ID
        level : str, optional
            difficulty level to apply. Defaults to the level the environment
            was built with. At levels 2 and 3 a stuck actuator ignores the
            command; ``"1"`` applies every command.

        Returns:
        -------
        done : boolean
            event termination indicator
        """
        level = self._resolve_level(level)

        if self.ctrl and actions is not None:
            if isinstance(actions, dict):
                pairs = actions.items()
            elif isinstance(actions, (list, np.ndarray)):
                pairs = zip(self.config["action_space"], actions)
            else:
                raise ValueError(
                    "actions must be dict or list or np.ndarray \n got{}".format(
                        type(actions)
                    )
                )

            now = self._getCurrentSimulationDateTime() if level != "1" else None
            for asset, valve_position in pairs:
                if level != "1" and self._is_stuck(self.actuator_schedule, asset, now):
                    # the actuator is stuck, the command is lost
                    continue
                self._setValvePosition(asset, valve_position)

        # take the step !
        time = self.sim._model.swmm_step()
        done = time <= 0
        return done

    def reset(self):
        r"""
        Resets the simulation and returns the initial state

        The fault schedule drawn when the environment was built is kept.

        Returns
        -------
        initial_state : array
            initial state in the network

        """
        self.terminate()

        # Start the next simulation
        self.sim._model.swmm_open()
        self.sim._model.swmm_start()
        _mark_simulation_running(True)
        self._running = True

        # get the state
        state = self._state()
        return state

    def terminate(self):
        r"""
        Terminates the simulation. Safe to call more than once.
        """
        if not self._running:
            return

        self.sim._model.swmm_end()
        self.sim._model.swmm_close()
        self._running = False

        # swmm_close() releases the engine but leaves pyswmm's own bookkeeping
        # untouched, which would block every later scenario in this process.
        _mark_simulation_running(False)

    def initial_state(self):
        r"""
        Get the initial state in the stormwater network

        Returns
        -------
        initial_state : array
            initial state in the network
        """
        return self._state()

    # ------ Node Parameters  ----------------------------------------------
    def _getNodeDepth(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newDepth.value)

    def _getNodeFlooding(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.overflow.value)

    def _getNodeLosses(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.losses.value)

    def _getNodeVolume(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newVolume.value)

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    def _getNodePollutant(self, ID, pollutant_name=None):
        pollut_quantity = self.sim._model.getNodePollut(ID, tkai.NodePollut.nodeQual)
        pollut_id = self.sim._model.getObjectIDList(tkai.ObjectType.POLLUT.value)
        pollutants = {pollut_id[i]: pollut_quantity[i] for i in range(0, len(pollut_id))}
        if pollutant_name is None:
            return pollutants
        else:
            return pollutants[pollutant_name]

    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        self.sim._model.setLinkSetting(ID, valve)

    # ------ Link modifications --------------------------------------------

    def _getLinkPollutant(self, ID, pollutant_name=None):
        pollut_quantity = self.sim._model.getLinkPollut(ID, tkai.LinkPollut.linkQual)
        pollut_id = self.sim._model.getObjectIDList(tkai.ObjectType.POLLUT.value)
        pollutants = {pollut_id[i]: pollut_quantity[i] for i in range(0, len(pollut_id))}
        if pollutant_name is None:
            return pollutants
        else:
            return pollutants[pollutant_name]

    def _getLinkDepth(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newDepth.value)

    def _getLinkVolume(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newVolume.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)

    def getCurrentSimulationDateTime(self):
        r"""
        Get the current time of the simulation for this timestep.

        Returns
        -------
        :return: current simulation datetime
        :rtype: datetime
        """
        return self._getCurrentSimulationDateTime()

    def getInitialSimulationDateTime(self):
        r"""
        Get the initial datetime of the simulation.

        Returns
        -------
        :return: initial simulation datetime
        :rtype: datetime
        """
        return self._getInitialSimulationDateTime()

    # ------- Obtain the current simulation time to compute the timestep ----------
    def _getCurrentSimulationDateTime(self):
        r"""
        Get the current time of the simulation for this timestep.
        Can be used to compute the current timestep.

        Returns
        -------
        :return: current simulation datetime
        :rtype: datetime
        """
        return self.sim._model.getCurrentSimulationTime()

    def _getInitialSimulationDateTime(self):
        r"""
        Get the initial datetime of the simulation.
        Can be used to compute the initial timestep.

        Returns
        -------
        :return: initial simulation datetime
        :rtype: datetime
        """
        return self.sim._model.getSimulationDateTime(
            tkai.SimulationTime.StartDateTime.value
        )
