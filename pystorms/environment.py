"""
Environment abstraction for SWMM.
"""
import numpy as np
import pyswmm.toolkitapi as tkai
from pyswmm.simulation import Simulation


class environment:
    r"""Environment for controlling the swmm simulation

    This class acts as an interface between swmm's simulation
    engine and computational components. This class's methods are defined
    as getters and setters for generic stormwater attributes. So that, if need be, this
    class can be updated with a different simulation engine, keeping rest of the
    workflow stable.


    Attributes
    ----------
    config : dict
        dictionary with swmm_ipunt and, action and state space `(ID, attribute)`
    ctrl : boolean
        if true, config has to be a dict, else config needs to be the path to the input file
    binary: str
        path to swmm binary; this enables users determine which version of swmm to use

    Methods
    ----------
    step
        steps the simulation forward by a time step and returns the new state
    initial_state
        returns the initial state in the stormwater network
    terminate
        closes the swmm simulation
    reset
        closes the swmm simulaton and start a new one with the predefined config file.
    """

    def __init__(self, config, ctrl=True, binary=None):

        # control expects users to define the state and action space
        # this is required for querying state and setting control actions
        self.ctrl = ctrl
        if self.ctrl:
            # read config from dictionary;
            # example config can be found in documentation
            # TODO: Add link to config documentation
            self.config = config

            # load the swmm object
            self.sim = Simulation(self.config["swmm_input"])
        else:
            # load the swmm objection based on the inp file path
            if type(config) == str:
                self.sim = Simulation(INPPATH=config)
            else:
                raise ValueError(f"Given input file path is not valid {config}")

        # start the swmm simulation
        # this reads the inp file and initializes elements in the model
        self.sim.start()

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

    def _state(self):
        r"""
        Query the stormwater network states based on the config file.
        """
        if self.ctrl:
            state = []
            for _temp in self.config["states"]:
                ID = _temp[0]
                attribute = _temp[1]

                if attribute == "pollutantN" or attribute == "pollutantL":
                    pollutant_index = _temp[2]
                    state.append(self.methods[attribute](ID, pollutant_index))
                else:
                    state.append(self.methods[attribute](ID))

            state = np.asarray(state)
            return state
        else:
            print("State config not defined! \n ctrl is defined as False")
            return np.array([])

    def step(self, actions=None):
        r"""
        Implements the control action and forwards
        the simulation by a step.

        Parameters:
        ----------
        actions : list or array of dict
            actions to take as an array (1 x n)

        Returns:
        -------
        new_state : array
            next state
        done : boolean
            event termination indicator
        """

        if (self.ctrl) and (actions is not None):
            # implement the actions based on type of argument passed
            # if actions are an array or a list
            if type(actions) == list or type(actions) == np.ndarray:
                for asset, valve_position in zip(self.config["action_space"], actions):
                    self._setValvePosition(asset, valve_position)
            elif type(actions) == dict:
                for valve_position, asset in enumerate(actions):
                    self._setValvePosition(asset, valve_position)
            else:
                raise ValueError(
                    "actions must be dict or list or np.ndarray \n got{}".format(
                        type(actions)
                    )
                )

        # take the step !
        time = self.sim._model.swmm_step()
        done = False if time > 0 else True
        return done

    def reset(self):
        r"""
        Resets the simulation and returns the initial state

        Returns
        -------
        initial_state : array
            initial state in the network

        """
        self.terminate()

        # Start the next simulation
        self.sim._model.swmm_open()
        self.sim._model.swmm_start()

        # get the state
        state = self._state()
        return state

    def terminate(self):
        r"""
        Terminates the simulation
        """
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()

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
        pollut_quantity = self.sim._model.getNodePollut(ID, tkai.LinkPollut.linkQual)
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
