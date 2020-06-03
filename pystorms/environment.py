"""
Environment abstraction for SWMM.
"""
import numpy as np
from pyswmm.simulation import Simulation
import pyswmm.toolkitapi as tkai
from pyswmm.lib import DLL_SELECTION
import ctypes


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
        self.ctrl = ctrl
        if self.ctrl:
            # SWMM Config
            self.config = config

            # SWMM object
            if binary is None:
                self.sim = Simulation(self.config["swmm_input"])
            else:
                DLL_SELECTION.dll_loc = self.config["binary"]
                self.sim = Simulation(self.config["swmm_input"])

            self.sim.start()

        else:
            # SWMM object
            if type(config) == str:
                self.sim = Simulation(config)
                self.sim.start()
            else:
                raise ValueError("Path to inp file not defined")

        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "pollutantN": self._getNodePollutant,
            "pollutantL": self._getLinkPollutant,
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
            raise NameError("State config not defined !")

    def step(self, actions=None):
        r"""
        Implements the control action and forwards
        the simulation by a step.

        Parameters:
        ----------
        actions : list or array
            actions to take as an array (1 x n)

        Returns:
        -------
        new_state : array
            next state
        done : boolean
            event termination indicator
        """
        if (self.ctrl) and (actions is not None):
            # implement the actions
            for asset, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(asset, valve_position)

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

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    def _getNodePollutant(self, ID, pollutant_index):
        """
        Get the pollutant concentration in a node
        :param str ID: Node ID
        :param int NUMPOLLUTANT: Number of pollutants
        :return: Pollutant as list
        """
        index = self.sim._model.getObjectIDIndex(tkai.ObjectType.NODE.value, ID)
        result = ctypes.c_double()
        errorcode = self.sim._model.SWMMlibobj.swmm_getNodePollutant(
            index, pollutant_index, ctypes.byref(result)
        )
        self.sim._model._error_check(errorcode)
        return result.value

    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        self.sim._model.setLinkSetting(ID, valve)

    # ------ Link modifications --------------------------------------------

    def _getLinkPollutant(self, ID, pollutant_index):
        """
        Get the pollutant concentration in a node
        :param str ID: Node ID
        :param int NUMPOLLUTANT: Number of pollutants
        :return: Pollutant as list
        """
        index = self.sim._model.getObjectIDIndex(tkai.ObjectType.NODE.value, ID)
        result = ctypes.c_double()
        errorcode = self.sim._model.SWMMlibobj.swmm_getLinkPollutant(
            index, pollutant_index, ctypes.byref(result)
        )
        self.sim._model._error_check(errorcode)
        return result.value

    def _getLinkDepth(self, link_id):
        return self.sim._model.getLinkResult(link_id, tkai.LinkResults.newDepth.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)
