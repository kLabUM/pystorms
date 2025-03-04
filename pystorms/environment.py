"""
Environment abstraction for SWMM.
"""
import numpy as np
import pyswmm.toolkitapi as tkai
from pyswmm.simulation import Simulation
import pandas as pd

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

    def __init__(self, config, ctrl=True, binary=None,version="1",level="1"):
        

        #print("version = ", version)
        #print("level = ", level)
        self.version = version
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
        
        self.actuator_schedule = None
        self.sensor_schedule = None
        
        # for levels 2 and 3, schedule random faults in sensors and actuators
        if level == "2":
            # define drift rate
            if self.sim.system_units == "SI": # metric
                base_drift_rate = 0.03/100 # 0.03 centimeters / day (in meters)
            elif self.sim.system_units == "US": # imperial
                base_drift_rate = (0.03/100) * 3.28084 # 0.03 centimeters expressed in ft / day
            chance_of_drift = 0.15 # 15% chance of drift for any given sensor
            # define drifts as an array of length len(states) with each entry having chance_of_drift likelihood of one, and otherwise zero
            drift_rates = np.random.choice([0, 1], size=len(self.config['states']), p=[1-chance_of_drift, chance_of_drift])
            # multiply drifts by np.random.uniform(0.5, 1.5)
            # to create a drift rate for each sensor
            self.drift_rates = drift_rates * np.random.uniform(0.5, 1.5) * base_drift_rate
            
            #print("drifts\n", self.drift_rates)
            # define bias
            # an array of length len(state) with entries sampled from random uniform between 0.99 and 1.01
            self.bias = np.random.uniform(0.99, 1.01, size=len(self.config['states']))
            #print("bias\n", self.bias)

            # create an actuator schedule with columns the action space.
            # rows will be event times
            # entries will be events. for now, just "stuck" and "fix"
            actuator_schedule = pd.DataFrame(columns = self.config['action_space'])
            for actuator in actuator_schedule.columns:
                if np.random.rand() > 0.4:
                    # a fault will occur
                    fault_duration = np.random.uniform(0.1, 0.3) # 10 to 30% of duration
                    fault_time = np.random.uniform(0.0, 1.0-fault_duration)
                    fault_datetime = self.sim.start_time + (self.sim.end_time - self.sim.start_time) * fault_time
                    actuator_schedule.loc[fault_datetime, actuator] = "stuck"
                    fix_datetime = fault_datetime + (self.sim.end_time - self.sim.start_time) * fault_duration
                    actuator_schedule.loc[fix_datetime, actuator] = "fix"
            
            # ensure there are no duplicate columns in actuator_schedule
            actuator_schedule = actuator_schedule.loc[:,~actuator_schedule.columns.duplicated()]
            self.actuator_schedule = actuator_schedule
            
            # if actuator_schedule is empty, make it None
            if actuator_schedule.empty:
                self.actuator_schedule = None
            #print(self.actuator_schedule)
        if level == "3":
            # define drift rate
            if self.sim.system_units == "SI": # metric
                base_drift_rate = 1/100 # 1 centimeters / day (in meters)
            elif self.sim.system_units == "US": # imperial
                base_drift_rate = (1/100) * 3.28084 # 0.03 centimeters expressed in ft / day
            chance_of_drift = 0.50 # 40% chance of drift for any given sensor
            # define drifts as an array of length len(states) with each entry having chance_of_drift likelihood of one, and otherwise zero
            drift_rates = np.random.choice([0, 1], size=len(self.config['states']), p=[1-chance_of_drift, chance_of_drift])
            # multiply drifts by np.random.uniform(0.5, 1.5)
            # to create a drift rate for each sensor
            self.drift_rates = drift_rates * np.random.uniform(1.0, 2.0) * base_drift_rate
            
            #print("drifts\n", self.drift_rates)
            # define bias
            # an array of length len(state) with entries sampled from random uniform between 0.99 and 1.01
            self.bias = np.random.uniform(0.9, 1.1, size=len(self.config['states']))            


            sensor_ids = [self.config['states'][i][0] for i in range(len(self.config['states']))]
            sensor_schedule = pd.DataFrame(columns = sensor_ids)
            for sensor in sensor_schedule.columns:
                if np.random.rand() > 0.2:
                    # a fault will occur
                    fault_duration = np.random.uniform(0.05, 0.2)
                    fault_time = np.random.uniform(0.0, 1.0-fault_duration)
                    fault_datetime = self.sim.start_time + (self.sim.end_time - self.sim.start_time) * fault_time
                    sensor_schedule.loc[fault_datetime, sensor] = "stuck"
                    fix_datetime = fault_datetime + (self.sim.end_time - self.sim.start_time) * fault_duration
                    sensor_schedule.loc[fix_datetime, sensor] = "fix"
            sensor_schedule = sensor_schedule.loc[:,~sensor_schedule.columns.duplicated()]
            self.sensor_schedule = sensor_schedule
            if sensor_schedule.empty:
                self.sensor_schedule = None 
            #print(self.sensor_schedule)


            actuator_schedule = pd.DataFrame(columns = self.config['action_space'])
            for actuator in actuator_schedule.columns:
                if np.random.rand() > 0.2:
                    # a fault will occur
                    fault_duration = np.random.uniform(0.2, 0.5) # 
                    fault_time = np.random.uniform(0.0, 1.0-fault_duration)
                    fault_datetime = self.sim.start_time + (self.sim.end_time - self.sim.start_time) * fault_time
                    actuator_schedule.loc[fault_datetime, actuator] = "stuck"
                    fix_datetime = fault_datetime + (self.sim.end_time - self.sim.start_time) * fault_duration
                    actuator_schedule.loc[fix_datetime, actuator] = "fix"
            actuator_schedule = actuator_schedule.loc[:,~actuator_schedule.columns.duplicated()]
            self.actuator_schedule = actuator_schedule
            if actuator_schedule.empty:
                self.actuator_schedule = None
            #print(self.actuator_schedule)


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

    def _state(self,level="1"):
        r"""
        Query the stormwater network states based on the config file.
        """
        if self.ctrl:
            state = []
            for _temp in self.config["states"]:
                ID = _temp[0]
                attribute = _temp[1]

                if level == "3" and self.sensor_schedule is not None and self.sensor_schedule[ID] is not None:
                    # if the current time has a "stuck" before it and a "fix" after it for the column "ID"
                    # assign the most recent value in the data_log and continue
                    # check if self.current_time has "stuck" before and "fix" after in sensor_schedule[ID]
                    # find the index value of the most recent "stuck" before self.current_time
                    stuck_times = self.sensor_schedule[ID].index[self.sensor_schedule[ID] == "stuck"]
                    fix_times = self.sensor_schedule[ID].index[self.sensor_schedule[ID] == "fix"]
                    if len(stuck_times) > 0:
                        most_recent_stuck_time = stuck_times[stuck_times < self.sim.current_time].max()
                        most_recent_fix_time = fix_times[fix_times > most_recent_stuck_time].min()
                        if most_recent_fix_time > self.sim.current_time:
                            # the sensor is stuck
                            #print("sensor ", ID, " zeroed out")
                            state.append(0.0)
                        else: 
                            # the sensor is not stuck
                            if attribute == "pollutantN" or attribute == "pollutantL":
                                pollutant_index = _temp[2]
                                state.append(self.methods[attribute](ID, pollutant_index))
                            else:
                                state.append(self.methods[attribute](ID))   
                    else: 
                        # the sensor is not stuck
                        if attribute == "pollutantN" or attribute == "pollutantL":
                            pollutant_index = _temp[2]
                            state.append(self.methods[attribute](ID, pollutant_index))
                        else:
                            state.append(self.methods[attribute](ID)) 
                else:
                    if attribute == "pollutantN" or attribute == "pollutantL":
                        pollutant_index = _temp[2]
                        state.append(self.methods[attribute](ID, pollutant_index))
                    else:
                        state.append(self.methods[attribute](ID))

            state = np.asarray(state)
            
            # noise and sensor faults (implemented as "levels")
            
            if level == "1":
                noise_multiplier = 0.0
                drift_mag = 0.0
                bias = np.ones(len(state))
            elif level == "2":    
                noise_multiplier = 1.0
                drift_mag = self.drift_rates * (self.sim.current_time - self.sim.start_time).total_seconds() / 86400.0
                # 86400 seconds in a day
                bias = self.bias
            elif level == "3":
                noise_multiplier = 6.0
                drift_mag = self.drift_rates * (self.sim.current_time - self.sim.start_time).total_seconds() / 86400.0
                bias = self.bias
                
            if self.sim.system_units == "SI": # metric
                noise_mag = 0.025 # 2.5 centimeters = 0.025 meters
            elif self.sim.system_units == "US": # imperial
                noise_mag = 0.025 * 3.28084 # 2.5 centimeters ~ 0.082 feet
                
            #print("drift_mag" , drift_mag)
            #print("clean state (faults, no noise)", state)
            state = bias*state + drift_mag + np.random.normal(0, noise_multiplier*noise_mag, state.shape)
            #print("state after noise", state)
            return state
        else:
            print("State config not defined! \n ctrl is defined as False")
            return np.array([])

    def step(self, actions=None, level="1"):
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
                    if (level == "2" or level == "3") and self.actuator_schedule is not None and self.actuator_schedule[asset] is not None:
                        # if the current time has a "stuck" before it and a "fix" after it for the column "asset"
                        # assign the most recent value in the data_log and continue
                        # check if self.current_time has "stuck" before and "fix" after in actuator_schedule[asset]
                        # find the index value of the most recent "stuck" before self.current_time
                        stuck_times = self.actuator_schedule[asset].index[self.actuator_schedule[asset] == "stuck"]
                        fix_times = self.actuator_schedule[asset].index[self.actuator_schedule[asset] == "fix"]
                        if len(stuck_times) > 0:
                            most_recent_stuck_time = stuck_times[stuck_times < self.sim.current_time].max()
                            most_recent_fix_time = fix_times[fix_times > most_recent_stuck_time].min()
                            if most_recent_fix_time > self.sim.current_time:
                                # the actuator is stuck
                                #print("actuator ", asset, " not changed")
                                continue
                    self._setValvePosition(asset, valve_position)
            elif type(actions) == dict:
                for valve_position, asset in enumerate(actions):
                    if (level == "2" or level == "3") and self.actuator_schedule is not None and self.actuator_schedule[asset] is not None:
                        stuck_times = self.actuator_schedule[asset].index[self.actuator_schedule[asset] == "stuck"]
                        fix_times = self.actuator_schedule[asset].index[self.actuator_schedule[asset] == "fix"]
                        if len(stuck_times) > 0:
                            most_recent_stuck_time = stuck_times[stuck_times < self.sim.current_time].max()
                            most_recent_fix_time = fix_times[fix_times > most_recent_stuck_time].min()
                            if most_recent_fix_time > self.sim.current_time:
                                # the actuator is stuck
                                #print("actuator ", asset, " not changed")
                                continue
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
