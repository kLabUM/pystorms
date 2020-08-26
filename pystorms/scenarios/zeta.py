from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold, exponentialpenalty
import yaml


class zeta(scenario):
    r"""Zeta Scenario

    Separated stormwater network driven by a idealized event.

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step:

    Notes
    -----
    Objectives are the following:
    1. Minimization of accumulated CSO volume
    2. Minimization of CSO to the river/creek
    3. Maximizing flow to the WWTP
    4. Minimizing roughness of control.

    Performance is measured as the following:
    1. *2 for the flow to the river/creek
    2. *(−1) for the flow to the WWTP
    3. *(0.01) for the roughness of the control

    Weights of tanks:
    Tank1 - 1000
    Tank2 - 5000
    Tank3 - 5000
    Tank4 - 5000
    Tank5 - 5000
    Tank6 - 10000

    """

    def __init__(self):
        # Network configuration
        self.config = yaml.load(open(load_config("zeta"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        self.penalty_weight = {
            "T1": 1,
            "T2": 5,
            "T3": 5,
            "T4": 5,
            "T5": 5,
            "T6": 10
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

    def step(self, actions=None, log=True):
        # Implement the actions and take a step forward
        done = self.env.step(actions)

        # Log the flows in the networks
        if log:
            self._logger()

        # Initialize temporary variables
        __performance = 0.0 #
        #__simtime = self.sim._model.swmm_step() #TODO
        #__prevsimtime = self.data_log["simulation_time"][-1] #TODO
        #__timestep = (__simtime - __prevsimtime)/pd.Timedelta(1, unit='s') #TODO
        __timestep = 30 # seconds

        for ID, attribute in self.config["performance_targets"]:
            __floodrate = 0.0
            __throttleflow = 0.0
            __weight = 0.0
            __volume = 0.0
            __prevflow = 0.0
            __flowrate = 0.0
            if attribute == "flooding":  # compute penalty for CSO overflow
                __floodrate = self.env.methods[attribute](ID) # flooding rate
                __volume = __floodrate * __timestep # flooding volume
                if ID in ["T1", "T2", "T3", "T4", "T5", "T6"]:
                    __weight = self.penalty_weight[ID]
                else:
                    __weight = 1
            else:  # compute reward for flow to WWTP, and penalty for change in flow from previous timestep due to control (i.e. throttle flow)
                __flowrate = self.env.methods[attribute](ID)
                if ID == "C14": #conduit connected to "Out_to_WWTP" node
                    __volume = __flowrate * __timestep
                    __weight = -1
                else:
                    __prevflowrate = self.data_log[attribute][ID][-1]
                    __throttleflowrate = np.abs(__prevflowrate - __flowrate)
                    __volume = __throttleflowrate * __timestep
                    __weight = 0.01
            __performance += __volume * __weight


        # Record the _performance
        self.data_log["performance_measure"].append(__performance)
        #log the timestep here?

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
