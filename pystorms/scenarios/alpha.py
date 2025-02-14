from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold, exponentialpenalty
import yaml
import swmmio

class alpha(scenario):
    r"""Alpha Scenario

    Separated stormwater network driven by a 2-year storm event.
    Scenario is adapted from SWMM Apps tutorial Example 8.

    Parameters
    ----------
    config : yaml configuration file
        physical attributes of the network.

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    -----
    Objective is the following:
    1. To minimize the combined sewer overflow of the system
    2. To avoid flooding

    Performance is measured as the following:
    1. First, any combined sewer overflow,
    2. Second, any flooding in the network

    """

    def __init__(self,version="1", level="1"):
        # Network configuration
        self.config = yaml.load(open(load_config("alpha"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])
        #print(self.config["swmm_input"])
        # suppress print statements from this script


        if version == "2":
            # make the action space the weirs in additon to the orifices
            model = swmmio.Model(self.config["swmm_input"])
            #print(self.config['action_space'])
            #print(model.inp.weirs)
            for item in model.inp.weirs.index.tolist():
                self.config['action_space'].append(item)
            #print(self.config['action_space'])
            '''
            for col in model.inp.xsections:
                print(col)
            for col in model.inp.weirs:
                print(col)
            # set the max weir height such that they can completely block flow
            for weir in model.inp.weirs.index:
                # find the max height of the upstream regulator. that's the same number with a prefix of R instead of W
                regulator = "R" + weir[1:]
                print(weir)
                print(regulator)
                print(model.inp.junctions.loc[regulator, 'MaxDepth'])
                print(model.inp.weirs.loc[weir, 'CrestHeight'])
                model.inp.xsections.loc[weir, 'Geom1'] = model.inp.junctions.loc[regulator, 'MaxDepth'] - model.inp.weirs.loc[weir, 'CrestHeight']
                
                '''
            # no changes to the underlying model (at least for now)
            #model.inp.save(str(self.config["swmm_input"][:-4] + "_v2.inp"))
            #self.config["swmm_input"] = str(self.config["swmm_input"][:-4] + "_v2.inp")
            



        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True,version=version,level=level)

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "simulation_time": [],
            "flow": {},
            "volume": {},
            "flooding": {},
            "simulation_time": [],
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True,level="1",version="1"):
        # Implement the actions and take a step forward
        done = self.env.step(actions,level=level)

        # Temp variables
        __performance = 0.0

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

        # Estimate the performance
        for ID, attribute in self.config["performance_targets"]:
            # compute penalty for flooding
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += __flood * (10 ** 6)
            # compute penalty for CSO overflow
            # TODO - decide if we want to have a penalty for negative flow
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __volume = __timestep * __flow
                __performance += threshold(value=__volume, target=0.0, scaling=1.0)

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done
