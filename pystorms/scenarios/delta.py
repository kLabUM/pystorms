from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold, exponentialpenalty
import yaml
import swmmio

class delta(scenario):
    r"""Delta Scenario

    Separated stormwater network driven by a idealized event.

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
    1. To maintain levels of three detention ponds within a range of depths, and
    2. To maintain the network outflow below a threshold.

    Performance is measured as the following:
    1. First, deviation of depth above/below the "desired" depth range for the three detention ponds,
    2. Second, deviation of depth above/below "maximum/minimum" depth ranges for the three detention ponds
        and one other infiltration pond,
    3. Any flooding through the network, and
    4. Any deviation above the threshold of the outflow.

    """

    def __init__(self,version="1",level="1"):
        # Network configuration
        self.config = yaml.load(open(load_config("delta"), "r"), yaml.FullLoader)
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
        
        if version == "2":
            # threshold more stringent
            self.threshold = 0.5
            model = swmmio.Model(self.config["swmm_input"])
            # extend end date
            model.inp.options.loc['END_DATE', 'Value'] = '4/27/2016'
            # remove downstream structural flow limitation
            model.inp.xsections.loc["conduit_Eup","Geom1"] = 5.0 
            model.inp.xsections.loc["conduit_Edown","Geom1"] = 5.0 
            # eliminate uncontrollable subcatchment flooding
            model.inp.xsections.loc["conduit_Csc","Geom1"] = 3.0
            model.inp.xsections.loc["conduit_N1sc","Geom1"] = 3.0
            #print(model.inp.infiltration)
            
            model.inp.options.loc['VARIABLE_STEP',"Value"] = 0.00 # ensure all sims have same number of steps
            model.inp.options.loc["ROUTING_STEP","Value"] = "0:00:05" # X second hydraulic routing
            #print(model.inp.options)
            # change infiltration method to horton (error 200 in original)
            #model.inp.options.loc['INFILTRATION', 'Value'] = 'MODIFIED_HORTON'   
            # initial moisture deficit is indicated as 4.0 for most subcatchments. can't be greater than 1.0
            for subcatch in model.inp.infiltration.index:
                if model.inp.infiltration.loc[subcatch, 'IMDmax'] > 1.0:
                    #print(subcatch)
                    model.inp.infiltration.loc[subcatch, 'IMDmax'] = 0.4 # assume typo. decimal one point over.
                
            # increase all rainfall intensities  
            # cast entries in model.inp.timeseries['Value'] to float
            model.inp.timeseries.loc[:, 'Value'] = model.inp.timeseries['Value'].astype(float)
            #print(model.inp.timeseries['Value'][-1])
            model.inp.timeseries.loc[:, 'Value'] = 1.3*model.inp.timeseries.loc[:, 'Value']
            #print(model.inp.timeseries['Value'][-1])
            model.inp.save(str(self.config["swmm_input"][:-4] + "_v2.inp")) 
            self.config["swmm_input"] = str(self.config["swmm_input"][:-4] + "_v2.inp")

        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True)

        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
            "depthN": {},
            "flow": {},
            "flooding": {},
            "simulation_time": []
        }

        # Data logger for storing _performance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True,version="1",level="1"):
        # Implement the actions and take a step forward
        done = self.env.step(actions,level=level)

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
