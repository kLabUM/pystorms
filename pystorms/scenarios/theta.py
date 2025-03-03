from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold
import yaml
#import swmm_api
import swmmio

class theta(scenario):
    r"""Theta Scenario

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
    Performance is measured as the deviation from the threshold.

    """

    def __init__(self,version="1",level="1"):
        # Network configuration
        self.config = yaml.load(open(load_config("theta"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])
        

        self.version = version
        
  
        self.threshold = 0.5

        
        if version == "2":
            # make the threshold more stringent
            self.threshold = self.threshold * (1.0/2.0)
            #  make the max depth of one of the nodes smaller. error because sim not running. maybe use swmm-api?
            #print(self.config['states'][0][0])
            model = swmmio.Model(self.config["swmm_input"])

            #print(model.inp.storage)
            #print(model.inp.storage.loc[self.config['states'][0][0] , 'MaxD'])
            model.inp.storage.loc[self.config['states'][0][0] , 'MaxD'] = model.inp.storage.loc[self.config['states'][0][0] , 'MaxD'] / 2.0
            #print(model.inp.storage)
            
            #print(self.config["swmm_input"])
            # end the simulation sooner, half the length of the original
            #print(model.inp.options)
            model.inp.options.loc['END_DATE', 'Value'] = "2/26/2018"
            model.inp.options.loc["END_TIME", "Value"] = "12:00:00"
            #print(model.inp.options)
            model.inp.save(str(self.config["swmm_input"][:-4] + "_v2.inp")) 
            self.config["swmm_input"] = str(self.config["swmm_input"][:-4] + "_v2.inp")

            
        # Create the environment based on the physical parameters
        self.env = environment(self.config, ctrl=True,version=version,level=level)


        # Create an object for storing the data points
        self.data_log = {
            "performance_measure": [],
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
            if attribute == "flooding":
                __flood = self.env.methods[attribute](ID)
                if __flood > 0.0:
                    __performance += 10 ** 6
            if attribute == "flow":
                __flow = self.env.methods[attribute](ID)
                __performance = threshold(
                    value=__flow, target=self.threshold, scaling=10.0
                )

        # Record the _performance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:   
            self.env.terminate()

        return done
