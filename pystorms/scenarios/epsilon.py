from pystorms.environment import environment
from pystorms.networks import load_network
from pystorms.config import load_config
from pystorms.scenarios import scenario
from pystorms.utilities import threshold
import yaml
import swmmio

class epsilon(scenario):
    r"""Epsilon Scenario

    Stormwater network with control structures in pipes

    Parameters
    ----------
    config : yaml file

    Methods
    ----------
    step: implement actions, progress simulation by a timestep, and compute performance metric

    Notes
    ----------
    Objective : Route flows to maintain constant outflow at the outlet

    """

    def __init__(self,version="1",level="1"):
        # Network configuration
        self.config = yaml.load(open(load_config("epsilon"), "r"), yaml.FullLoader)
        self.config["swmm_input"] = load_network(self.config["name"])

        # Dry weather TSS loading, measured at the outlet of the network
        self._performormance_threshold = 1.075  # Kg/sec
        
        self.version = version

        if version == "2":
            # make the threshold more stringent
            self._performormance_threshold = self._performormance_threshold * (7.0/10.0)

            model = swmmio.Model(self.config["swmm_input"])
            #print(model)
            #print(model.inp)
            #print(model.inp.files)
            
            # set end date to feb 15 2017
            #print(model.inp.options)
            model.inp.options.loc['END_DATE', 'Value'] = '02/15/2017'
            #print(model.inp.options)

            # increase the rainfall intensity by 10% throughout
            #print(model.inp.timeseries)
            # cast entries in model.inp.timeseries['Value'] to float
            model.inp.timeseries.loc[:, 'Value'] = model.inp.timeseries['Value'].astype(float)
            #print(model.inp.timeseries)
            model.inp.timeseries.loc[:, 'Value'] = 1.1*model.inp.timeseries.loc[:, 'Value']
            #print(model.inp.timeseries)
            model.inp.save(str(self.config["swmm_input"][:-4] + "_v2.inp")) 
            self.config["swmm_input"] = str(self.config["swmm_input"][:-4] + "_v2.inp")


        # Create the env based on the config file
        self.env = environment(self.config, ctrl=True, binary=self.config["binary"],version=version,level=level)

        # Create an object for storing data
        self.data_log = {
            "performance_measure": [],
            "loading": {},
            "pollutantL": {},
            "flow": {},
            "flooding": {},
            "simulation_time": []
        }

        # Data logger for storing _performormance data
        for ID, attribute in self.config["performance_targets"]:
            self.data_log[attribute][ID] = []

    def step(self, actions=None, log=True,version="1",level="1"):
        # Implement the action and take a step forward
        done = self.env.step(actions,level=level)

        # Log the flows in the networks
        if log:
            self._logger()

        # Estimate the _performormance
        __performance = 0.0  # temporary variable

        for ID, attribute in self.config["performance_targets"]:
            if attribute == "flooding":
                flood = self.env.methods[attribute](ID)
                if flood > 0.0:
                    __performance += 10 ** 9
            elif attribute == "loading":
                pollutantLoading = (
                    self.env.methods["pollutantL"](ID, 'TSS')
                    * self.env.methods["flow"](ID)
                    * 28.3168
                    / (10 ** 6)
                )
                __performance += threshold(pollutantLoading, self._performormance_threshold)

        # Record the _performormance
        self.data_log["performance_measure"].append(__performance)

        # Terminate the simulation
        if done:
            self.env.terminate()

        return done

    def _logger(self):
        # Log all the _performormance values;
        # additionally, other components can be added here
        for ID, attribute in self.config["performance_targets"]:
            if attribute == "loading":
                pollutantLoading = (
                    self.env.methods["pollutantL"](ID, 'TSS')
                    * self.env.methods["flow"](ID)
                    * 28.3168
                    / (10 ** 6)
                )
                self.data_log[attribute][ID].append(pollutantLoading)
            else:
                self.data_log[attribute][ID].append(self.env.methods[attribute](ID))
