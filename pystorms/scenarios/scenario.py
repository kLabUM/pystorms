import abc
import numpy as np
from pystorms.utilities import perf_metrics


# Create a abstract class to force scenario class definition
class scenario(abc.ABC):
    @abc.abstractmethod
    # Specific to the scenario
    def step(self, actions=None, log=True):
        pass

    def _logger(self):
        for attribute in self.data_log.keys():
            if attribute not in ["performance_measure", "simulation_time"]:
                for element in self.data_log[attribute].keys():
                    self.data_log[attribute][element].append(
                        self.env.methods[attribute](element)
                    )
            elif attribute == 'simulation_time':
                self.data_log[attribute].append(self.env.methods[attribute]())

    def state(self,level="1"):
        return self.env._state(level=level)

    def performance(self, metric="cumulative"):
        return perf_metrics(self.data_log["performance_measure"], metric)

    def save(self, path=None):
        if path is None:
            path = "{0}/data_{1}.npy".format("./", self.config["name"])
        return np.save(path, self.data_log)
