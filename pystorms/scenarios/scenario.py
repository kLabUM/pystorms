import abc
from pystorms.utilities import perf_metrics


# Create a abstract class to force scenario class definition
class scenario(abc.ABC):
    @abc.abstractmethod
    # Specific to the scenario
    def step(self, actions=None, log=True):
        pass

    def _logger(self):
        for attribute in self.data_log.keys():
            if attribute != "performance_measure":
                for element in self.data_log[attribute].keys():
                    self.data_log[attribute][element].append(
                        self.env.methods[attribute](element)
                    )

    def state(self):
        return self.env._state()

    def performance(self, metric="mean"):
        return perf_metrics(self.data_log["performance_measure"], metric)
