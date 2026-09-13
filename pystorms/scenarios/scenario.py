import abc
import numpy as np
from pystorms.utilities import perf_metrics


def validate_version(version, supported, name):
    r"""Normalise *version* to a string and check the scenario supports it.

    Parameters
    ----------
    version : str or int
        requested scenario version
    supported : tuple of str
        versions the scenario defines
    name : str
        scenario name, used in the error message

    Returns
    -------
    version : str
    """
    version = str(version)
    if version not in supported:
        raise ValueError(
            "scenario {0} supports version {1}; got {2!r}".format(
                name, " and ".join(supported), version
            )
        )
    return version


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

    def state(self, level=None):
        r"""Query the network state.

        Parameters
        ----------
        level : str, optional
            difficulty level to apply to the readings. Defaults to the level
            the scenario was built with. ``"1"`` can always be passed to read
            the true, noise free state.
        """
        return self.env._state(level=level)

    def performance(self, metric="cumulative"):
        return perf_metrics(self.data_log["performance_measure"], metric)

    def save(self, path=None):
        if path is None:
            path = "{0}/data_{1}.npy".format("./", self.config["name"])
        return np.save(path, self.data_log)

    def terminate(self):
        r"""Close the SWMM simulation.

        ``step`` does this on its own when the event ends. Call it yourself if
        you stop an episode early; SWMM allows one open simulation per process,
        so the next scenario cannot be built until this one is closed. Safe to
        call more than once.
        """
        self.env.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.terminate()
        return False
