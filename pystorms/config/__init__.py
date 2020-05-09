import os

# Local Path
HERE = os.path.abspath(os.path.dirname(__file__))


def load_config(name):
    r""" returns the path to the desired network.

    Parameters
    ----------
    name : str
        name of the scenario. *alpha*, *beta*, *gamma*, *delta*, *epsilon*, and 'zeta' are valid
        keywords

    Returns
    -------
    path : str
        path to the scenario config
    """

    # Parse the file name
    path = os.path.join(HERE, name + ".yaml")

    # Check if network exists
    if not (os.path.isfile(path)):
        raise ValueError("Undefined Scenario, please refer to the documentation")

    return path
