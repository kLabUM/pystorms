import os

# Local Path
HERE = os.path.abspath(os.path.dirname(__file__))


def load_network(name):
    r""" returns the path to the desired network.

    Parameters
    ----------
    name : str
        name of the network. *alpha*, *beta*, *gamma*, *delta*, *epsilon*, and 'zeta' are valid
        keywords

    Returns
    -------
    path : str
        path to the network
    """

    # Parse the file name
    path = os.path.join(HERE, name + ".inp")

    # Check if network exists
    if not (os.path.isfile(path)):
        raise ValueError("Undefined Network, please refer to the documentation")

    return path
