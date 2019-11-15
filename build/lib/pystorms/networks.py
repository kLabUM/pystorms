import os

# Local Path
HERE = os.path.abspath(os.path.dirname(__file__))


def load_network(name):
    r""" returns the path to the desired network.

    Parameters
    ----------
    name : str
        name of the network. *alpha*, *beta*, *gamma*, *epsilon* and *delta* are valid
        keywords

    Returns
    -------
    path : str
        path to the network
    """
    if name == "alpha":
        path = os.path.join(HERE, "networks/alpha.inp")
    elif name == "beta":
        path = os.path.join(HERE, "networks/beta.inp")
    elif name == "delta":
        path = os.path.join(HERE, "networks/delta.inp")
    elif name == "epsilon":
        path = os.path.join(HERE, "networks/epsilon.inp")
    elif name == "gamma":
        path = os.path.join(HERE, "networks/gamma.inp")
    elif name == "theta":
        path = os.path.join(HERE, "networks/theta.inp")
    else:
        raise ValueError("Undefined Network, refer to the documentation")
    return path
