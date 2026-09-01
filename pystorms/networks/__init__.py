import os

# Local Path
HERE = os.path.abspath(os.path.dirname(__file__))


def load_network(name):
    r""" returns the path to the desired network.

    Parameters
    ----------
    name : str
        name of the network. *alpha*, *beta*, *gamma*, *delta*, *epsilon*, and *zeta* are valid
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


def derived_network_path(source_path, suffix):
    r""" returns a writable path for a network derived from *source_path*.

    Scenario versions above ``"1"`` are built by rewriting the shipped ``.inp``
    file. The package directory is not writable in a normal install, so the
    derived network is placed in a per-user cache directory instead.

    Parameters
    ----------
    source_path : str
        path to the shipped network the derived network is built from
    suffix : str
        suffix identifying the derivation, e.g. *v2*

    Returns
    -------
    path : str
        path to write the derived network to
    """

    cache = os.environ.get("PYSTORMS_CACHE")

    if cache is None:
        cache = os.path.join(
            os.environ.get(
                "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
            ),
            "pystorms",
            "networks",
        )

    os.makedirs(cache, exist_ok=True)

    name = os.path.basename(source_path)[:-4]

    return os.path.join(cache, name + "_" + suffix + ".inp")
