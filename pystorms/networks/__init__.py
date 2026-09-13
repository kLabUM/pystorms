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


def cache_dir():
    r""" returns the per-user directory pystorms writes run files to.

    ``PYSTORMS_CACHE`` overrides the location. Otherwise it is
    ``$XDG_CACHE_HOME/pystorms/networks``, falling back to
    ``~/.cache/pystorms/networks``. The directory is created if needed.

    Returns
    -------
    path : str
        path to the cache directory
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

    return cache


def derived_network_path(source_path, suffix):
    r""" returns a writable path for a network derived from *source_path*.

    Scenario versions above ``"1"`` are built by rewriting the shipped ``.inp``
    file. The package directory is not writable in a normal install, so the
    derived network is placed in the cache directory instead.

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

    name = os.path.splitext(os.path.basename(source_path))[0]

    return os.path.join(cache_dir(), name + "_" + suffix + ".inp")


def run_file_paths(inp_path):
    r""" returns the report and binary output paths for a run of *inp_path*.

    SWMM writes a ``.rpt`` and a ``.out`` file for every run. By default they
    land next to the input file, which for the shipped networks is inside the
    installed package. Both are redirected into the cache directory instead.

    Parameters
    ----------
    inp_path : str
        path to the SWMM input file being run

    Returns
    -------
    report, output : tuple of str
        paths for the report and binary output files
    """

    name = os.path.splitext(os.path.basename(inp_path))[0]
    cache = cache_dir()

    return os.path.join(cache, name + ".rpt"), os.path.join(cache, name + ".out")
