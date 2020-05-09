import os
import platform

# Local Path
HERE = os.path.abspath(os.path.dirname(__file__))


def load_binary(name):
    r""" returns specified swmm binary based on the operating system.

    Parameters
    ----------
    name : str
        name of the binary. refer to the documentation for the key words

    Returns
    -------
    path : str
        path to the binary
    """
    # Find the path to the library
    op_system = platform.system()
    if op_system == "Linux":
        path = "/linux/libswmm5.so"
    elif op_system == "Windows":
        path = "/windows/swmm5.dll"
    elif op_system == "Darwin":
        path = "/macos/libswmm5.so"
    else:
        raise ValueError(
            "Operating system not identified, please check the binary path or overwrite it. \
                    Refer to the documentation for additional details."
        )

    # Choose the binary
    if name == "pollutant_support":
        path = name + path
        path = os.path.join(HERE, path)
    else:
        raise ValueError(
            "binaries not found. Refer to the documentation for the list avaiable binaries"
        )

    return path
