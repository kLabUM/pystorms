from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pystorms",
    version="0.4.0",
    description="Simulation Sandbox for the development and evaluation of stormwater control algorithms",
    author="Abhiram Mullapudi, Sara C. Troutman, Sara Rimer, Branko Kerkez",
    author_email="abhiramm@umich.edu, stroutm@umich.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kLabUM/pystorms",
    packages=['pystorms'],
    package_data={
        "pystorms": [
            "networks/*.inp",
            "networks/*.py",
            "event_drivers/*.npy",
            "config/*.yaml",
            "config/*.py",
            "scenarios/*.py",
            "binaries/*.py",
            "binaries/pollutant_support/macos/libswmm5.so",
            "binaries/pollutant_support/linux/libswmm5.so",
            "binaries/pollutant_support/windows/swmm5.dll"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.18.4",
        "pyswmm",
        "pyyaml>=5.3"
    ],
)
