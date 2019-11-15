from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pystorms",
    version="0.1.1",
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
            "StormLibrary/*.npy",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy"
    ],
)
