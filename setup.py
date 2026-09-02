import re

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("pystorms/__init__.py", "r") as fh:
    version = re.search(r'__version__ = "([^"]+)"', fh.read()).group(1)

setup(
    name="pystorms",
    version=version,
    description="Simulation sandbox for stormwater control algorithms",
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
        ]
    },
    license="GPL-3.0-only",
    license_files=["LICENSE"],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy>=1.18.4",
        "pyswmm>=1.0.1",
        "pyyaml>=5.3",
        "pandas>=1.0",
        "swmmio>=0.6.11"
    ],
)
