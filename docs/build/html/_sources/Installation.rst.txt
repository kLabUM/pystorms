Installation
============

To use the scenarios we have developed with our default stormwater simulator, two installations must occur: (i) an installation of pyswmm_lite and (ii) the corresponding Benchmarking library. Additionally, both of these also require python3 and numpy.

Before installation, we encourage users to consider setting up their Python environments using the virtualenv/venv packages for beetter control and organization of various Python packages and libraries. Details on these packages can be viewed here and here, with an additional tool called virtualenvwrapper available here.

The Easy Installation instructions below attempts to install both pyswmm_lite, the Benchmarking library, and the corresponding dependencies (if not yet installed) altogether, but if you run into any errors or would like demarcated installation instructions, follow the additional instructions presented in the Alternative Installation section.
Easy Installation

The easiest way to install the Benchmarking library and its dependencies is to run the following:

pip3 install git+https://github.com/kLabUM/Benchmarking

In case you encounter trouble in installing pyswmm_lite, then you can use the following set of commands instead:

git clone https://github.com/kLabUM/pyswmm_lite.git
cd pyswmm_lite
pip3 install .

Alternative Installation

An alternative method to install the library is the following:

    Clone/download the zip file of the Benchmarking repository from https://github.com/kLabUM/Benchmarking.git
    Move the downloaded Benchmarking to your desired directory
    Open your python client (Anaconda or generic python) from within the Benchmarking folder and run pip3 install .

Repeat the same method for pyswmm_lite if needed.

NOTE : You could also you pip instead of pip3

Raise an issue, if you run into any issues installing or working with the library.

https://github.com/kLabUM/Benchmarking


This repo has been developed in an effort to systematize quantitative analysis of stormwater control algorithms. It is a natural extension of the the Open-Storm mission to open up and ease access into the technical world of smart stormwater systems. Our initial efforts allowed us to develop open source and free tools for anyone to be able to deploy flood sensors, measure green infrastructure, or even control storm or sewer systems. Now we have developed a tool to be able to test the performance of algorithms used to coordinate these different sensing and control technologies that have been deployed throughout urban water systems.

For reference on this Benchmarking we refer the reader to our manuscript describing the motivation behind the Benchmarking effort and details on the specifics behind this corresponding repo. In general, this repo provides three components:

    A library of scenarios that are built to allow for systematic quantitative evaluation of stormwater control algorithms,
    A stormwater hydraulic simulator named pyswmm_lite and forked heavily from OWA's SWMM and pyswmm, and
    An environment script that links the pyswmm_lite simulator to the scenarios, and can be edited/updated by users who might want to interface the scenarios with other stormwater simulator software (the environment script is included in pyswmm-lite).


https://www.gnu.org/licenses/gpl-3.0
