|:cloud_with_lightning_and_rain:| |:cloud_with_lightning_and_rain:| Welcome to the ``pystorms`` |:cloud_with_lightning_and_rain:| |:cloud_with_lightning_and_rain:|
===================================================================================================================================================================

What is pystorms?
-----------------
Flooding is the leading cause of natural disaster deaths worldwide. Simultaneously, untold quantities of metals, bacteria, nutrients, and other pollutants are washing-off during storms into our streams and rivers. As a result, many parts of the world are dealing with chronically impaired coastlines due to algal blooms and other ecological disasters. Many of these challenges are presently addressed through classic approaches and new infrastructure construction (larger pipes, bigger basins, storage tanks, etc.).

Instead of building more, there is now an unprecedented opportunity to use internet-of-things and cyber-physical technologies to embed water systems with intelligence. The Internet-connected water systems of the future will control themselves (gates, valves, pumps), similar to self-driving cars, to reduce flooding and improve water quality. However, there is a dire need for more researchers to help enable the next generation of data and control algorithms for smart water systems.

We developed ``pystorms`` as a tool to collaboratively develop new algorithms for solving urban watershed specific control problems. In particular, we built this tool to bridge the gap between (i) experts in algorithms and control systems and (ii) experts in urban water systems (as well as anyone in between).

``pystorms`' is an easy-to-use, open-source Python library to allow quick evaluation of control strategies on a set of real world-inspired smart stormwater scenarios. It includes (1) a collection of real-world inspired stormwater networks and (2) a streamlined programming interface to analyze performance of control algorithms on the included networks. The library has been written so that it can be easily adapted and used with other stormwater networks or simulators, while maintaining the focus on the control algorithm itself.

``pystorms`` is built on `pyswmm <https://github.com/OpenWaterAnalytics/pyswmm>`_, which is a Python wrapper developed to interface with the United State's Environmental Protection Agency Storm Water Management Model `EPA-SWMM <https://www.epa.gov/water-research/storm-water-management-model-swmm>`_, thus allowing users to bypass the need for utilizing the EPA-SWMM GUI (available only on Windows operating systems) or compiling the EPA-SWMM source code for usage.

``pystorms`` is developed by Open-Storm, the open-source outreach efforts of a consortium of university researchers, industry experts, and municipal practitioners, and lead by the `Real-Time Water Systems Lab <http://www-personal.umich.edu/~bkerkez/>`_ at the University of Michigan.

Installation
------------

This package is built in python 3.7 and is supported on all operating systems (Windows, Mac OS, Linux).

.. code:: bash

   pip install pystorms

If you run into any issues installing the package, please refer to the advanced section for additional installation instructions or feel free to contact us.

Getting Started
---------------

The following code can be run to ensure the package has been successfully installed, and to learn the basics of how to use ``pystorms``.

.. code:: python

   import pystorms

   # (3) Define the control algorithm
   def controller(state):
        actions = np.ones(2)
        if state[0] > 0.5:
                actions[0] = 0.5
        if state[1] > 0.5:
                actions[1] = 0.5
        return actions

   # (1) Call and define the scenario
   env = pystorms.scenarios.theta()

   # (2) Run the simulation
   done = False
   while not done:
       state = env.state()
       actions = controller(state)
       done = env.step(actions)

The code above can be broken down into the following components:

(1) Define a scenario
  To run a stormwater control simulation using ``pystorms``, a **scenario** must first be defined. **Scenarios are the fundamental components of the pystorms library**, and each consist of (i) a stormwater network, (ii) a driving precipitation event, and (iii) a control objective.

  As seen in the code above, a scenario is called by defining a class (e.g. :code:`env`) that initializes the default simulation modules specific for the corresponding scenario. In the code above, the corresponding scenario is *theta*, which is a basic stormwater control scenario developed specifically for testing and debugging of new code and/or control algorithms in :code:`pystorms`.

  For the example above, the scenario class is defined using the default precipitation event and control objective of the theta scenario; it is defined fully by the following line of code: :code:`env = pystorms.scenarios.theta()`).

(2) Delineate a simulation
  A **simulation** for the scenario is then delineated using the method calls of the scenario class (i.e. :code:`env` in this example). For ``pystorms`` the method calls of the scenario class were defined to follow the typical actions that occur in-situ for actual control systems. Namely, there are two specific components: (i) querying the state of the system, and (ii) implementing control actions based on the system's state.

    i. States in the stormwater network (e.g. water levels, flows, pollutant concentrations) can be queried at any point of the simulation using the :code:`env.state()` method.

    ii. Control actions can be implemented in the network, and the simulation can be progressed forward for a specified time-step using the :code:`env.step(<your actions here>)` call. Please refer to the scenarios section for more information.

(3) Implementing the control algorithm
  The simulation set up in this way eases the ability explicitly define a control algorithm to decide on what control actions are to be implemented. As can be seen in  ``pystorms`` 


Research using pystorms
-----------------------


Citation
--------
While ``pystorms`` can be used freely, we ask that the origins of this tool be credited by using the following reference:

.. code:: latex

        @inproceedings{10.1145/3302509.3313336,
        author = {Rimer, Sara P. and Mullapudi, Abhiram and Troutman, Sara C. and Kerkez, Branko},
        title = {A Benchmarking Framework for Control and Optimization of Smart Stormwater Networks: Demo Abstract},
        year = {2019},
        isbn = {9781450362856},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi-org.proxy.lib.umich.edu/10.1145/3302509.3313336},
        doi = {10.1145/3302509.3313336},
        booktitle = {Proceedings of the 10th ACM/IEEE International Conference on Cyber-Physical Systems},
        pages = {350–351},
        numpages = {2},
        keywords = {real-time control, water infrastructure, smart cities},
        location = {Montreal, Quebec, Canada},
        series = {ICCPS ’19}
        }


License
-------
``pystorms`` is licensed under a GNU General Public License.

.. image:: ./figures/gplv3-or-later.svg
  :width: 100
  :align: left
