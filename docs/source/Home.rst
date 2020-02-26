|:cloud_with_lightning_and_rain:| |:cloud_with_lightning_and_rain:| Welcome to the ``pystorms`` |:cloud_with_lightning_and_rain:| |:cloud_with_lightning_and_rain:|
===================================================================================================================================================================

What is pystorms?
-----------------

``pystorms`` is a python library for the development and evaluation of stormwater control algorithms created by the open-storm group. It is built on `pyswmm <https://github.com/OpenWaterAnalytics/pyswmm>`_, a python wrapper for the interacting with EPA-SWMM.
This library provides a collection of stormwater networks and streamlined programming interface that can be adopted for analyzing the performance of control algorithms. 

Installation
------------

This package is built in python 3.7 and is supported on all operating systems (Windows, Mac OS, Linux).

.. code:: bash

   pip install pystorms

If you run into any issues installing the package, please refer to the advanced section for additional installation instructions or feel free to contact us.

Getting Started
---------------

.. code:: python

   import pystorms

   def controller(state):
        actions = np.ones(2)
        if state[0] > 0.5:
                actions[0] = 0.5
        if state[1] > 0.5:
                actions[1] = 0.5
        return actions

   env = pystorms.scenarios.theta()

   done = False 
   while not done:
       state = env.state()
       actions = controller(state)
       done = env.step(actions)

Scenarios are the fundamental components of the pystorms library. A scenario constitutes of a stormwater network, a driving rainevent, and a control objective. When :code:`env = pystorms.scenarios.theta()` is called, all the necessary simulation modules are initialized. Simulation can be controlled using the method calls of :code:`env` class. States in the stormwater network(e.g. water levels, flows, pollutant concentrations) can be queried at any point of the simulation using the :code:`env.state()` method. Control actions can be implemented in the network and simulation can be progressed a time-step using the :code:`env.step(<your actions here>)` call. Please refer to the scenarios section for more information.

Research using pystorms
-----------------------


Citation
--------

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


Licence
-------
.. image:: ./figures/gplv3-or-later.svg
  :width: 100
  :align: left
