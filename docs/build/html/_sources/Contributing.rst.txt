Contributing
============

Thank you for your interest !

Steps for contributing
----------------------

Master branch in the repository houses the code that has been completely
tested and is bug free to the best of out knowledge. If you encounter a
bug in the code, please raise an issue in the repository.

`Instructions on raising a
issue <https://help.github.com/en/articles/creating-an-issue>`__

Addressing bug fix
------------------

1. Fork the repository and clone it into your local machine

.. code:: bash

   git clone https://github.com/kLabUM/pystorms
   cd pystorms 

2. Create a new branch with your or just

For example, I am fixing a issue with the pollutant, I would create a
new branch using this command.

.. code:: bash

   git checkout -b abhiramm7_pollutantfix 

3. Add you fixes and push the changes into your fork of the repository.

.. code:: bash

   git add <your changed files>
   git push origin abhiramm7_pollutantfix 

`More
details <https://help.github.com/en/articles/pushing-to-a-remote>`__

4. Once you confident on the changes, you can create a pull request on
   the benchmarking repository.

`Raising a pull
request <https://help.github.com/en/articles/creating-a-pull-request-from-a-fork>`__

We can then work though the pull request.

Testing
-------

Please create a unit test for the code addition or any contribution you
wish make to the library. This repository uses ``pytest`` for testing.
Unit tests can be found in the ``/pystorms/tests/``

Details on where to include what
--------------------------------

``networks.py``
~~~~~~~~~~~~~~~

This file provides access to the input files available in the network.
So if you were to add a new network to the library, you would do the
following.

1. Add your processed input file (refer to building scenario on how to
   process your input file) to the networks folder. **Once you add your
   network to the library, it would be public. So please be carefull on
   what you upload**

2. Once the file has been updated to the networks folder, add the
   reference to the name in the ``networks.py``

.. code:: python

   ...
   elif name="<your network>"
       path = os.path.join(HERE, 'networks/<your network>.inp')

3. Add test to the ``test/tests_networks.py``

.. code:: python

   network = benchmarking.networks.load_network("<your name>")
   assert("inp"=  == network[-3:])

``utilities.py``
~~~~~~~~~~~~~~~~

Any **general** function you might need in developing scenarios would be
in the ``utilities.py``.

For example, we use ``append_rainfall`` function for adding rainfall
timeseries to input files. This can be used on *any* input file and is
not specific to a particular scenario. Hence, this function would be in
``utilities.py``.

``scenario.py``
~~~~~~~~~~~~~~~

This contains the main scenarios used for testing control
algorithms. Refer to building scenarios for more details on how to build
a scenario.

Anything that is **specific** to a network or scenario would go here.

``config``
~~~~~~~~~~

All the configuration files, like the input file, state and action space, and performance metrics are placed in this file. Create a new file for the scenario you are creating.  

.. code:: yaml

        # Configuration file for scenario theta 

        # swmm inp file 
        swmm_input: theta
        # state definitions
        states:
                - !!python/tuple 
                  - P1
                  - depthN
                - !!python/tuple
                  - P2
                  - depthN
        # Action space 
        action_space:
                - "1"
                - "2"
        # Performance Targets
        performance_targets:
                - !!python/tuple
                  - "8"
                  - flow
                - !!python/tuple
                  - P1
                  - flooding
                - !!python/tuple
                  - P2
                  - flooding

