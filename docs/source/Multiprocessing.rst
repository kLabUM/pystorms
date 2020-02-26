Multi-processing
================

pystorms environments can be seamlessly adopted for multiprocessing

.. code:: ipython3

    import pystorms
    import numpy as np
    from multiprocessing import Pool

``worker``
----------

This function takes a controller as an argument, enabling us to evaluate
multiple control strategies simultaniously.

.. code:: ipython3

    def worker(config):
        env = pystorms.scenarios.gamma()
        done = False
        # different controllers
        controller = config["controller"]
        while not done:
            actions = controller(env.state())
            done = env.step(actions)
        return env.performance()

``swarm``
---------

This function maps the worker function onto multiple processors and
return the performance

.. code:: ipython3

    def generate_swarm(config, worker, processors, jobs):
        """
        Generate workers based on the environment and controller
        """
        if type(config) == list:
            swarm_inputs = config
        else:
            swarm_inputs = [config for i in range(0, jobs)]
    
        with Pool(processors) as p:
            data = p.map(worker, swarm_inputs)
        return data

Example:
--------

.. code:: ipython3

    # Define two generic controllers
    def control_1(state):
        return np.ones(11)
    
    def control_2(state):
        return np.zeros(11)
    
    # Create the config file
    config = [{"controller": control_1}, {"controller": control_2}]

.. code:: ipython3

    generate_swarm(config, worker, 2, 2)




.. parsed-literal::

    [15736.942557976538, 282982.5873304134]



Lets time it to check that the function is running on mutiple
processors. If sucessful, simulation time should be half.

Serial and Parallel
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%timeit
    worker(config[0])
    worker(config[1])


.. parsed-literal::

    15.1 s ± 434 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


.. code:: ipython3

    %%timeit
    generate_swarm(config, worker, 2, 2)


.. parsed-literal::

    10.8 s ± 42.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Thats 3 seconds more than what i expected. This might be due the
initialization cost. This should go down as the number of simulations
increase.

.. code:: ipython3

    %%timeit
    worker(config[0])
    worker(config[1])
    worker(config[0])
    worker(config[1])
    worker(config[0])
    worker(config[1])


.. parsed-literal::

    44.1 s ± 79.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


.. code:: ipython3

    %%timeit
    generate_swarm(config, worker, 3, 6)


.. parsed-literal::

    10.8 s ± 68.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


.. code:: ipython3

    %%timeit
    generate_swarm(config, worker, 6, 6)


.. parsed-literal::

    11.1 s ± 451 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Thats consistent!
