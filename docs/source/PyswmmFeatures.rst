``pyswmm`` functionality
========================

``pystorms`` is built on ``pyswmm``. It uses ``pyswmm`` as its back-end
for interacting with EPA-SWMM’s computational engine. Hence, all the
functionality in ``pyswmm`` is inherently available in ``pystorms``.
``pystorms`` by defaults supports a subset of the states though its API.
These subsets of states (i.e. depth, flows, inflows) were chosen to
represent frequently used parameters for making control decisions. Refer
to the documentation on states for more details on the supported
parameters. ``pystorms`` architecture is designed to enable users easy
access to all the existing pyswmm functionality.

.. code:: ipython3

    import pystorms
    import pyswmm.toolkitapi as tkai

This example demonstrates how pyswmm functionality can be invoked from
`pystorms`.

.. code:: ipython3

    env = pystorms.scenarios.theta()

All function calls being used in the environment for populating state
vector are listed in the ``env.env.methods`` dictionary.

.. code:: ipython3

    env.env.methods




.. parsed-literal::

    {'depthN': <bound method environment._getNodeDepth of <pystorms.environment.environment object at 0x11929d690>>,
     'depthL': <bound method environment._getLinkDepth of <pystorms.environment.environment object at 0x11929d690>>,
     'flow': <bound method environment._getLinkFlow of <pystorms.environment.environment object at 0x11929d690>>,
     'flooding': <bound method environment._getNodeFlooding of <pystorms.environment.environment object at 0x11929d690>>,
     'inflow': <bound method environment._getNodeInflow of <pystorms.environment.environment object at 0x11929d690>>,
     'pollutantN': <bound method environment._getNodePollutant of <pystorms.environment.environment object at 0x11929d690>>,
     'pollutantL': <bound method environment._getLinkPollutant of <pystorms.environment.environment object at 0x11929d690>>}



Let us say, we want to use volume as a state. All we have to do is add
the function call reading volume to the dict.

.. code:: ipython3

    def _getNodeVolume(NodeID):
        return env.env.sim._model.getNodeResult(NodeID, tkai.NodeResults.newVolume.value)

`env` refers to the scenario being initialized.
For example, if the scenario was initialized as `sce` the first class in the return statement would be `sce`. env in `<scenario class>.env` refers to the environment class used to communicate with pyswmm/EPA-SWMM.
`sim._model` refers to the EPA-SWMM simulation initialized by invoking the scenario class.
`getNodeResult` is the functional call that queries the volume from the EPA-SWMM.

.. code:: ipython3

    env.env.methods["volumeN"] = _getNodeVolume

Lets add volume to the state vector

.. code:: ipython3

    env.config["states"]




.. parsed-literal::

    [('P1', 'depthN'), ('P2', 'depthN')]



.. code:: ipython3

    env.config["states"].append(('P1', 'volumeN'))
    env.config["states"].append(('P2', 'volumeN'))

*NOTE:* Arguments to the volume function are tuple appended to the config dict. Refer to `environment.py <https://github.com/kLabUM/pystorms/blob/ffa3564ef5f80811ca246b10ffeb0e38f36befdb/pystorms/environment.py#L74>`_ for more details on how state vector is populated. 


.. code:: ipython3

    env.config["states"]




.. parsed-literal::

    [('P1', 'depthN'), ('P2', 'depthN'), ('P1', 'volumeN'), ('P2', 'volumeN')]



Now when ``env.state()`` is called, it returns all both depth and volume
in nodes

.. code:: ipython3

    env.state()




.. parsed-literal::

    array([0., 0., 0., 0.])



Refer to pyswmm documentation for details on the all supported
parameters.

Example
-------

.. code:: python

        import pystorms
        import pandas as pd
        import pyswmm.toolkitapi as tkai
        import matplotlib.pyplot as plt


        # Create the function call for reading volume
        def getNodeVolume(NodeID):
            return env.env.sim._model.getNodeResult(NodeID, tkai.NodeResults.newVolume.value)


        # Initalize scenario
        env = pystorms.scenarios.theta()

        # Update the methods dict
        env.env.methods["volumeN"] = getNodeVolume
        # Update state vector
        env.env.config["states"].append(("P1", "volumeN"))
        env.env.config["states"].append(("P2", "volumeN"))

        done = False
        data = []
        while not done:
            state = env.state()
            done = env.step([1, 1])
            data.append(state)

        data = pd.DataFrame(data, columns=["depthP1", "depthP2", "volumeP1", "volumeP2"])
        data.plot()
        plt.show()
