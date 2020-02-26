=========================
Building Custom Scenarios
=========================

Developing custom scenarios for contributing to the ``pystorms`` library or your personal use is really easy.

**Steps for building a custom scenarios**

1. Anonymize the stormwater network, if you are considering contributing to the ``pystorms`` repository.
2. Identify the event drivers (e.g. stormevents, tides). 
3. Choose an objective and performance metric for evaluating the objective. 
4. Decide on the state and action space. 
5. Build the class and populate the yaml file. 

Anonymize the stormwater network
--------------------------------

To ensure that the stormwater network is anonymized, we apply randomized
transformations on the network that transform the coordinates of the
network, while maintaining the integrity of the network topology.

In a SWMM input file the geometry of the model elements (length, elevation, cross section, etc.,) and their coordinate locations (i.e. the location to be displayed in the visualization of the model in a program) are defined in separate sections.
This separation allows the transformation of the coordinate data without altering the geometry of the model elements.
This means that the coordinates of the model elements can be rotated, scaled, and translated without effecting any of the geometry values relevant to the execution of the model.

Anonymization procedure
^^^^^^^^^^^^^^^^^^^^^^^
        1. **Rotate** all coordinates by a common (random) angle between 0 and 360 degrees. This randomly generated rotation angle is not exposed.
        2. **Scale** all coordinates such that they are within a 100,000 x 100,000 unit box.
        3. **Translate** coordinates to be within the bounding box defined by its two corners of (0,0) and (100000, 100000).
        4. **Output** the anonymized coordinates to a new SWMM input file. By default comments are scrubbed from the input file, therefore no identifying metadata is kept in the anonymized version.
        5. **Inspect** anonymized SWMM input file for any identifiable information remaining.

More information on the anonymization routine and the code to do so can be found
`here <https://github.com/kLabUM/AnonymizeINP/blob/master/AnonymizeFullRoutine.ipynb>`__.

Once the network is anonymized, the names of the nodes, links, or any components that can be traced back to the original model have to be updated.
Currently, we do not have a proper nomenclature style for renaming the network components, so we leave this up to the better judgement of the users. 

Event driver
------------

The event driver can be a rainfall event, time series data of flows, or some initial volumes in nodes.
Basically, anything that is supported by the stormwater simulation engine.

Control objective and performance metric
----------------------------------------

The performance measure quantifies the ability of the control algorithm in achieving its objective.
For example, if your objective was to minimize the CSO volume, a performance measure could be the CSO volume that occurs with the implementation of control.

Choose the state and action space
---------------------------------

After choosing an appropriate performance measure for quantifying the control objective, pick the states (i.e., the network elements where certain states, such as flow, water level, or pollutant concentrations, are measured) and the action space (i.e., controllable network assets).

Building ``pystorms`` scenario class
------------------------------------
.. code:: python

   def MyAwesomeScenario: # Choose a name for your Scenario 
       """
       description of scenario 


       Methods
       -------
       < make a note of any new methods implemented for the class >
       
       Notes
       -----
       < any additional information you would want to provide about the scenario >

       """
       def __init__(self): # The scenario class ideally should not take any arguments
           
           # Scenario meta data is defined in the yaml file 
           self.config = yaml.load(open(PATH + "/config/<scenario name>.yaml", "r"), yaml.FullLoader) # Meta data is loaded into the class as a dict file, which can be modified by the users. 
           
           # Load the network input file.
           self.config["swmm_input"] = load_network(self.config["swmm_input"])
           # Refer to adding a network for more details on how to contribute a network to the library
           
           # Create the simulation environment
           self.env = environment(self.config, ctrl = True)
               
           # Create an object for logging data, this can include all the data being used for computing performance metrics and data that can be used for debugging
           self.data_log = {"performance_measure" : [],
                            "flows" : {},
                            "flooding" : {}}

           # Populate the data_log with elements whose measurements have to be recorded.
           for ID, attribute in self.config["performance_targets"]:
               self.data_log[attribute][ID] = []
               
       def step(self, actions, log= True):
           # Step ahead and update the state 
           _, done = self.env.step(actions)
           
           # A logger of the states, actions, performance targets, and any other useful information
           if log:
                self._logger()
           
           # Create a performance measure 
           __performance = 0.0
           
           for ID, attribute in self.config["performance_targets"]:
               """
               Implement your performance evaluation here.
               """               
               self.data_log["performance_measure"].append(__performance)
           return done
