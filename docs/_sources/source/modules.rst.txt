Modules
=======


Analysis module
---------------
.. automodule:: modules.analysis.AQiPTanalysis
   :members: DataManager, ArrayNNClassifier, Data, ImageData, TableData, ArrayData, AQiPTData, Trc
   :undoc-members:
   :show-inheritance:

Control module
--------------
.. automodule:: modules.control.AQiPTcontrol
   :members: function, pulse, track, sequence, experiment, producer, IAC, Grid, Sweep, quadratic_sweep, NCO, Device, mapSW
   :undoc-members:
   :show-inheritance:

DAQ module
----------
.. automodule:: modules.daq.AQiPTdaq
   :members: inspector, dashboard, plotSequences, plotLiveimage, graph, surface3D, colormap2D, plot1D, scatter2D, scatter3D
   :undoc-members:
   :show-inheritance:

Directory module
----------------
.. automodule:: modules.directory.AQiPTdirectory
   :members: directory_tree
   :undoc-members:
   :show-inheritance:

Emulator module
---------------
.. automodule:: modules.emulator.AQiPTemulator
   :members: atomicModel, atomicQRegister, scan, producer, actor, field, optical, acoustic, magnetic, microwave, rf, electric, gravity, optElement, OptSetup, beam
   :undoc-members:
   :show-inheritance:

Interface:API module
--------------------
.. automodule:: modules.interface.APIs.API
   :members: constant, wait, digital, digitals, analog, analogs, compose, data, showDAQ, generateSpecifications, generateDirector, generateProducer, runSimulation, machine, prepareAcquisition
   :undoc-members:
   :show-inheritance:

Kernel module
-------------
.. automodule:: modules.kernel.AQiPTkernel
   :members: atomSpecs, atomicData, datacell, hardwareSpecs, softwareSpecs, VARIABLES, IDsBench, IDs, RydbergQubitSchedule, RydbergQubit, RydbergQuantumRegister
   :undoc-members:
   :show-inheritance:
