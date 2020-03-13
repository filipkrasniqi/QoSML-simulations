# Overview
This repository contains the code that allows to generate and run an environment containing ns3 simulations. The simulations are implemented with the use of ns3 simulator (https://www.nsnam.org/); for more details on how to build the project and run with the (highly suggested) optimized compilation, have a look to the [tutorial](https://www.nsnam.org/docs/tutorial/html/). For those familiar to the simulator, the workspace contains the version **3.29**. In addition to ns3, some modules has been touched. The file [**requirements.txt**](https://github.com/filipkrasniqi/QoSML-simulations/blob/master/ns3/workspace/ns-allinone-3.29/ns-3.29/requirements.txt) contains all the python dependencies and they can be installed with pip. I will provide details on the code that creates the setting for the simulations, executes them and builds the dataset all according to the files (all in the [workspace](https://github.com/filipkrasniqi/QoSML-simulations/tree/master/ns3/workspace/ns-allinone-3.29) directory). In order:
* [**generate_simulations.py**](https://github.com/filipkrasniqi/QoSML-simulations/blob/master/ns3/workspace/ns-allinone-3.29/ns-3.29/generate_simulations.py) requires a directory related to a topology, draws links and traffic distribution and for each instance of those it executes the related ns3 simulation
* [**simulation.cc**](https://github.com/filipkrasniqi/QoSML-simulations/blob/master/ns3/workspace/ns-allinone-3.29/ns-3.29/scratch/simulation.cc) requires a directory containing information about the simulation, topology and network (both traffic and links). Runs an ns3 simulation and creates the trace files regarding traffic, delay (e2e and per link), dropped packets, load. Executes build_dataset.py to create the dataset
* [**build_dataset.py**](https://github.com/filipkrasniqi/QoSML-simulations/blob/master/ns3/workspace/ns-allinone-3.29/ns-3.29/build_dataset.py) requires a directory containing the trace files from simulation.cc, and creates the datasets by considering timeframes of length T [seconds].

**IMPORTANT**: first things a user of this repository would need to do is to:
1. **make waf executable**: go ns3/workspace/ns-allinone-3.29/ns-3.29/ and execute chmod +x waf
2. **configure and compile ns3**: the code has been usually compiled with waf. Commands to do that are the following (as explained in the [ns3 tutorial](https://www.nsnam.org/docs/tutorial/html/)):

Configuration (mandatory first time, then to be done only in case of switch from build optimized to debug):
```shell
./waf configure --build-profile=optimized --enable-examples --enable-tests
```

Compilation:
```shell
./waf
```

3. **be coherent with the defined structure**. The [**datasets**](https://github.com/filipkrasniqi/QoSML-simulations/tree/master/ns3/workspace/ns-allinone-3.29/ns-3.29/datasets) directory contains all the datasets as expected from the learning. This directory is meant to be inside the ns3 environment, so the user should either just stick to the provided structure (pull the repository on the user directory in Linux, i.e., the scratch directory is in ~/ns3/workspace/ns-allinone-3.29/ns-3.29/), or change the variables inside generate_simulations.py and simulation.cc.
- generate_simulations.py: the variable is base_dir_proj, defined as '~/ns3/workspace/ns-allinone-3.29/ns-3.29'
- simulation.cc: the variable is base_directory

**LAST BUT NOT THE LEAST: some information on the location of the files**. The repository keeps the structure of ns3, additional file (comprehending the code that is explained in the next sections) are all in the [workspace](https://github.com/filipkrasniqi/QoSML-simulations/blob/master/ns3/workspace/ns-allinone-3.29/ns-3.29). Here you can find, except for the common files in ns3, the c++ implementation of the simulation in the *scratch* directory, the python code in this exact directory, the [datasets](https://github.com/filipkrasniqi/QoSML-simulations/blob/master/ns3/workspace/ns-allinone-3.29/ns-3.29/datasets) and, eventually, once you run also the training, the exported models in the exported directory (the code in the learning repository will save them here if no changes are made).

# Generate simulations
This paragraph will focus on providing details regarding how to create traffic under different distributions of the parameters.
* **Requirement**: directory \<topology\> containing {simulation.txt, links.txt} or topology.xml, together with a routing.txt file. Need to set properly the base directory on the variable **folder**.
* **Structure of the files**:
    * simulation.txt is a single row file containing \<start\> \<T\> \<P\> \<N\>, being start the seconds after which the execution should start (usually 0), T the timeframe, P the number of periods of size T, N the number of nodes
    * links.txt is an L+1 rows file. First row contains L, following rows contain the pairs \<N1\> \<N2\> identifying the L edges
    * topology.xml is the file containing the network from http://sndlib.zib.de/home.action
    * routing.txt is a single row file containing N^4 values, one for each possible OD pair o, and N1-N2 edge e. The value can be 0 or 1, representing whether edge e is traversed during the flow related to OD pair o.

General command:
```shell
python generate_simulations.py <topology> <identifier> <num_threads> <num_intensities> <num_propagation_delays> <num_capacities> <num_simulations> <run>
```
being
* topology: directory containing the
* identifier: directory, that will be created inside the topology directory, that will contain all the outputs
* num_threads: max number of simulations running simultaneously
* num_intensities: number of intensities the user wants to generate for each environment. Their distribution depends on the capacity one
* num_propagation_delays: number of propagation delays distributions to generate the traffic from.
* num_capacities: number of capacities distributions
* run: True if the user wants to directly execute all the simulations
* num_simulations: number of simulations to be drawn for each pair (c, p, i). The values of the traffic will change among the different simulations for each OD flow, but they will belong to the same distribution

Example command:
```shell
python generate_simulations.py abilene v1 8 10 10 50 1 True
```

that requires the abilene topology to contain the files as mentioned above, and generates 500 link distributions (capacity, propagation delay), each of which will communicate at 10 different traffic intensities and 1 different simulations for a single (capacity, propagation delay, intensity).

generate_simulations.py will handle the scheduling of the simulations.

# Execute a simulation
The simulation.cc code will build the network and start the applications for each OD flow. The output will be in the ns3_output/ directory and will contain some temporary trace files required from build_dataset.py. Note that these files can become really heavy depending on the application, as they contain the history of all the traffic. build_dataset.py will delete the files at the end of his execution.

The trace files are the followings:
* enqueue.tr: when packets have been enqueued
* dequeue.tr: when packets have been dequeued. Together with enqueue, allows to compute the delay matrix
* tx.tr: when packets are being sent. Allows to compute the traffic matrix and the load
* rx.tr: when packets are being received
* routing.routes: exported file from ns3 related to the routing
* addresses.tr: list of addresses for each node. Allows to compute, together with routing.routes, the routing matrix
* dropped.tr: when a packet has been dropped

For more details on this, please refer to this [presentation](https://github.com/filipkrasniqi/QoSML-simulations/tree/master/assets/FKpresentation1707.pptx).

Remember that the **run** parameter to generate_simulations allows to build the directories from which the user could run the single simulation. Once done that, a suggestion on the command to run is given. An example is the following:

```shell
./waf --run "scratch/simulation --topology=abilene --identifier=v1 --intensity_level=intensity_0 --capacity_number=0 --propagation_delay_number=0"
```

that executes the simulation related to the abilene network, and the first drawn capacity, intensity and propagation delay distributions.

# Build a dataset
It builds the datasets from the trace files. The datasets are prefixed by \<NODES\>\_\<ACTIVE_NODES\>\_1\_\<PERIODS\>\_P\_\<OUTPUT\>.txt are the followings:
* OUTPUT = TM: traffic matrix
* dropped
* delay_pl: per link delay
* delaye2e: end to end delay
* routing
* load
* links: queue sizes and capacities
* packet_info: number of sent packets and their average size
* packet_info_link: number of sent packets and their average size groupped by link
* packet_info_od: number of sent packets and their average size groupped by OD flow

# Integrate existing datasets
To reuse already generated datasets that for disk space reason are **not** present in this repository (adviced in case the user wants to skip the execution of the simulations, as they may take some time) you should:
* download the dataset from the server where they are (at this moment, there is not such a place)
* copy them and stick to how the current structure is, i.e., to the structure in the datasets [**datasets**](https://github.com/filipkrasniqi/QoSML-simulations/tree/master/ns3/workspace/ns-allinone-3.29/ns-3.29/datasets)
    * you can notice that the [datasets](https://github.com/filipkrasniqi/QoSML-simulations/tree/master/ns3/workspace/ns-allinone-3.29/ns-3.29/datasets) folder is meant to contain three directories
        * **ns3**: contains a directory for each topology, each of which contains the files that are used to start the simulations. Each topology directory will be filled with a dataset, which name is the identifier
        * **understanding**: contains a directory for each topology, and each topology contains the raw data plus the links file (must be created separately). 
        * **routenet**: contains a directory for each topology, and each topology contains the raw data plus the links file (must be created separately).
    * e.g.: you want to import the abilene dataset associated to level 1 that is, let's say, in abilene\_l1.tar.gz. Once the content of the file has been extracted, you should expect a directory (let's say, simulation\_v1) containing a *list of directories **intensity_\<something\>** *, and the files links.txt, simulation.txt, log.txt. Move the entire content of the directory simulation\_v1 inside the directory of abilene and... les jeux sont faits!

If, for any reason, the structure to be kept is not really clear, try to execute a quick simulation and have a look to how the dataset get built.
