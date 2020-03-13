# Overview
This repository contains the code that allows to generate and run an environment containing ns3 simulations. In order:
* **generate_simulations.py** requires a directory related to a topology, draws links and traffic distribution and for each instance of those it executes the related ns3 simulation
* **simulation.cc** requires a directory containing information about the simulation, topology and network (both traffic and links). Runs an ns3 simulation and creates the trace files regarding traffic, delay (e2e and per link), dropped packets, load. Executes build_dataset.py to create the dataset
* **build_dataset.py** requires a directory containing the trace files from simulation.cc, and creates the datasets by considering timeframes of length T [seconds].

# Generate simulations
This paragraph will focus on providing details regarding how we create more distributions regarding the traffic.
* **Requirement**: directory \<topology\> containing {simulation.txt, links.txt} or topology.xml, together with a routing.txt file. Need to set properly the base directory on the variable **folder**.
* **Structure of the files**:
    * simulation.txt is a single row file containing \<start\> \<T\> \<P\> \<N\>, being start the seconds after which the execution should start (usually 0), T the timeframe, P the number of periods of size T, N the number of nodes
    * links.txt is an L+1 rows file. First row contains L, following rows contain the pairs \<N1\> \<N2\> identifying the L edges
    * topology.xml is the file containing the network from http://sndlib.zib.de/home.action
    * routing.txt is a single row file containing N^4 values, one for each possible OD pair o, and N1-N2 edge e. The value can be 0 or 1, representing whether edge e is traversed during the flow related to OD pair o.

General command:
```shell
python generate_simulations.py <topology> <num_environments> <run> <identifier> <num_threads> <num_intensities> <num_simulations>
```
being
* topology: directory containing the requirements
* num_environments: number of link distributions the user wants to generate
* run: True if the user wants to directly execute all the simulations
* identifier: directory, that will be created inside the topology directory, that will contain all the outputs
* num_threads: max number of simulations running simultaneously
* num_intensities: number of intensities the user wants to generate for each environment. Their distribution depends on the capacity one
* num_simulations: number of simulations to be drawn for each pair (e, i). The values of the traffic will change among the different simulations for each OD flow, but they will belong to the same distribution

Example command:
```shell
python generate_simulations.py abilene 30 True v1 5 10 15
```

that requires the abilene topology to contain the files as mentioned above, and generates 30 environments, each of which having 10 intensities and 15 different simulations for a single (environment, intensity).

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
