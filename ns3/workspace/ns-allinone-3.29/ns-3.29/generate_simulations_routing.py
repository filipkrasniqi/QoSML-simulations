import concurrent.futures
import subprocess
import pandas as pd
import random
import numpy as np
import re

import os
from os import listdir
from os.path import join, expanduser
import pathlib
from multiprocessing.dummy import Pool as ThreadPool

import sys
sys.path.insert(0, '/home/k80054556/notebooks/libs/')
from columns import *
from shutil import copyfile

import scipy.stats as ss
import xml.etree.ElementTree as ET
import networkx as nx
import itertools

import time
start = time.process_time()

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , sys.argv)

assert len(sys.argv) == 8, "Errore"

arguments = sys.argv

log = True
'''
Input: topology, number of simulations, environment name, #threads, number of intensities.
Topology dir MUST contain two alternative set of files:
- {link, simulation, routing}
- {topology.xml, simulation, routing}, being .xml obtained in the same format as provided in sndlib.zib.de
Eventually, can contain also traffic. If not present, it creates a traffic for all pairs.
Output: prepares links.txt, simulation.txt, traffic.txt for every combination (simulation, intensity).
Fixed a simulation, all intensities have in common the link properties.
Fixed an intensity, all simulations have in common the traffic distribution.
'''
folder = join(*[expanduser('~'), 'notebooks', 'datasets', 'ns3'])+"/"
TOPOLOGY_NAME = str(arguments[1])
TOPOLOGY_DIRECTORY = folder+TOPOLOGY_NAME+"/"
NUMBER_OF_SIMULATIONS = int(arguments[2])
START_PROCESS = bool(arguments[3] == "True")
ENV_NAME = str(arguments[4])
MAX_WORKERS = int(arguments[5])
NUM_INTENSITIES = int(arguments[6])
NUMBER_OF_ROUTINGS = int(arguments[7])
print("Start process: {}".format(START_PROCESS))
CURRENT_SIM_DIR = "simulation_"+ENV_NAME

print("Max workers: {}".format(MAX_WORKERS))

'''
Values read from simulation file
'''
SIMULATION_START = 0
PERIODS_LENGTH = 0
NUM_PERIODS = 0
NUM_NODES = 0

'''
Read simulation info from file. If not present, it requires the topology.xml and builds both links and simulation
'''
weights = []
try:
    simulation_f = open(TOPOLOGY_DIRECTORY+'simulation.txt' , 'r' )
    links_f = open(TOPOLOGY_DIRECTORY+'links.txt' , 'r' )
    # read simulation
    for count_line, line in enumerate(simulation_f.readlines()):
        if(count_line == 0):
            sim_props = line.split(" ")
            assert len(sim_props) == 4, sim_props
            SIMULATION_START = int(sim_props[0])
            PERIODS_LENGTH = float(sim_props[1])
            NUM_PERIODS = int(sim_props[2])
            NUM_NODES = int(sim_props[3])

    simulation_f.close()

    '''
    Read links from file
    '''
    links_base = []
    # read links
    for count_line, line in enumerate(links_f.readlines()):
        if(count_line == 0):
            NUM_LINKS = int(line)
        else:
            if count_line <= NUM_LINKS:
                link_props = line.split(" ")
                assert len(link_props) == 2, link_props
                links_base.append(line.rstrip())

                first_node, second_node = int(link_props[0]), int(link_props[1])
                weights.append({"first_node": first_node, "second_node": second_node, "weight": 1})

        links_f.close()
except:
    print("Trying to set simulation and links from topology.xml...")
    tree = ET.parse(TOPOLOGY_DIRECTORY+'topology.xml')
    root = tree.getroot()
    domain = "{http://sndlib.zib.de/network}"

    networkStructure = root[1]
    nodes_xml = networkStructure.findall(domain+"nodes")[0].findall(domain+"node")
    links_xml = networkStructure.findall(domain+"links")[0].findall(domain+"link")

    nodes = []
    links_content = str(len(links_xml))+"\n"
    for ns in nodes_xml:
        nodes.append(ns.get('id'))

    for l in links_xml:
        source = l.find(domain+'source').text
        target = l.find(domain+'target').text
        links_content += str(nodes.index(source)) + " "+str(nodes.index(target))+"\n"

    # write simulation.txt
    f = open(TOPOLOGY_DIRECTORY+"simulation.txt", "w")
    f.write('0 0.1 500 '+str(len(nodes)))
    f.close()

    f = open(TOPOLOGY_DIRECTORY+"links.txt", "w")
    f.write(links_content)
    f.close()

'''
Read traffics from file. If not present, generates two traffic flows for each pair of nodes
'''
traffics_base = []

try:
    traffics_f = open(TOPOLOGY_DIRECTORY+'traffic.txt' , 'r' )
except:
    n_nodes = NUM_NODES
    traffic_content = str(n_nodes**2 - n_nodes)+"\n"
    for i in range(0,n_nodes):
        for j in range(0,n_nodes):
            if i != j:
                traffic_content += str(i) + " "+str(j) +"\n"
    f = open(TOPOLOGY_DIRECTORY+"traffic.txt", "w")
    f.write(traffic_content)
    f.close()
    traffics_f = open(TOPOLOGY_DIRECTORY+'traffic.txt' , 'r' )

for count_line, line in enumerate(traffics_f.readlines()):
    if(count_line == 0):
        NUM_OD_FLOWS = int(line)
    else:
        if count_line <= NUM_OD_FLOWS:
            traffic_props = line.split(" ")
            assert len(traffic_props) == 2, "Errore"
            traffics_base.append(line.rstrip())
            traffics_base.append(line.rstrip())
traffics_f.close()

'''
Functions used to convert the values of capacity and rate from string to int
'''
#assumes Mbps and returns integer as int[Mbps]
def capacity_from_string(string_value):
    return int(string_value.split("Mbps")[0])
#assumes Mbps or Kbps and returns integer as int[Kbps]
def string_to_k(string_value):
    if(string_value.find("Mbps") > 0):
        return int(string_value.split("Mbps")[0]) * 10**3
    else:
        return int(string_value.split("Kbps")[0])

'''
About the links
'''

# create values to draw from for capacity
capacities_10M = np.repeat("10Mbps", 1)
capacities_20M = np.repeat("20Mbps", 2)
capacities_30M = np.repeat("30Mbps", 3)
capacities_40M = np.repeat("40Mbps", 4)
capacities_50M = np.repeat("50Mbps", 4)
capacities_60M = np.repeat("60Mbps", 4)
capacities_70M = np.repeat("70Mbps", 5)
capacities_80M = np.repeat("80Mbps", 5)
capacities_90M = np.repeat("90Mbps", 5)
capacities_100M = np.repeat("100Mbps", 5)
capacities_110M = np.repeat("110Mbps", 4)
capacities_120M = np.repeat("120Mbps", 3)
capacities_130M = np.repeat("130Mbps", 3)
capacities_140M = np.repeat("140Mbps", 3)
capacities_150M = np.repeat("150Mbps", 3)
capacities_160M = np.repeat("160Mbps", 2)
capacities_170M = np.repeat("170Mbps", 2)
capacities_180M = np.repeat("180Mbps", 2)
capacities_190M = np.repeat("190Mbps", 1)
capacities_200M = np.repeat("200Mbps", 1)

# create capacities distribution
capacities = np.concatenate((
    capacities_10M,capacities_20M,capacities_30M,capacities_40M,capacities_50M,capacities_60M,capacities_70M,
    capacities_80M,capacities_90M,capacities_100M,capacities_110M,capacities_120M,capacities_130M,capacities_140M,
    capacities_150M,capacities_160M,capacities_170M,capacities_180M,capacities_190M,capacities_200M)
)

# compute average capacity
capacities_int = []
capacities_int = [capacity_from_string(capacity) for capacity in capacities]

# compute max paths traversing one link to adjust the average rate in case of medium intensity
filename_routing = TOPOLOGY_DIRECTORY+'routing.txt'
df_routing_single = pd.read_csv(filename_routing, sep=" ", header=None,names=build_columns_routing(NUM_NODES),index_col=False)
paths_per_link = []
for i in range(0,NUM_NODES):
    paths_per_link.append([])
    for j in range(0,NUM_NODES):
        # for each link, count number of flows on which we have one
        counter = 0
        for k in range(0,NUM_NODES):
            for l in range(0,NUM_NODES):
                counter += df_routing_single["OD_{}_{} link_{}_{}".format(k,l,i,j)].values[0]
        paths_per_link[i].append(counter)
#paths_per_link_mat = np.matrix(paths_per_link)
max_num_paths_per_link = np.max(paths_per_link)
avg_capacity = np.mean(capacities_int)

#avg_rate: capacity is expressed in Mbps, I want avg_rate in Kbps
avg_rate = (avg_capacity/max_num_paths_per_link) * 10**3

# create queue distribution
queue_lengths = np.array([50,100,150,200,250])
# propagation delay
prop_delays = np.array(["{}ms".format(i) for i in range(10, 101)])

log_results = True

#create packet size distribution
packet_sizes = np.array([int(i * 50) for i in range(8, 16)])
protocols = np.array(["UDP","TCP"])

simulations = []
directory_environment = TOPOLOGY_DIRECTORY+CURRENT_SIM_DIR+"/"
if(not os.path.isdir(directory_environment)):
    pathlib.Path(directory_environment).mkdir(parents=True, exist_ok=True)

#copy files for log purposes related to link, simulation, traffic, routing
copyfile(TOPOLOGY_DIRECTORY+"simulation.txt", directory_environment+"simulation.txt")
copyfile(TOPOLOGY_DIRECTORY+"routing.txt", directory_environment+"routing.txt")
copyfile(TOPOLOGY_DIRECTORY+"traffic.txt", directory_environment+"traffic.txt")
copyfile(TOPOLOGY_DIRECTORY+"links.txt", directory_environment+"links.txt")

links = []      #matrix of NUMBER_OF_SIMULATIONS * NUM_LINKS values having, for each simulation, a list of (prop_delay, capacity, queue_length, 25). Same for different intensities.
OD_flows = []   #matrix of NUMBER_OF_SIMULATIONS * NUM_OD_FLOWS values having, for each simulation, a list of (packet size, protocol). Rate will be drawn in that moment for the specific intensity

for i in range(0, NUMBER_OF_SIMULATIONS):
    links.append([])
    OD_flows.append([])
    # read all links and draw capacity, queue length, prop delay
    for j in range(0,NUM_LINKS):
        capacity = capacities[random.randint(0,len(capacities)-1)]
        queue_length = queue_lengths[random.randint(0,len(queue_lengths)-1)]
        prop_delay = prop_delays[random.randint(0,len(prop_delays)-1)]
        links[i].append((capacity, queue_length, prop_delay, 25))

    for j in range(0,NUM_OD_FLOWS):
        # two flows are generated for each. First one is UDP to ensure some communication will happen...
        packet_size = 400
        protocol = "UDP"
        OD_flows[i].append((packet_size, protocol))
        # ... second one is drawn from a distribution
        packet_size = packet_sizes[random.randint(0,len(packet_sizes)-1)]
        protocol = protocols[random.randint(0,len(protocols)-1)]
        OD_flows[i].append((packet_size, protocol))

'''
About the traffic: I compute the index of the rate considering the capacities and the routing policy.
Then, compute three values of mu (high_mu, medium_mu, low_mu). medium_mu is strictly related to avg_rate,
low_mu = medium_mu + lower_shift_scale * 50Kbps, high_mu = medium_mu + higher_shift_scale * 50Kbps
I assume normal distribution. I will draw from the indices. The values of the rates are in Kbps.
Given I = #intensities provided in input, I compute I different mus, each associated to a normal distribution (mu, sigma) with constant sigma.
Each mu differs from the previous one by (high_mu - low_mu) / I
'''
distribution_sim = []
start_rate = 50
end_rate = 40000
num_values = end_rate/start_rate
rates_entire_domain = np.linspace(start_rate, end_rate, num_values)
rates_domain_in_k = rates_entire_domain
indices_rates = [i for i in range(0, len(rates_domain_in_k))]

std = len(indices_rates)**0.5
#scale = average difference between current intensity and medium in terms of DIFFERENCE, being DIFFERENCE = lower_shift_scale*end_rate/num_values). lower_shift_scale = lower one
lower_shift_scale = -20
higher_shift_scale = +50
ratio = (higher_shift_scale - lower_shift_scale) / NUM_INTENSITIES
min_index_for_intensity = 5 # describes, given intensity i, the coefficient to multiply for given intensity as minimum. I.e., you can't have, for intensity = i, less than min_index_for_intensity * i index. Ensures that intensities are different one from the other

avg_rates = []
for sim in range(NUMBER_OF_SIMULATIONS):
    # fixed a simulation, compute the normalized capacities
    normalized_capacities = []
    for index, link in enumerate(links_base):
        node_1 = int(link.split(" ")[0])
        node_2 = int(link.split(" ")[1])
        flows = paths_per_link[node_1][node_2] * 2
        capacity = capacity_from_string(links[sim][index][0])
        normalized_capacities.append(capacity/flows)
    avg_rates.append(np.min(np.array(normalized_capacities)) * 10**3)
    # print("1", np.mean(np.array(normalized_capacities)))
    # print(avg_capacity/max_num_paths_per_link)
    print(np.min(np.array(normalized_capacities)))
# I chose to use min for all distributions
f = open(directory_environment+"log_input.txt", "w")
log_input_content = "Input command: "
for arg in arguments:
    log_input_content +=arg
log_input_content += "\nAverage rate: {}\nLower shift scale: {}\nHigher shift scale: {}\nRatio: {}\nStart rate: {}\nEnd rate: {}\nStd: {}".format(avg_rates, lower_shift_scale, higher_shift_scale, ratio, start_rate, end_rate, std)
f.write(log_input_content)
f.close()

'''
Prepare array describing different simulations. In details, NUMBER_OF_SIMULATIONS * NUM_INTENSITIES simulations are executed.
NUM_INTENSITIES represents how many traffic distributions we are considering
NUMBER_OF_SIMULATIONS represents how many simulations we execute for each intensity, i.e., how many link properties we draw for each link
'''
for sim in range(NUMBER_OF_SIMULATIONS):
    rates_differences_domain_in_k = [abs(value - avg_rates[sim]) for value in rates_entire_domain]
    index_center = np.argmin(rates_differences_domain_in_k)
    distributions = [] # distributions for a single simulation
    for intensity in range(NUM_INTENSITIES):
        shift_scale = lower_shift_scale + ratio * intensity

        distribution_current_intensity = ss.norm.pdf(indices_rates,loc=max(intensity * min_index_for_intensity, index_center+shift_scale), scale = std)
        distribution_current_intensity = distribution_current_intensity / distribution_current_intensity.sum() #normalize the probabilities so their sum is 1

        distributions.append(('intensity_{}'.format(intensity), distribution_current_intensity, indices_rates, rates_entire_domain, rates_domain_in_k))
    distribution_sim.append(distributions)

if log_results:
    for sim in range(NUMBER_OF_SIMULATIONS):
        distributions = distribution_sim[sim]
        for dist in distributions:
            drawn_indices = np.random.choice(dist[2], 200, p=dist[1])
            mean = np.mean(np.array([dist[4][index] for index in drawn_indices]))
            median = np.argmax(dist[1])
            print(dist[0],mean, median)

# create array of graphs with weights on edges
weighted_graphs = []
outputs_routing = []
list_nodes = list(range(NUM_NODES))
for r in range(NUMBER_OF_ROUTINGS):
    current_weights = []

    G = nx.Graph()
    G.add_nodes_from(range(NUM_NODES))

    for w in weights:
        cur_weight = w['weight'] + (np.random.rand(1)[0]) * 1
        current_weights.append(cur_weight)
        G.add_edge(w['first_node'], w['second_node'], weight=cur_weight)

    weighted_graphs.append(G)
    output_routing = "{}\n".format(NUM_NODES ** 2 - NUM_NODES)

    for first_node, second_node in itertools.product(*[list_nodes, list_nodes]):
        if first_node != second_node:
            solution = nx.dijkstra_path(G,first_node,second_node)
            links_solution = [(s,solution[first_node+1]) for first_node,s in enumerate(solution) if first_node < len(solution) -1]

            weights_sol = []
            for (o,d,w) in weights:
                # plot solution
                weight = 0
                k = 0
                while k < len(links_solution) and weight == 0:
                    (o_s, d_s) = links_solution[k]
                    k += 1
                    if (o == o_s and d == d_s) or (o == d_s and d == o_s):
                        weight = 1
                weights_sol.append(weight)

            # solution into file is a list of routings for each OD flow
            output_routing += "{} {} ".format(first_node, second_node)
            for s, sol in enumerate(links_solution):
                if s == 0:
                    cur_output_routing = "{}-{}".format(sol[0], sol[1])
                else:
                    cur_output_routing = sol[1]
                output_routing += "{}-".format(cur_output_routing)
            output_routing = output_routing[:-1] + "\n"
    outputs_routing.append(output_routing)

# For each simulation I have a different distribution according to the computed capacity
for i in range(0, NUMBER_OF_SIMULATIONS):
    for j, dist in enumerate(distribution_sim[i]):
        dist_type = dist[0]
        links_strings = []
        traffics_strings = []
        links_strings.append(links_base.copy())
        traffics_strings.append(traffics_base.copy())
        #read all links and draw capacity, queue length, prop delay
        print(len(links), [len(_) for _ in links], NUM_LINKS, NUMBER_OF_SIMULATIONS)
        try:
            for k in range(0,NUM_LINKS):
                capacity = links[i][k][0]
                queue_length = links[i][k][1]
                prop_delay = links[i][k][2]
                RED_queue_length = links[i][k][3]
                links_strings[0][k] += " "+prop_delay+" "+capacity+" "+str(queue_length)+" "+str(RED_queue_length)+"\n"
        except:
            print(i,j)
            err
        for k in range(0, NUM_OD_FLOWS * 2):
            packet_size = OD_flows[i][k][0]
            protocol = OD_flows[i][k][1]
            # according to the current scheme, we have that
            if k % 2 == 0:
                min_packet_for_period = 4
                # whenever k % 2 == 0 -> it is UDP and we want to ensure approximately 2 packets to be sent for sure. I set rate = min_packet_for_period * bits / timeframe
                rate_val = str(int(packet_size * 8 * min_packet_for_period / PERIODS_LENGTH))
            else:
                # otherwise draw rate from distribution
                rate_index = np.random.choice(dist[2], 1, p=dist[1])[0]
                rate_val = str(int(dist[4][rate_index]*1000))
            traffics_strings[0][k] += " "+protocol+" "+str(packet_size)+" 0 "+rate_val+"\n"

        PATH_SIMULATION = CURRENT_SIM_DIR+"/"+dist_type+"/simulation_"+str(i)

        for routing in range(NUMBER_OF_ROUTINGS):
            # create environment for same routing
            PATH_ROUTING = PATH_SIMULATION+"/routing_"+str(routing)
            # apply Dijsktra algorithm
            folder_output = TOPOLOGY_DIRECTORY+PATH_ROUTING+"/"
            if(not os.path.isdir(folder_output)):
                #os.mkdir(folder_output)
                pathlib.Path(folder_output).mkdir(parents=True, exist_ok=True)
                #copy simulation.txt
                copyfile(TOPOLOGY_DIRECTORY+"simulation.txt", folder_output+"simulation.txt")
                #write links
                f = open(folder_output+"links.txt", "w")
                links_string=str(NUM_LINKS)+"\n"
                for l in links_strings[0]:
                    links_string+=l
                    f.write(links_string)
                f.close()
                    #write traffic
                f = open(folder_output+"traffic.txt", "w")
                traffic_string=str(NUM_OD_FLOWS * 2)+"\n"
                for t in traffics_strings[0]:
                    traffic_string+=t
                    f.write(traffic_string)
                f.close()
                nx.write_gpickle(folder_output, "weighted_graph.gpickle")
                f = open("{}routing.txt".format(folder_output), "w")
                f.write(outputs_routing[routing])
                f.close()

            simulations.append((dist_type, i, routing))

print("Running {} simulations. Max simultaneously: {}".format(len(simulations), MAX_WORKERS))

# Retrieve a single page and report the URL and contents
def execute_shell(intensity_level, simulation_number, routing_number):
    #draws values that are really high. High frequency or high mean service time?
    process = ['./waf', '--run', 'scratch/simulation_routing --topology='+TOPOLOGY_NAME+' --environment='+CURRENT_SIM_DIR+' --intensity_level='+intensity_level+' --simulation_number='+simulation_number+' --routing_number='+routing_number ]
    string_p = ""
    for p in process:
        string_p += p+" "
    if START_PROCESS:
        print("Running process: "+string_p)
        subprocess.run(process)
    else:
        print("String for running process: "+string_p)

count_errors = 0

# with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Start the simulation marking each with its index
    futures = {executor.submit(execute_shell, intensity, str(simulation), str(routing)): simulation for intensity, simulation, routing in simulations}
    counter = 0
    for future in concurrent.futures.as_completed(futures):
        index = futures[future]
        counter += 1
        try:
            data = future.result()
        except Exception as exc:
            print('Error: %r generated an exception: %s' % (index, exc))
            count_errors += 1
        else:
            if START_PROCESS:
                print('Execution {} of {} is OK'.format(counter, len(simulations)))
    print("Errors: " +str(count_errors)+" among "+str(len(simulations)))
    end = time.process_time()
    print("Time needed for ALL the simulations: ", end - start)
