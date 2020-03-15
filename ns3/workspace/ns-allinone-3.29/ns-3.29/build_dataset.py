'''
End-to-end Delay Prediction Based on Traffic Matrix Sampling. 
Filip Krasniqi, Jocelyne Elias, Jeremie Leguay, Alessandro E. C. Redondi.
IEEE INFOCOM WKSHPS - NI: The 3rd International Workshop on Network Intelligence. Toronto, July 2020.
'''

import pandas as pd
import numpy as np
import re

import time
import subprocess

import os
from os.path import join, expanduser

from columns import *
import sys

start = time.process_time()

assert len(sys.argv) >= 8, "Errore"

arguments = sys.argv

'''
Variables that define networks, simulation and aggregation. All these (*) are required in order as args,
and they are provided by simulation.cc

PERIODS (*): number of periods P. Coincides with the number of rows for each dataset
NUM_NODES (*): number of nodes N in the network.
NUM_NODES_COMMUNICATING (*): number of nodes NC communicating in the network.
INTENSITY (*): defines the intensity of the traffic inside the network
PERIOD_LENGTH (*): defines how long a period lasts. The simulation lasts for PERIOD_LENGTH*PERIODS
DISTRIBUTION (*): defines the distribution of the traffic. We'll have P defining Poisson. Just a string for the output dataset
PERIOD_LENGTH_NS: PERIOD_LENGTH*(10**9)
ns3_base_directory: directory containing config files for ns3 and on which ns3 created directory and output the tracing
ns3_output_directory: directory containing the outputs from ns3
dataset_output_directory: directory where to output the datasets
'''
log = False

PERIODS = int(arguments[1])
NUM_NODES = int(arguments[2])
NUM_NODES_COMMUNICATING = int(arguments[3])
INTENSITY = int(arguments[4])
PERIOD_LENGTH = float(arguments[5])
DISTRIBUTION = str(arguments[6])
ns3_output_directory = str(arguments[7])
ns3_base_directory = str(arguments[8])
dataset_output_directory = "datasets"

PERIOD_LENGTH_NS = PERIOD_LENGTH*(10**9)
ranges = np.arange(0, PERIODS*PERIOD_LENGTH_NS+PERIOD_LENGTH_NS, PERIOD_LENGTH_NS)

'''
Create all the file paths for input and output
'''
complete_ns3_base_directory = ns3_base_directory
folder_input = complete_ns3_base_directory+ns3_output_directory+"/"
folder_output = complete_ns3_base_directory+dataset_output_directory+"/"
if(not os.path.isdir(folder_output)):
    os.mkdir(folder_output)
dropped_filename = folder_input+"dropped.tr"
delay_tx_filename = folder_input+"enqueue.tr"
delay_rx_filename = folder_input+"dequeue.tr"
traffic_tx_filename = folder_input+"tx.tr"
traffic_rx_filename = folder_input+"rx.tr"

routing_filename = folder_input+"routing.routes"
addresses_filename = folder_input+"addresses.tr"
link_filename = complete_ns3_base_directory+"links.txt"

filename_pl = '{}_{}_{}_{}_{}_delaypl.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_e2e = '{}_{}_{}_{}_{}_delaye2e.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_load = '{}_{}_{}_{}_{}_load.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)

filename_TM = '{}_{}_{}_{}_{}_TM.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_links = '{}_{}_{}_{}_{}_links.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_routing = '{}_{}_{}_{}_{}_routing.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_dropped = '{}_{}_{}_{}_{}_dropped.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_packet_info_global = '{}_{}_{}_{}_{}_packet_info.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_packet_info_link = '{}_{}_{}_{}_{}_packet_info_link.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)
filename_packet_info_od = '{}_{}_{}_{}_{}_packet_info_od.txt'.format(NUM_NODES,NUM_NODES_COMMUNICATING, INTENSITY, PERIODS, DISTRIBUTION)

# targets
complete_path_load = folder_output+filename_load
complete_path_delay_e2e = folder_output+filename_e2e
complete_path_delay_pl = folder_output+filename_pl

# fixed per simulation: one row
complete_path_links = folder_output+filename_links
complete_path_routing = folder_output+filename_routing

# infos on packets
complete_path_packet_info_link = folder_output+filename_packet_info_link
complete_path_packet_info_od = folder_output+filename_packet_info_od
complete_path_packet_info_global = folder_output+filename_packet_info_global

# P rows per simulation
complete_path_TM = folder_output+filename_TM
complete_path_dropped = folder_output+filename_dropped

# variable to access the correct value given string for link capacity
bit_UoM = [
    ["bps",10**0],
    ["Kbps",10**3],
    ["Mbps",10**6],
    ["Gbps",10**9]
]

# read tx
df_tx = pd.read_csv(traffic_tx_filename, index_col=False)                       # X
# read rx
df_rx = pd.read_csv(traffic_rx_filename, index_col=False)                       # X
# read dropped
df_dropped = pd.read_csv(dropped_filename, index_col=False)                     # X
# read enqueuing
df_enqueuing = pd.read_csv(delay_tx_filename, index_col=False)                  # X
# read dequeuing
df_dequeuing = pd.read_csv(delay_rx_filename, index_col=False)                  # X

df_enq_deq_uid = pd.merge(df_enqueuing,df_dequeuing, on=['UID'])                # X

# add O and D columns to all dataframes
df_tx['ORIGIN'] = [int(item.split('_')[0]) for i,item in enumerate(df_tx['O_D'].values)]
df_tx['DESTINATION'] = [int(item.split('_')[1]) for i,item in enumerate(df_tx['O_D'].values)]

df_rx['ORIGIN'] = [int(item.split('_')[0]) for i,item in enumerate(df_rx['O_D'].values)]
df_rx['DESTINATION'] = [int(item.split('_')[1]) for i,item in enumerate(df_rx['O_D'].values)]

df_tx_rx_uid = pd.merge(df_tx,df_rx, on=['UID'])                                # X
df_tx_OD = df_tx[df_tx.ORIGIN == df_tx.TX]                                      # X

del df_tx
del df_rx

df_enqueuing['ORIGIN'] = [int(item.split('_')[0]) for i,item in enumerate(df_enqueuing['O_D'].values)]
df_enqueuing['DESTINATION'] = [int(item.split('_')[1]) for i,item in enumerate(df_enqueuing['O_D'].values)]

df_dequeuing['ORIGIN'] = [int(item.split('_')[0]) for i,item in enumerate(df_dequeuing['O_D'].values)]
df_dequeuing['DESTINATION'] = [int(item.split('_')[1]) for i,item in enumerate(df_dequeuing['O_D'].values)]

# init dataframes that need to consider only instances with O/D as Tx/RX node. Useful for those dataframes related to traffic measurements
df_dequeuing_OD = df_dequeuing[df_dequeuing.DESTINATION == df_dequeuing.RX]     # X
df_enqueuing_OD = df_enqueuing[df_enqueuing.ORIGIN == df_enqueuing.TX]          # X

del df_enqueuing
del df_dequeuing

if log:
    print("INFO: Dropped shape: ",df_dropped.shape)
    print("INFO: Enqueuing instances related to origin has shape: ", df_enqueuing_OD.shape)
    print("INFO: Dequeuing instances related to destination has shape: ",df_dequeuing_OD.shape)

# filter enqueuing by removing instances that don't fulfill the receiving
difference_enqueuing_dequeuing=pd.concat([df_enqueuing_OD, df_dequeuing_OD, df_dequeuing_OD], sort=False).drop_duplicates(subset='PID',keep=False)                          # X
df_enqueuing_OD_received=pd.concat([df_enqueuing_OD, difference_enqueuing_dequeuing, difference_enqueuing_dequeuing], sort=False).drop_duplicates(subset='PID',keep=False)  # X

del difference_enqueuing_dequeuing

df_enqueuing_OD_received.drop(columns=['END','SIZE'], inplace = True)

if log:
    print("INFO: Enqueuing that during an OD flow were received has shape: ",df_enqueuing_OD_received.shape)

# filter removing dropped packets for those measurements that must not consider drops
df_enqueuing_OD_no_drops = pd.concat([df_enqueuing_OD_received, df_dropped, df_dropped], sort=False).drop_duplicates(subset='PID',keep=False)
del df_enqueuing_OD_received
df_enqueuing_OD_no_drops.drop(columns=['TIME'], inplace = True)
df_dequeuing_OD_no_drops=pd.concat([df_dequeuing_OD, df_dropped, df_dropped], sort=False).drop_duplicates(subset='PID',keep=False)

del df_dequeuing_OD

df_dequeuing_OD_no_drops.drop(columns=['TIME'], inplace = True)

if log:
    print("INFO: Dequeuing that during an OD flow did not drop (should be the same) has shape: ",df_dequeuing_OD_no_drops.shape)
    print("INFO: Enqueuing that during an OD flow did not drop (should be the same) has shape: ",df_enqueuing_OD_no_drops.shape)

df_enq_deq_pid = pd.merge(df_enqueuing_OD_no_drops, df_dequeuing_OD_no_drops, on=['PID'])
del df_enqueuing_OD_no_drops
del df_dequeuing_OD_no_drops

reading_time = time.process_time()
delta = reading_time - start
print("INFO: reading values required {} seconds ".format(delta))


'''
Start to build datasets. I already built the filtered dataframes for each case.
1) Case of multiple measures for each simulation
I always do the following steps:
- add columns for making the computation (eg: delay, bins, packets)
- aggregation depending on how the computation must be done
    - bins: always necessary. I create a column every time I need to aggregate,
            on which I put, given the time of measurement, the related interval.
            The number of intervals is PERIODS, and I already built in ranges all
            the values, built considering PERIOD_LENGTH_NS and PERIODS
    - link: used when every instance is associated to a link
    - O_D: used when every instance is associated to an OD
- computation of the value: depending on what the measure consists on, I need to do some computation.
    - for dropped packets, I need to count the rows. So, I add a column packets = 1 and sum after the aggregation
    - for the load, I need to sum all the sent bytes from the origin and divide it by the period length
    - for the delay, regardless from the fact that it is PL or e2e (the aggregation will be different),
        I need to sum the difference between START and END, that is related to enqueuing/dequeuing
    - for the traffic matrix, I do the same as delay, but it is associated to different data
2) Single measure for each simulation: I parse the files accordingly and build the dataframe of a single row
'''

'''
LOAD
'''

'''
Merge datasets of Tx and Rx based on UID. Given tx and rx, merging via UID allows to find the pair sent-received
'''

df_tx_rx_uid['delay'] = df_tx_rx_uid['END'] - df_tx_rx_uid['START']
df_tx_rx_uid['packets'] = 1

if df_tx_rx_uid.shape[0] > 0:
    df_tx_rx_uid['link'] = df_tx_rx_uid.apply(lambda row: str(row['TX_x']) + '_' + str(row['RX_x']), axis=1)
else:
    df_tx_rx_uid['link'] = ""
df_tx_rx_uid['bins'] = pd.cut(df_tx_rx_uid["START"], ranges, False)

all_links = df_tx_rx_uid.link.unique()
all_ODs = df_tx_rx_uid.O_D_x.unique()

aggregated_df = df_tx_rx_uid.groupby(['link','bins'])
del df_tx_rx_uid
keys_of_aggregation = aggregated_df.groups.keys()
total_num_packets = 0
global_size_packets = 0
total_num_packets_OD = {}
global_size_packets_OD = {}
total_num_packets_link = {}
global_size_packets_link = {}

for l in all_links:
    total_num_packets_link[l] = global_size_packets_link[l] = 0
for od in all_ODs:
    total_num_packets_OD[od] = global_size_packets_OD[od] = 0

columns = build_columns_only_load(NUM_NODES)
dataset = {}
for col in columns:
    dataset[col] = []
for r in ranges:
    interval = pd.Interval(left=r, right=r+PERIOD_LENGTH_NS, closed='left')
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):
            link = str(i)+"_"+str(j)
            key = (link, interval)
            if key not in keys_of_aggregation:
                load = 0
            else:
                values = aggregated_df.get_group(key).sum()
                values['load'] = values['SIZE_x'] * 8 / PERIOD_LENGTH  # expressed in bit/s
                load = values['load']
                total_num_packets += values['packets']
                global_size_packets += values['SIZE_x'] * 8 #in bits

                total_num_packets_link[link] += values['packets']
                global_size_packets_link[link] += values['SIZE_x'] * 8 #in bits

            dataset['load_{}_{}'.format(i,j)].append(load)

avg_size_packet = global_size_packets/total_num_packets
df_load = pd.DataFrame(data=dataset)
df_load = df_load.round(5)
load_time = time.process_time()
delta = load_time-reading_time
print("INFO: load dataset required {} seconds ".format(delta))
print("INFO: Total number of packets: {}, Total size of packets: {}, Avg. size of a packet in bit: {} ".format(total_num_packets, global_size_packets, avg_size_packet))

'''
DELAY E2E
'''

'''
Merge datasets of enqueuing and dequeuing based on PID. Given enq and deq, merging via PID allows to find the pair origin-destination
'''
df_enq_deq_pid['delay'] = df_enq_deq_pid['END'] - df_enq_deq_pid['START']
df_enq_deq_pid['packets'] = 1
df_enq_deq_pid['bins'] = pd.cut(df_enq_deq_pid["START"], ranges, False)

aggregated_df = df_enq_deq_pid.groupby(['O_D_x','bins'])
del df_enq_deq_pid
keys_of_aggregation = aggregated_df.groups.keys()

columns = build_columns_only_delay_e2e(NUM_NODES)
dataset = {}
missing_delays_e2e = 0
for col in columns:
    dataset[col] = []
for r in ranges:
    interval = pd.Interval(left=r, right=r+PERIOD_LENGTH_NS, closed='left')
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):
            OD_pair = str(i)+"_"+str(j)
            key = (OD_pair,interval)
            if key not in keys_of_aggregation:
                delay_per_bit = 0
                delay_per_packet = 0
                if i != j:
                    missing_delays_e2e += 1
            else:
                values = aggregated_df.get_group(key).sum()
                values['delay_per_bit'] = values['delay'] / (values['SIZE']*8)
                delay_per_bit = values['delay_per_bit'] # not considered anymore
                if values['packets'] > 0:
                    delay_per_packet = values['delay'] / values['packets']
                else:
                    delay_per_packet = 0
                    if i != j:
                        missing_delays_e2e += 1

                total_num_packets_OD[OD_pair] += values['packets']
                global_size_packets_OD[OD_pair] += values['SIZE'] * 8 #in bits
            dataset['delay_e2e_{}_{}'.format(i,j)].append(delay_per_packet)

df_delay_e2e = pd.DataFrame(data=dataset)
df_delay_e2e = df_delay_e2e.round(5)
delaye2e_time = time.process_time()
delta = delaye2e_time-load_time
print("INFO: delay e2e dataset required {} seconds".format(delta))

'''
DELAY PL
'''

'''
Merge datasets of enqueuing and dequeuing based on UID. Given enq and deq, merging via PID allows to find the pair sent-received
'''
df_enq_deq_uid['delay'] = df_enq_deq_uid['END'] - df_enq_deq_uid['START']
df_enq_deq_uid['packets'] = 1
if df_enq_deq_uid.shape[0] > 0:
    df_enq_deq_uid['link'] = df_enq_deq_uid.apply(lambda row: str(row['TX_x']) + '_' + str(row['RX_x']), axis=1)
else:
    df_enq_deq_uid['link'] = ""
df_enq_deq_uid['bins'] = pd.cut(df_enq_deq_uid["START"], ranges, False)

aggregated_df = df_enq_deq_uid.groupby(['link','bins'])
del df_enq_deq_uid
keys_of_aggregation = aggregated_df.groups.keys()

columns = build_columns_only_delay_pl(NUM_NODES)
dataset = {}
missing_delays_pl = 0
for col in columns:
    dataset[col] = []
for r in ranges:
    interval = pd.Interval(left=r, right=r+PERIOD_LENGTH_NS, closed='left')
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):
            link = str(i)+"_"+str(j)
            key = (link, interval)
            if key not in keys_of_aggregation:
                delay = 0
                delay_per_packet = 0
                missing_delays_pl += 1
            else:
                values = aggregated_df.get_group(key).sum()
                values['delay_per_bit'] = values['delay'] / (values['SIZE']*8)
                delay = values['delay_per_bit']

                if values['packets'] > 0:
                    delay_per_packet = values['delay'] / values['packets']
                else:
                    delay_per_packet = 0
                    missing_delays_pl += 1

            dataset['delay_pl_{}_{}'.format(i,j)].append(delay_per_packet)

df_delay_pl = pd.DataFrame(data=dataset)
df_delay_pl = df_delay_pl.round(5)
delaypl_time = time.process_time()
delta = delaypl_time-delaye2e_time
print("INFO: delay pl required {} seconds".format(delta))

'''
TRAFFIC MATRIX
'''

'''
Only using the dataframe related to tx that contain origin. No consideraiton on drops.
'''

df_tx_OD['bins'] = pd.cut(df_tx_OD["START"], ranges, False)
aggregated_df = df_tx_OD.groupby(['O_D','bins'])
del df_tx_OD
keys_of_aggregation = aggregated_df.groups.keys()

columns = build_columns_only_traffic(NUM_NODES)
dataset = {}
for col in columns:
    dataset[col] = []
for r in ranges:
    interval = pd.Interval(left=r, right=r+PERIOD_LENGTH_NS, closed='left')
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):
            OD_pair = str(i)+"_"+str(j)
            key = (OD_pair,interval)
            if key not in keys_of_aggregation:
                traffic = 0
            else:
                values = aggregated_df.get_group(key).sum()
                values['traffic'] = values['SIZE'] * 8 / PERIOD_LENGTH  # expressed in bit/s
                traffic = values['traffic']
            dataset['traffic_{}_{}'.format(i,j)].append(traffic)

df_TM = pd.DataFrame(data=dataset)
df_TM = df_TM.round(5)
traffic_time = time.process_time()
delta = traffic_time- delaypl_time
print("INFO: traffic matrix dataset required {} seconds".format(delta))

'''
DROPPED
'''

'''
Dropped packets. Directly consider the drops
'''

if df_dropped.shape[0] > 0:
    df_dropped['link'] = df_dropped.apply(lambda row: str(row['TX']) + '_' + str(row['RX']), axis=1)
else:
    df_dropped['link'] = ""
df_dropped['bins'] = pd.cut(df_dropped["TIME"], ranges, False)
df_dropped['packets'] = 1

aggregated_df = df_dropped.groupby(['link','bins'])
del df_dropped
keys_of_aggregation = aggregated_df.groups.keys()

columns = build_columns_only_dropped(NUM_NODES)
dataset = {}
for col in columns:
    dataset[col] = []
for r in ranges:
    interval = pd.Interval(left=r, right=r+PERIOD_LENGTH_NS, closed='left')
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):
            link = str(i)+"_"+str(j)
            key = (link, interval)
            if key not in keys_of_aggregation:
                dropped = 0
            else:
                values = aggregated_df.get_group(key).sum()
                dropped = values['packets']
            dataset['dropped_{}_{}'.format(i,j)].append(dropped)

df_drop = pd.DataFrame(data=dataset)
df_drop = df_drop.round(5)
drop_time = time.process_time()
delta = drop_time - traffic_time
print("INFO: dropped packets dataset required {} seconds".format(delta))

'''
ROUTING POLICY
'''

# init address values. Resulting addresses array: list of N lists.
# Each sublist at i-th position contains the addresses of i-th node
addresses = []
addresses = [[] for i in range(0,NUM_NODES)]
current_node = 0
with open(addresses_filename, 'r') as content_file:
    for l in content_file:
        values = l.split(" ")
        for v in values:
            if not(v.isdigit() or v == "\n"):#case of address
                addresses[current_node].append(v)
        current_node += 1

'''
Read from file the routing policy
assumption: either I have 0.0.0.0, or I have the correct node address, i.e., either there is 0.0.0.0 or there is next hop for each node. No forwarding to another network.
so, when I have not only 0.0.0.0, I have the next hop (=gateway) that refers to a node, not to a network.
in our case it is like this. Otherwise I should also check the subnet mask.
result of this block: pairs_destination_nexthop list, that is, for each node a list of pairs that contains all the (destination, gateway) for that node.

Additional notes about the parsing: each node is defined with two blocks, divided by \n. I am interested
in the second one, so when current_block %2 == 0 I read lines until I find \n. In that case,
I increase current_block and set current_row_for_node = 0, and increase current_node and I enter in current_block %2 == 1.
In this case I am considering node of index current_node and I am interested in the lines starting from the third (if(current_row_for_node > 3))
Then, I starting appending [destination, gateway]
'''
current_node = 0
current_block = 0
current_row_for_node = 0
pairs_destination_nexthop = []
pairs_destination_nexthop = [[] for i in range(0,NUM_NODES)]
with open(routing_filename, 'r') as content_file:
    for l in content_file:
        # file is
        if current_block %2 == 1 and l != "\n":
            current_row_for_node += 1
            if(current_row_for_node > 3):
                l = re.sub('\s+', ' ', l).strip()#remove multilines
                values = l.split(" ")
                destination = values[0]
                gateway = values[1]
                pairs_destination_nexthop[current_node].append([destination,gateway])
        if l == "\n":
            current_block += 1
            if current_block %2 == 0:
                current_node += 1
                current_row_for_node = 0
'''
Input: array pairs_destination_nexthop, containing at index i the pairs (destination, gateway) for node with ID i
Output: pair_destination_nexthop_id, that is a list of N lists, being N = number of nodes.
The i-th list contains the hops as in pairs_destination_nexthop but for the IDs. Plus,
- in case of 0.0.0.0, the sublist will contain in every position the default gateway
- otherwise it contains the index of the next hop (no more the address)
Result: not only now I will have the ID instead of the address, but plus my list is complete, i.e., node i will contain the next hop for all destinations except itself (set to -1)
'''
pair_destination_nexthop_id = []
pair_destination_nexthop_id = [[] for i in range(0,NUM_NODES)]
i = 0
for pairs in pairs_destination_nexthop:
    # pairs[i] contains the pair (destination, gateway) for the destination at position i
    # pairs[i][0] contains the pair destination, pairs[i][1] contains the gateway
    if len(pairs) == 1 and pairs[0][0] == '0.0.0.0':  #thats the destination
        #Goal of the loop: find index that has an array with same value of gateway addresses
        next_hop_id = -1
        index = 0
        while next_hop_id < 0 and index < len(addresses):
            addresses_of_node = addresses[index]
            if pairs[0][1] in addresses_of_node:
                next_hop_id = index
            else:
                index += 1
        # for each nodeId as a destination, I will go for next hop. Particular case because I have the default gateway
        pair_destination_nexthop_id[i] = [next_hop_id for i in range(0,NUM_NODES)]
        # I set to -1 the diagonal
        pair_destination_nexthop_id[i][i] = -1
    else:
        #look for i s.t. pairs[i][0] is your destination
        origin_addresses = addresses[i]
        #set everything to -1. In the end I will have only the diagonal
        pair_destination_nexthop_id[i] = [-1 for i in range(0,NUM_NODES)]
        for j in range(0,len(pairs)):
            address_current_destination = pairs[j][0]
            address_current_nexthop = pairs[j][1]
            '''
            Find id for that destination. This is the only difference from the
            previous case, as here we need to search for it, while previously
            we had the default gateway and so we updated all the values related to that i.
            First while fulfills this search; second while does the same as above
            '''
            destination_id = -1
            index = 0
            while destination_id < 0 and index < len(addresses):
                addresses_of_node = addresses[index]
                if address_current_destination in addresses_of_node:
                    destination_id = index
                else:
                    index += 1
            next_hop_id = -1
            index = 0
            while next_hop_id < 0 and index < len(addresses):
                addresses_of_node = addresses[index]
                if address_current_nexthop in addresses_of_node:
                    next_hop_id = index
                else:
                    index += 1
            #check whether for that I already searched i.e. the index is already > 0. I assume a deterministic routing policy
            if(pair_destination_nexthop_id[i][destination_id] < 0):
                pair_destination_nexthop_id[i][destination_id] = next_hop_id
    i += 1
'''
Given pair_destination_nexthop_id, I have a list of N sublists. Being k = pair_destination_nexthop_id[i][j], it means that
the next hop for OD flow (i,j) is (i,k), so I iteratively follow the list to obtain the path
Briefly speaking, I do the same operations the router does, but with nodeIds
'''
routing_policy = [0 for i in range(0,NUM_NODES**4)]
for i in range(0,NUM_NODES):
    for j in range(0,NUM_NODES):
        # I am in an OD flow
        if i != j:#if not, let it be 0
            # look for the routing table related to the i-th node
            current_node = i
            destination_node = j
            path_length = 0
            # I search until the destination is reached
            while(path_length <= 0 or destination_node != current_node):
                next_hop = pair_destination_nexthop_id[current_node][destination_node]
                flat_index = i*NUM_NODES**3+j*NUM_NODES**2+current_node*NUM_NODES+next_hop
                routing_policy[flat_index] = 1
                current_node = next_hop
                path_length += 1

log_routing = False
if(log_routing):
    for i in range(0,NUM_NODES):
        for j in range(0,NUM_NODES):
            sliced = routing_policy[i*NUM_NODES**3+j*NUM_NODES**2:i*NUM_NODES**3+j*NUM_NODES**2+NUM_NODES**2]
            print("INFO: OD flow {}-{} : {} {}".format(i,j,sliced,len(sliced)))

routing_time = time.process_time()
delta = routing_time - drop_time
print("INFO: routing policy dataset required {} seconds".format(delta))

'''
Fill link capacity and queue from the files. I put 0 to the non existing links,
and the capacity/queue value for the link
'''
links_capacity = [0 for i in range(0,NUM_NODES**2)]
links_queue = [0 for i in range(0,NUM_NODES**2)]
with open(link_filename, 'r') as content_file:
    links = int(content_file.readline())
    for i in range(0,links):
        link = content_file.readline().split(" ")
        first_node = int(link[0])
        second_node = int(link[1])
        links_capacity[first_node*NUM_NODES+second_node] = link[3]
        links_queue[first_node*NUM_NODES+second_node] = link[4]
links_queue = list(map(int, links_queue))
for i, capacity in enumerate(links_capacity):
    if(capacity != 0):
        value_in_bits = 0
        index=0
        occurence = -1
        while(value_in_bits<=0 and index<len(bit_UoM)):
            occurence = capacity.find(bit_UoM[index][0])
            if occurence > 0:
                value_in_bits = bit_UoM[index][1]
                if (capacity.split(bit_UoM[index][0])[0]).isnumeric():
                    value_in_bits = int(value_in_bits)
                else:
                    occurence = -1
                    value_in_bits = 0
                    index += 1
            else:
                index += 1
        if occurence > 0:
            links_capacity[i] = value_in_bits*int(capacity.split(bit_UoM[index][0])[0])
        else:
            links_capacity[i] = 0

capacity_time = time.process_time()
delta = capacity_time - routing_time
print("INFO: link dataset (capacity and queue) required {} seconds".format(delta))

'''
Create dataframes containing the link capacities and the queues. Only one row
'''
df_c = pd.DataFrame(data=[links_capacity],columns=build_columns_capacity(NUM_NODES))
df_q = pd.DataFrame(data=[list(map(int, links_queue))],columns=build_columns_queue(NUM_NODES))
df_link_single = pd.concat([df_c,df_q],axis=1, sort=False,join='inner')
'''
Create dataframe for routing. Only one row
'''
df_routing_single = pd.DataFrame(data=[routing_policy],columns=build_columns_routing(NUM_NODES))

print("SUCCESS: writing final datasets!")

df_load.to_csv(path_or_buf=complete_path_load, sep=' ',index=False, header=False)
df_delay_e2e.to_csv(path_or_buf=complete_path_delay_e2e, sep=' ',index=False, header=False)
df_delay_pl.to_csv(path_or_buf=complete_path_delay_pl, sep=' ',index=False, header=False)

df_link_single.to_csv(path_or_buf=complete_path_links, sep=' ',index=False, header=False)
df_routing_single.to_csv(path_or_buf=complete_path_routing, sep=' ',index=False, header=False)

f = open(complete_path_packet_info_global, "w")
f.write("{} {} {} {} {}".format(total_num_packets, global_size_packets, avg_size_packet, missing_delays_e2e, missing_delays_pl))
f.close()

f = open(complete_path_packet_info_link, "w")
for l in total_num_packets_link.keys():
    f.write("{} {} {} {}\n".format(l, total_num_packets_link[l], global_size_packets_link[l], global_size_packets_link[l]/total_num_packets_link[l]))
f.close()

f = open(complete_path_packet_info_od, "w")
for od in total_num_packets_OD.keys():
    f.write("{} {} {} {}\n".format(od, total_num_packets_OD[od], global_size_packets_OD[od], global_size_packets_OD[od]/total_num_packets_OD[od]))
f.close()

df_TM.to_csv(path_or_buf=complete_path_TM, sep=' ',index=False, header=False)
df_drop.to_csv(path_or_buf=complete_path_dropped, sep=' ',index=False, header=False)

# print for log purposes all the files
print("SUCCESS: here are the links for each dataset\n")
print("Links properties: ",complete_path_links)
print("Routing: ",complete_path_routing)
print("Traffic matrix: ",complete_path_TM)
print("Dropped: ",complete_path_dropped)

print("LOAD: ",complete_path_load)
print("e2e delay: ",complete_path_delay_e2e)
print("per link delay: {}\n".format(complete_path_delay_pl))

# clear files
delete_all_files = True
# set to true to avoid deleting particular simulations
skip_delete_exceptions = False
simulation_except_number = [3]
intensity_except_number = [0, 9]

current_sim = int(ns3_base_directory.split("/")[-2].split("_")[1])
current_intensity = int(ns3_base_directory.split("/")[-3].split("_")[1])

if skip_delete_exceptions:
    if current_sim in simulation_except_number and current_intensity in intensity_except_number:
        delete_all_files = False
        print("WARNING: skipping delete for I = {}, S = {}".format(current_intensity, current_sim))

if delete_all_files:
    process = ['rm', folder_input+'dequeue.tr', folder_input+'enqueue.tr', folder_input+'rx.tr', folder_input+'tx.tr', folder_input+'dropped.tr']
    string_p = "Executing "
    for p in process:
        string_p += p+" "
    print(string_p+"\n")
    subprocess.run(process)

end = time.process_time()
print("INFO: Time needed for building dataset: ", end - start)
