import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
from tqdm import tqdm
import csv

# import and split skeletons into separate entry of list
skeletons_csv = pd.read_csv('axon_dendrite_split_analysis/splittable_skeletons_left1.csv', header=0, skipinitialspace=True, keep_default_na = False)
list_skeletons = proskel.split_skeleton_lists(skeletons_csv)

# convert each skeleton morphology CSV into networkx graph
skeleton_graphs = []
for i in tqdm(range(len(list_skeletons))):
    skeleton_graph = proskel.skid_as_networkx_graph(list_skeletons[i])
    skeleton_graphs.append(skeleton_graph)
    
# identify roots of each networkx graph
roots = []
for i in tqdm(range(len(skeleton_graphs))):
    root = proskel.identify_root(skeleton_graphs[i])
    #print("skeleton %i has root %i" %(skeleton[], root))
    roots.append(root)
    

connectors = pd.read_csv('axon_dendrite_split_analysis/splittable_connectors_left1.csv', header=0, skipinitialspace=True, keep_default_na = False)
list_connectors = proskel.split_skeleton_lists(connectors)

# calculate distances of each connector to root for each separate graph
connectdists_list = []
for i in tqdm(range(len(skeleton_graphs))):
    connectdists = proskel.connector_dists(skeleton_graphs[i], list_connectors[i], roots[i])
    connectdists_list.append(connectdists)


# normalizing neuron lengths
connectdists_list_norm = connectdists_list
for i in tqdm(range(len(connectdists_list_norm))):
    dists = []
    for j in range(len(connectdists_list_norm[i])):
        dist = connectdists_list_norm[i][j]['distance_root']
        dists.append(dist)
    dist_max = max(dists)
    dist_mean = np.mean(dists)
    dist_var = np.var(dists)

    for j in range(len(connectdists_list_norm[i])):
        connectdists_list_norm[i][j]['distance_root'] = (connectdists_list_norm[i][j]['distance_root'])/dist_max
'''
# normalizing neuron lengths
connectdists_list_norm2 = connectdists_list
for i in tqdm(range(len(connectdists_list_norm2))):
    dists = []
    for j in range(len(connectdists_list_norm2[i])):
        dist = connectdists_list_norm2[i][j]['distance_root']
        dists.append(dist)
    dist_max = max(dists)
    dist_mean = np.mean(dists)
    dist_var = np.var(dists)

    for j in range(len(connectdists_list_norm2[i])):
        connectdists_list_norm2[i][j]['distance_root'] = (connectdists_list_norm2[i][j]['distance_root']-dist_mean)/dist_var
'''


with open('axon_dendrite_split_analysis/splittable_connectdists_left1_raw.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['nodeid', 'type', 'distance_root'])
    for i in range(len(connectdists_list)):
        for j in range(len(connectdists_list[i])):
            nodeid = connectdists_list[i][j]['nodeid']
            typ = connectdists_list[i][j]['type']
            distance_root = connectdists_list[i][j]['distance_root']
            csv_writer.writerow([nodeid, typ, distance_root])

with open('axon_dendrite_split_analysis/splittable_connectdists_left1_norm.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['nodeid', 'type', 'distance_root'])
    for i in range(len(connectdists_list_norm)):
        for j in range(len(connectdists_list_norm[i])):
            nodeid = connectdists_list_norm[i][j]['nodeid']
            typ = connectdists_list_norm[i][j]['type']
            distance_root = connectdists_list_norm[i][j]['distance_root']
            csv_writer.writerow([nodeid, typ, distance_root])
'''
with open('axon_dendrite_split_analysis/splittable_connectdists_left1_norm2.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['nodeid', 'type', 'distance_root'])
    for i in range(len(connectdists_list_norm2)):
        for j in range(len(connectdists_list_norm2[i])):
            nodeid = connectdists_list_norm2[i][j]['nodeid']
            typ = connectdists_list_norm2[i][j]['type']
            distance_root = connectdists_list_norm2[i][j]['distance_root']
            csv_writer.writerow([nodeid, typ, distance_root])
'''
