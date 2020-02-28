import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
from tqdm import tqdm
import csv
'''
skeleton_csv = pd.read_csv('data/test_skeleton.csv', header=0, skipinitialspace=True, keep_default_na = False)
graph = proskel.skid_as_networkx_graph(skeleton_csv)

graphs = []
graphs.append(graph)
graphs.append(graph)

print(graph.nodes(data=True))
print(graphs)
'''

# import and split skeletons into separate entry of list
skeletons_csv = pd.read_csv('data/test_skeleton_group.csv', header=0, skipinitialspace=True, keep_default_na = False)
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
    roots.append(root)
    

connectors = pd.read_csv('data/test_connectors_group.csv', header=0, skipinitialspace=True, keep_default_na = False)
list_connectors = proskel.split_skeleton_lists(connectors)

# calculate distances of each connector to root for each separate graph
connectdists_list = []
for i in tqdm(range(len(skeleton_graphs))):
    connectdists = proskel.connector_dists(skeleton_graphs[i], list_connectors[i], roots[i])
    connectdists_list.append(connectdists)
    
    
#connectdists_list = pd.DataFrame(connectdists_list)
#connectdists_list.to_csv('outputs/connectdist.csv')

#print(connectdists_list[0])
#print(connectdists_list[1])


with open('outputs/connectdists.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(connectdists_list)):
        for j in range(len(connectdists_list[i])):
            nodeid = connectdists_list[i][j]['nodeid']
            typ = connectdists_list[i][j]['type']
            distance_root = connectdists_list[i][j]['distance_root']
            csv_writer.writerow([nodeid, typ, distance_root])

