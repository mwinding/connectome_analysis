import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math

# import and store skeleton morphology data as graph
graph = proskel.skid_as_networkx_graph('data/test_skeleton.csv')
root = proskel.identify_root(graph)

# import connectors (inputs/outputs)
connectors = pd.read_csv('data/test_connectors.csv', header=0, skipinitialspace=True, keep_default_na = False)

# calculating distance to root for each connector
connector_dist = pd.DataFrame(columns = ['nodeid', 'type', 'distance_root'])

for i in range(0, len(connectors['treenode_id'])):

    if(connectors['relation_id'][i]=="presynaptic_to"):
        dist = proskel.calculate_dist_2nodes(graph, connectors['treenode_id'][i], root)
        connector_dist.append({'nodeid': connectors['treenode_id'][i], 'type': 'presynaptic', 'distance_root': dist}, ignore_index=True)
        print(dist)

    if(connectors['relation_id'][i]=="postsynaptic_to"):
        dist = proskel.calculate_dist_2nodes(graph, connectors['treenode_id'][i], root)
        connector_dist.append({'nodeid': connectors['treenode_id'][i], 'type': 'postsynaptic', 'distance_root': dist}, ignore_index=True)
        print(dist)


#print(connector_dist)


'''
# test case for calculating total distance between two treenode ids
path = nx.bidirectional_shortest_path(graph, 1867733, root)
path = list(pairwise(path))

dist = []
for i in path:
    p1 = np.array([graph.nodes(data=True)[i[0]]['x'], graph.nodes(data=True)[i[0]]['y'], graph.nodes(data=True)[i[0]]['z']])
    p2 = np.array([graph.nodes(data=True)[i[1]]['x'], graph.nodes(data=True)[i[1]]['y'], graph.nodes(data=True)[i[1]]['z']])

    dist.append(np.linalg.norm(p1-p2))
    
total_dist = np.sum(dist)
print("from source: %d to root: %d is %d nm" % (1867733, root, total_dist))
'''

#print(proskel.calculate_dist_2nodes(graph, 1867733, root))
#print(list(pairwise(nx.bidirectional_shortest_path(graph, 1867733, root))))