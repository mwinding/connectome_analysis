import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import math

# import skeleton CSV (of single skeleton) in CATMAID export skeleton format
def skid_as_networkx_graph(skeleton):

    G = nx.Graph()

    # adding nodes from child-parent structure of skeleton to networkx object
    # x,y,z attributes are spatial coordinates in nm
    # identifies rootnode, whose parent id is an empty cell or '' (due to import style)
    for i in range(0, len(skeleton['treenode_id'])):
        if(skeleton['parent_treenode_id'].iloc[i]==''):
            G.add_node(skeleton['treenode_id'].iloc[i], x = skeleton['x'].iloc[i], y = skeleton['y'].iloc[i], z = skeleton['z'].iloc[i], root = True)

        if not(skeleton['parent_treenode_id'].iloc[i]==''):
            G.add_node(skeleton['treenode_id'].iloc[i], x = skeleton['x'].iloc[i], y = skeleton['y'].iloc[i], z = skeleton['z'].iloc[i], root = False)
            

    # adding edges from child-parent structure of skeleton to networkx object
    # also identifies the rootnode, whose parent id is an empty cell or '' (due to import style)
    for i in range(0, len(skeleton['treenode_id'])):
        if not(skeleton['parent_treenode_id'].iloc[i]==''):
            G.add_edge(int(skeleton['treenode_id'].iloc[i]), int(skeleton['parent_treenode_id'].iloc[i]))

    return(G)

# find rootnode id
# G is a treenode graph
def identify_root(G):
    root=nx.get_node_attributes(G,'root')

    rootnode = []
    for i in root:
        if(root[i]==True):
            rootnode = i

    return(rootnode)

def longest_dist(G):
    dist = []
    return(dist)

# identify distance between two nodes in a treenode graph
# G is a treenode graph
def calculate_dist_2nodes(G, source, target):
    path = nx.bidirectional_shortest_path(G, source, target)
    path = list(pairwise(path))

    dist = []
    for i in path:
        p1 = np.array([G.nodes(data=True)[i[0]]['x'], G.nodes(data=True)[i[0]]['y'], G.nodes(data=True)[i[0]]['z']])
        p2 = np.array([G.nodes(data=True)[i[1]]['x'], G.nodes(data=True)[i[1]]['y'], G.nodes(data=True)[i[1]]['z']])

        dist.append(np.linalg.norm(p1-p2))

    total_dist = np.sum(dist)

    return(total_dist)


# calculating distance to root for each connector
def connector_dists(G, connectors, root):

    connector_dist = []

    for i in range(0, len(connectors['treenode_id'])):

        if(connectors['relation_id'].iloc[i]=="presynaptic_to"):
            dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
            connector_dist.append({'nodeid': connectors['treenode_id'].iloc[i], 'type': 'presynaptic', 'distance_root': dist})

        if(connectors['relation_id'].iloc[i]=="postsynaptic_to"):
            dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
            connector_dist.append({'nodeid': connectors['treenode_id'].iloc[i], 'type': 'postsynaptic', 'distance_root': dist})

    return(connector_dist)

def split_skeleton_lists(connector_list):
    unique_skeletons = np.unique(connector_list['skeleton_id'].values)

    skelmorph_list = []
    for i in unique_skeletons:
        indices = np.where(connector_list['skeleton_id'].values == i)
        skelmorph_list.append(connector_list.iloc[indices[0]])

    return(skelmorph_list)
