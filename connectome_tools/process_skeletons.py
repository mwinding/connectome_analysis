import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import csv

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
            connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'presynaptic', 'distance': dist})

        if(connectors['relation_id'].iloc[i]=="postsynaptic_to"):
            dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
            connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'postsynaptic', 'distance': dist})

    return(connector_dist)  

def connector_dists_centered(G, connectors, root, split):

    connector_dist = []

    for i in range(0, len(connectors['treenode_id'])):

        path = nx.bidirectional_shortest_path(G, connectors['treenode_id'].iloc[i], root)

        if(connectors['relation_id'].iloc[i]=="presynaptic_to"):
            if(np.isin(split, path)):
                dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
                connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'presynaptic', 'distance': dist})
            if(~np.isin(split, path)):
                dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
                connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'presynaptic', 'distance': -dist})

        if(connectors['relation_id'].iloc[i]=="postsynaptic_to"):
            if(np.isin(split, path)):
                dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
                connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'postsynaptic', 'distance': dist})
            if(~np.isin(split, path)):
                dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
                connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'postsynaptic', 'distance': -dist})

    return(connector_dist) 

def split_skeleton_lists(connector_list):
    unique_skeletons = np.unique(connector_list['skeleton_id'].values)

    skelmorph_list = []
    for i in unique_skeletons:
        indices = np.where(connector_list['skeleton_id'].values == i)
        skelmorph_list.append(connector_list.iloc[indices[0]])

    return(skelmorph_list)

def connector_dist_batch(path_skeletons, path_connectors, output_path_raw, output_path_norm):
    # import and split skeletons into separate entry of list
    skeletons_csv = pd.read_csv(path_skeletons, header=0, skipinitialspace=True, keep_default_na = False)
    list_skeletons = split_skeleton_lists(skeletons_csv)

    # convert each skeleton morphology CSV into networkx graph
    skeleton_graphs = []
    for i in tqdm(range(len(list_skeletons))):
        skeleton_graph = skid_as_networkx_graph(list_skeletons[i])
        skeleton_graphs.append(skeleton_graph)
    
    # identify roots of each networkx graph
    roots = []
    for i in tqdm(range(len(skeleton_graphs))):
        root = identify_root(skeleton_graphs[i])
        #print("skeleton %i has root %i" %(skeleton[], root))
        roots.append(root)

    connectors = pd.read_csv(path_connectors, header=0, skipinitialspace=True, keep_default_na = False)
    list_connectors = split_skeleton_lists(connectors)

    # calculate distances of each connector to root for each separate graph
    connectdists_list = []
    for i in tqdm(range(len(skeleton_graphs))):
        connectdists = connector_dists(skeleton_graphs[i], list_connectors[i], roots[i])
        connectdists_list.append(connectdists)

    write_connectordists(output_path_raw, connectdists_list)

    # normalizing neuron lengths
    connectdists_list_norm = connectdists_list
    for i in tqdm(range(len(connectdists_list_norm))):
        dists = []
        for j in range(len(connectdists_list_norm[i])):
            dist = connectdists_list_norm[i][j]['distance']
            dists.append(dist)
        dist_max = max(dists)
        #dist_mean = np.mean(dists)
        #dist_var = np.var(dists)

        for j in range(len(connectdists_list_norm[i])):
            connectdists_list_norm[i][j]['distance'] = (connectdists_list_norm[i][j]['distance'])/dist_max

    write_connectordists(output_path_norm, connectdists_list_norm)
'''
    # normalizing neuron lengths
    connectdists_list_norm = connectdists_list
    for i in tqdm(range(len(connectdists_list_norm))):
        dists = []
        for j in range(len(connectdists_list_norm[i])):
            dist = connectdists_list_norm[i][j]['distance']
            dists.append(dist)
        dist_max = max(dists)
        dist_mean = np.mean(dists)
        dist_var = np.var(dists)

        for j in range(len(connectdists_list_norm[i])):
            connectdists_list_norm[i][j]['distance'] = (connectdists_list_norm[i][j]['distance']-dist_mean)/dist_var

    write_connectordists(output_path_norm2, connectdists_list_norm)
'''

def write_connectordists(path, connectdists_list):
    with open(path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['skeletonid', 'nodeid', 'type', 'distance'])
        for i in range(len(connectdists_list)):
            for j in range(len(connectdists_list[i])):
                skeletonid = connectdists_list[i][j]['skeletonid']
                nodeid = connectdists_list[i][j]['nodeid']
                typ = connectdists_list[i][j]['type']
                distance_root = connectdists_list[i][j]['distance']
                csv_writer.writerow([skeletonid, nodeid, typ, distance_root])
