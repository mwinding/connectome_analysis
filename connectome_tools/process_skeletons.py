import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import csv
import pymaid
import navis

# split axons and dendrites and return neurons
def split_axons_dendrites(list_neurons, split):

    # convert to list if given single skid so it will run
    if(type(list_neurons)!=list):
        list_neurons = [list_neurons]

    axons = []
    dendrites = []
    for neuron in list_neurons:
        #identify split
        if(split in neuron.tags.keys()):
            if(len(neuron.tags[split])==1):
                split_id = neuron.tags[split][0]
                axon, dendrite = navis.cut_neuron(neuron, split_id)
                axons.append(axon)
                dendrites.append(dendrite)
            if(len(neuron.tags[split])>1):
                print(f'{neuron.id}: more than one split tag!')
        if(split not in neuron.tags.keys()):
            print(f'{neuron.id}: no split tag!!')
        
    return(axons, dendrites)

# pull connector groups after axon_dendrite split
def get_connectors_group(skids, output_separate=False):

    # convert to list if given single skid so it will run
    if(type(skids)!=list):
        skids = [skids]

    neurons = []
    for skid in skids:
        neurons.append(pymaid.get_neuron(skid))
    axons, dendrites = split_axons_dendrites(neurons, 'mw axon split')

    axon_outputs = []
    dendrite_inputs = []
    for axon, dendrite in zip(axons, dendrites):
        axon_outputs.append(axon.connectors[axon.connectors['type']==0])
        dendrite_inputs.append(dendrite.connectors[dendrite.connectors['type']==1])

    if(output_separate):
        return(axon_outputs, dendrite_inputs)
    if(output_separate==False):
        return(pd.concat(axon_outputs, axis=0), pd.concat(dendrite_inputs, axis=0))

def axon_dendrite_centroid(skid):

    axon_outputs, dendrite_inputs = get_connectors_group(skid, output_separate=True)
    try:
        axon_outputs = axon_outputs[0]
        dendrite_inputs = dendrite_inputs[0]

        # middle of commissure; less than 50500 is right, greater/equal is left
        commissure_x = 50500
        
        # collect a few annotations to deal with some edge cases
        bilateral_axon = pymaid.get_skids_by_annotation('mw bilateral axon')
        ipsi_dendrite = pymaid.get_skids_by_annotation('mw ipsilateral dendrite')
        bilateral_dendrite = pymaid.get_skids_by_annotation('mw bilateral dendrite')
        left = pymaid.get_skids_by_annotation('mw left')
        right = pymaid.get_skids_by_annotation('mw right')

        # normal ipsilateral axons/ipsilateral dendrites and contralateral axons/ipsilaateral dendrites
        axon_x = axon_outputs.mean(axis=0).x/1000
        axon_y = axon_outputs.mean(axis=0).y/1000
        axon_z = axon_outputs.mean(axis=0).z/1000

        dendrite_x = dendrite_inputs.mean(axis=0).x/1000
        dendrite_y = dendrite_inputs.mean(axis=0).y/1000
        dendrite_z = dendrite_inputs.mean(axis=0).z/1000

        # identify and deal with edge-cases
        if(skid in bilateral_axon):
            if(skid in ipsi_dendrite):

                dendrite_x = dendrite_inputs.mean(axis=0).x/1000
                dendrite_y = dendrite_inputs.mean(axis=0).y/1000
                dendrite_z = dendrite_inputs.mean(axis=0).z/1000

                if(skid in left):
                    axon_x = axon_outputs[axon_outputs.x>=commissure_x].mean(axis=0).x/1000
                    axon_y = axon_outputs[axon_outputs.x>=commissure_x].mean(axis=0).x/1000
                    axon_z = axon_outputs[axon_outputs.x>=commissure_x].mean(axis=0).x/1000

                if(skid in right):
                    axon_x = axon_outputs[axon_outputs.x<commissure_x].mean(axis=0).x/1000
                    axon_y = axon_outputs[axon_outputs.x<commissure_x].mean(axis=0).x/1000
                    axon_z = axon_outputs[axon_outputs.x<commissure_x].mean(axis=0).x/1000

            if(skid in bilateral_dendrite):

                dendrite_x_left = dendrite_inputs[dendrite_inputs.x>=commissure_x].mean(axis=0).x/1000
                dendrite_y_left = dendrite_inputs[dendrite_inputs.x>=commissure_x].mean(axis=0).y/1000
                dendrite_z_left = dendrite_inputs[dendrite_inputs.x>=commissure_x].mean(axis=0).z/1000
                axon_x_left = axon_outputs[axon_outputs.x>=commissure_x].mean(axis=0).x/1000
                axon_y_left = axon_outputs[axon_outputs.x>=commissure_x].mean(axis=0).x/1000
                axon_z_left = axon_outputs[axon_outputs.x>=commissure_x].mean(axis=0).x/1000

                dendrite_x_right = dendrite_inputs[dendrite_inputs.x<commissure_x].mean(axis=0).x/1000
                dendrite_y_right = dendrite_inputs[dendrite_inputs.x<commissure_x].mean(axis=0).y/1000
                dendrite_z_right = dendrite_inputs[dendrite_inputs.x<commissure_x].mean(axis=0).z/1000
                axon_x_right = axon_outputs[axon_outputs.x<commissure_x].mean(axis=0).x/1000
                axon_y_right = axon_outputs[axon_outputs.x<commissure_x].mean(axis=0).x/1000
                axon_z_right = axon_outputs[axon_outputs.x<commissure_x].mean(axis=0).x/1000

                dendrite_x = (dendrite_x_left + dendrite_x_right)/2
                dendrite_y = (dendrite_y_left + dendrite_y_right)/2
                dendrite_z = (dendrite_z_left + dendrite_z_right)/2

                axon_x = (axon_x_left + axon_x_right)/2
                axon_y = (axon_y_left + axon_y_right)/2
                axon_z = (axon_z_left + axon_z_right)/2

        distance = ((dendrite_x-axon_x)**2 + (dendrite_y-axon_y)**2 + (dendrite_z-axon_z)**2)**(1/2)
        centroids = pd.DataFrame([[skid, (axon_x, axon_y, axon_z), (dendrite_x, dendrite_y, dendrite_z), distance]], columns = ['skid', 'axon_centroids', 'dendrite_centroids', 'distance'])
        return(centroids)
        
    except:
        print('issue') 

# import skeleton CSV (of single skeleton) in CATMAID export skeleton format
def skid_as_networkx_graph(skeleton):

    G = nx.Graph()

    # adding nodes from child-parent structure of skeleton to networkx object
    # x,y,z attributes are spatial coordinates in nm
    # identifies rootnode, whose parent id is None (empty cell)
    for i in range(0, len(skeleton['treenode_id'])):
        if (type(skeleton['parent_node_id'].iloc[i]) is not int):
            G.add_node(skeleton['treenode_id'].iloc[i], x = skeleton['x'].iloc[i], y = skeleton['y'].iloc[i], z = skeleton['z'].iloc[i], root = True)

        if (type(skeleton['parent_node_id'].iloc[i]) is int):
            G.add_node(skeleton['treenode_id'].iloc[i], x = skeleton['x'].iloc[i], y = skeleton['y'].iloc[i], z = skeleton['z'].iloc[i], root = False)
            
    # adding edges from child-parent structure of skeleton to networkx object
    # also identifies the rootnode, whose parent id is None (empty cell)
    for i in range(0, len(skeleton['treenode_id'])):
        if (type(skeleton['parent_node_id'].iloc[i]) is int):
            G.add_edge(int(skeleton['treenode_id'].iloc[i]), int(skeleton['parent_node_id'].iloc[i]))

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

        if(connectors['relation'].iloc[i]=="presynaptic_to"):
            if(np.isin(split, path)):
                dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
                connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'presynaptic', 'distance': dist})
            if(~np.isin(split, path)):
                dist = calculate_dist_2nodes(G, connectors['treenode_id'].iloc[i], root)
                connector_dist.append({'skeletonid': connectors['skeleton_id'].iloc[i], 'nodeid': connectors['treenode_id'].iloc[i], 'type': 'presynaptic', 'distance': -dist})

        if(connectors['relation'].iloc[i]=="postsynaptic_to"):
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

def dist_from_split(skids, split_tag):

    skeletons = pymaid.get_treenode_table(skids)

    # identifying split points with tag "mw axon split"
    tags = skeletons['tags']

    split_nodes = []
    for i in np.arange(0, len(tags), 1):
        if(type(tags[i]) == list):
            if(split_tag in tags[i]):
                split_nodes.append(i)
            
    splitnode_data = skeletons.iloc[split_nodes]

    # check if all skeletons have the split tag
    def membership(list1, list2):
        set1 = set(list1)
        return [item in set1 for item in list2]

    member_index = membership(splitnode_data['skeleton_id'], skeletons['skeleton_id'])
    skeletons = skeletons[member_index]

    # load connectors of skids with split tag
    skids = np.unique(skeletons['skeleton_id'])
    neurons = pymaid.get_neurons(skids)
    connectors = pymaid.get_connector_links(neurons)

    list_skeletons = split_skeleton_lists(skeletons)
    list_connectors = split_skeleton_lists(connectors)

    # convert each skeleton morphology CSV into networkx graph
    skeleton_graphs = []
    for i in tqdm(range(len(list_skeletons))):
        skeleton_graph = skid_as_networkx_graph(list_skeletons[i])
        skeleton_graphs.append(skeleton_graph)

    # identify roots of each networkx graph
    roots = []
    for i in tqdm(range(len(skeleton_graphs))):
        root = identify_root(skeleton_graphs[i])
        roots.append(root)

    # order split points like skeleton ids
    split_ids = []
    for i in tqdm(range(len(list_skeletons))):
        for j in range(len(splitnode_data)):
            if(int(splitnode_data['skeleton_id'].iloc[j])==int(list_skeletons[i]['skeleton_id'].iloc[0])):
                split_ids.append(splitnode_data['treenode_id'].iloc[j])
                break # !! ignores multiple axon/dendrite splits (temporary)

    # order connector list like skeleton ids
    ordered_list_connectors = []
    for i in tqdm(range(len(list_skeletons))):
        for j in (range(len(list_connectors))):
            if(list_connectors[j]['skeleton_id'].iloc[0]==int(list_skeletons[i]['skeleton_id'].iloc[0])):
                ordered_list_connectors.append(list_connectors[j])

    # calculate distances of each connector to split for each separate graph
    connectdists_list = []
    for i in tqdm(range(len(skeleton_graphs))):

        # find shortest path to root
        # if path contains split point, the origin is in the axon
        connectdists = connector_dists_centered(skeleton_graphs[i], ordered_list_connectors[i], roots[i], split_ids[i])
        connectdists_list.append(pd.DataFrame(connectdists))

    return(connectdists_list)
