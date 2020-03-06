import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import connectome_tools.process_skeletons as proskel
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx 


connectors = pd.read_csv('axon_dendrite_data/splittable_connectors_all.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeletons = pd.read_csv('axon_dendrite_data/splittable_skeletons_all.csv', header=0, skipinitialspace=True, keep_default_na = False)
splits = pd.read_csv('axon_dendrite_data/brain_acc_split_nodes.csv', header=0, skipinitialspace=True, keep_default_na = False)

list_skeletons = proskel.split_skeleton_lists(skeletons)
list_connectors = proskel.split_skeleton_lists(connectors)


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

# identify split points
skids = np.unique(skeletons['skeleton_id'])

split_ids = []
for i in tqdm(range(len(skids))):
    split_ids.append(splits['treenode'][splits['skeleton'] == skids[i]].values[0])

# calculate distances of each connector to split for each separate graph
connectdists_list = []
for i in tqdm(range(len(skeleton_graphs))):

    # find shortest path to root
    # if path contains split point, the origin is in the axon
    connectdists = proskel.connector_dists_centered(skeleton_graphs[i], list_connectors[i], roots[i], split_ids[i])
    connectdists_list.append(connectdists)

proskel.write_connectordists('axon_dendrite_data/connectdists_all_centeredsplit.csv', connectdists_list)