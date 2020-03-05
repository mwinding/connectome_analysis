import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import connectome_tools.process_skeletons as proskel
import pandas as pd
from tqdm import tqdm
import numpy as np


connectors = pd.read_csv('axon_dendrite_data/splittable_connectors_all.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeletons = pd.read_csv('axon_dendrite_data/splittable_skeletons_all.csv', header=0, skipinitialspace=True, keep_default_na = False)
splits = pd.read_csv('axon_dendrite_data/brain_acc_split_nodes.csv', header=0, skipinitialspace=True, keep_default_na = False)

list_skeletons = proskel.split_skeleton_lists(skeletons)
list_connectors = proskel.split_skeleton_lists(connectors)


 # convert each skeleton morphology CSV into networkx graph
skeleton_graphs = []
for i in range(len(list_skeletons)):
    skeleton_graph = proskel.skid_as_networkx_graph(list_skeletons[i])
    skeleton_graphs.append(skeleton_graph)

# identify roots of each networkx graph
roots = []
for i in tqdm(range(len(skeleton_graphs))):
    root = proskel.identify_root(skeleton_graphs[i])
    #print("skeleton %i has root %i" %(skeleton[], root))
    roots.append(root)

# calculate distances of each connector to root for each separate graph
connectdists_list = []
for i in tqdm(range(len(skeleton_graphs))):
    connectdists = proskel.connector_dists(skeleton_graphs[i], list_connectors[i], roots[i])
    connectdists_list.append(connectdists)


# identify split points
skids = np.unique(skeletons['skeleton_id'])

split_ids = []
for i in tqdm(range(len(skids))):
    split_ids.append(splits['treenode'][splits['skeleton'] == skids[i]])

# calculate distance from roots to split points
splitdists_list = []
for i in tqdm(range(len(split_ids))):
    splitdist = proskel.connector_dists(skeleton_graphs[i], list_connectors[i], split_ids[i])
    splitdists_list.append(splitdist)

# subtract connector distances from split points
for i in tqdm(range(len(connectdists_list))):
    splitdist = proskel.connector_dists(skeleton_graphs[i], list_connectors[i], split_ids[i])
    splitdists_list.append(splitdist)

proskel.write_connectordists('axon_dendrite_data/dist_to_splitnodes.csv', connectdists_list)

