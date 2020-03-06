import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import connectome_tools.process_skeletons as proskel
import pandas as pd
from tqdm import tqdm
import numpy as np


connectors = pd.read_csv('axon_dendrite_data/connectors_test.csv', header=0, skipinitialspace=True, keep_default_na = False)
skeletons = pd.read_csv('axon_dendrite_data/skeletons_test.csv', header=0, skipinitialspace=True, keep_default_na = False)
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
    roots.append(root)

'''
# calculate distances of each connector to root for each separate graph
connectdists_list = []
for i in tqdm(range(len(skeleton_graphs))):
    connectdists = proskel.connector_dists(skeleton_graphs[i], list_connectors[i], roots[i])
    connectdists_list.append(connectdists)

#print(connectdists_list)
'''

# identify split points
skids = np.unique(skeletons['skeleton_id'])

split_ids = []
for i in tqdm(range(len(skids))):
    split_ids.append(splits['treenode'][splits['skeleton'] == skids[i]].values[0])

#print(split_ids)

# calculate distance from roots to split points
splitdists_list = []
for i in tqdm(range(len(split_ids))):
    splitdist = proskel.calculate_dist_2nodes(skeleton_graphs[i], split_ids[i], roots[i])
    splitdists_list.append(splitdist)


connector_dists = pd.read_csv('axon_dendrite_data/testdists_raw.csv', header=0, skipinitialspace=True, keep_default_na = False)


skids = np.unique(connector_dists['skeletonid'])
print(connector_dists)
for i in tqdm(range(len(splitdists_list))):
    for j in range(len(connector_dists['skeletonid'])):
        if(connector_dists['skeletonid'][j] == skids[i]):
            connector_dists.iat[j, 3] = (connector_dists['distance'][j] - splitdists_list[i])

#connector_dists.to_csv('axon_dendrite_data/testdists_centeredsplit.csv')

