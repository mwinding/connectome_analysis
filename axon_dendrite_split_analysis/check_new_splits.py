# %%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

# %%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pandas as pd
import numpy as np
import connectome_tools.process_matrix as promat
import connectome_tools.process_skeletons as proskel
import networkx as nx 
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, name, password, token)

# getting required skids
unsplittables = pymaid.get_skids_by_annotation('mw mixed axon/dendrite')
unsplittables_2 = pymaid.get_skids_by_annotation('mw mixed axon/dendrite <0.2 segregation index')
unsplittables_2_skids = np.setdiff1d(unsplittables_2, unsplittables) # old unsplittables that have now been split

# loading neurons from skids
unsplittables_2 = pymaid.get_neurons(unsplittables_2)
# %%
# loading neuron treenode IDs and all connectors
unsplittable2_morph = pymaid.get_treenode_table(unsplittables_2)
unsplittable2_connectors = pymaid.get_connector_details(unsplittables_2)

# %%
# identifying split tags "mw axon split"
tags = unsplittable2_morph['tags']

split_nodes = []
for i in np.arange(0, len(tags), 1):
    if(type(tags[i]) == list):
        if('mw axon split' in tags[i]):
            split_nodes.append(i)
        
unsplittable2_morph.iloc[split_nodes]

# %%

# #####need to start working here

list_skeletons = proskel.split_skeleton_lists(unsplittable2_morph)
list_connectors = proskel.split_skeleton_lists(unsplittable2_connectors)

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

# %%
