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
unsplittables_2 = pymaid.get_neurons(unsplittables_2_skids)

# loading neuron treenode IDs and all connectors
unsplittable2_morph = pymaid.get_treenode_table(unsplittables_2)
unsplittable2_connectors = pymaid.get_connector_links(unsplittables_2)


# %%
skeletons = unsplittable2_morph
connectors = unsplittable2_connectors

list_skeletons = proskel.split_skeleton_lists(skeletons)
list_connectors = proskel.split_skeleton_lists(connectors)

# identifying split points with tag "mw axon split"
tags = skeletons['tags']

split_nodes = []
for i in np.arange(0, len(tags), 1):
    if(type(tags[i]) == list):
        if('mw axon split' in tags[i]):
            split_nodes.append(i)
        
splitnode_data = skeletons.iloc[split_nodes]

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


print(len(skeleton_graphs))
print(len(ordered_list_connectors))
print(len(roots))
print(len(split_ids))

# calculate distances of each connector to split for each separate graph
connectdists_list = []
for i in tqdm(range(len(skeleton_graphs))):

    # find shortest path to root
    # if path contains split point, the origin is in the axon
    connectdists = proskel.connector_dists_centered(skeleton_graphs[i], ordered_list_connectors[i], roots[i], split_ids[i])
    connectdists_list.append(pd.DataFrame(connectdists))

connectdists = pd.concat(connectdists_list, axis=0)

splittable_inputs = connectdists[connectdists['type'] == 'postsynaptic']['distance']
splittable_outputs = connectdists[connectdists['type'] == 'presynaptic']['distance']
#%%
# plotting as histogram distribution

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(splittable_inputs, ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(splittable_outputs, ax = ax, hist = False, kde_kws = {'shade': True})
#sns.distplot(splittable_inputs, ax = ax)
#sns.distplot(splittable_outputs, ax = ax)


ax.set(xlim = (-125000, 200000))
plt.axvline(x=0, color = 'gray')
ax.set_yticks([])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150, 200])
ax.set_ylabel('Synapse Density')
ax.set_xlabel('Distance (in um)')    

plt.savefig('axon_dendrite_split_analysis/plots/new_splittables_less_than020_segIndex.pdf', format='pdf', bbox_inches = 'tight')

# %%
# plotting as raster plot
import matplotlib as ml

inputs_array = []
outputs_array = []
inputs_norm_array = []
outputs_norm_array = []

for i in range(len(connectdists_list)):
    connectdist = connectdists_list[i]
    output_index = connectdist['type']=='presynaptic'
    input_index = connectdist['type']=='postsynaptic'

    outputs = np.array(connectdist[output_index]['distance'])
    inputs = np.array(connectdist[input_index]['distance'])

    both = np.concatenate([outputs, inputs])
    min_both = min(both)
    max_both = max(both)

    norm_outputs = outputs
    norm_inputs = inputs

    for j in range(len(outputs)):
        if(outputs[j]>=0):
            norm_outputs[j] = norm_outputs[j]/max_both
        if(outputs[j]<0):
            norm_outputs[j] = -norm_outputs[j]/min_both

    for j in range(len(inputs)):
        if(inputs[j]>=0):
            norm_inputs[j] = norm_inputs[j]/max_both
        if(inputs[j]<0):
            norm_inputs[j] = -norm_inputs[j]/min_both

    inputs_array.append(inputs)
    outputs_array.append(outputs)
    inputs_norm_array.append(norm_inputs)
    outputs_norm_array.append(norm_outputs)

lineoffsets = np.arange(0, len(inputs_array), 1)


fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.set(xlim = (-1, 1))
ax.eventplot(inputs_norm_array, lineoffsets = lineoffsets, alpha = 0.5)
ax.eventplot(outputs_norm_array, lineoffsets = lineoffsets, color = 'orange', alpha = 0.5)


# %%
