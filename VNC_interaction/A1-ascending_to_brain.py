#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rm = pymaid.CatmaidInstance(url, name, password, token)
adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-09-22.csv', header = 0) # import pairs

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
br = pymaid.get_skids_by_annotation('mw brain neurons')

A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

VNC_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')

# %%
# paths from each A1 ascending neuron
from tqdm import tqdm

threshold = 0.01
hops = 3

A1_ascending_pairs = Promat.extract_pairs_from_list(A1_ascending, pairs)[0]

ascending_pair_paths = []
for index in tqdm(range(0, len(A1_ascending_pairs))):
    ds_ascending = VNC_adj.downstream_multihop(list(A1_ascending_pairs.loc[index]), threshold, min_members = 0, hops=hops)
    ascending_pair_paths.append(ds_ascending)

# %%

all_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, br)
dVNC_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, dVNC)
predVNC_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, pre_dVNC)

valid_ascendings = ~(all_type_layers.iloc[:, 0]==0)
all_type_layers = all_type_layers[valid_ascendings] # remove all ascendings with no strong brain connections
dVNC_type_layers = dVNC_type_layers[valid_ascendings]
predVNC_type_layers = predVNC_type_layers[valid_ascendings]

fig, axs = plt.subplots(
    1, 3, figsize=(6, 4)
)

ax = axs[0]
annotations = all_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(all_type_layers, annot = annotations, fmt = 's', cmap = 'Greens', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('Individual A1 Ascendings')
ax.set(title='Pathway Overview')

ax = axs[1]
annotations = dVNC_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(dVNC_type_layers, annot = annotations, fmt = 's', cmap = 'Reds', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='dVNC')

ax = axs[2]
annotations = predVNC_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(predVNC_type_layers, annot = annotations, fmt = 's', cmap = 'Blues', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='pre-dVNC')

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_ascending_paths_hops-{hops}.pdf', bbox_inches='tight')
# %%
# ascending paths upstream of particular dVNC paths

# setting up paramters for downstream_multihop from dVNC to A1
adj_A1 = pd.read_csv('VNC_interaction/data/axon-dendrite.csv', header = 0, index_col = 0)
inputs_A1 = pd.read_csv('VNC_interaction/data/input_counts.csv', index_col = 0)
inputs_A1 = pd.DataFrame(inputs_A1.values, index = inputs_A1.index, columns = ['axon_input', 'dendrite_input'])

A1_adj = Adjacency_matrix(adj_A1.values, adj_A1.index, pairs, inputs_A1,'axo-dendritic')

threshold = 0.01

source_dVNC, ds_dVNC = A1_adj.downstream(dVNC, threshold, exclude=dVNC)
edges, ds_dVNC_cleaned = A1_adj.edge_threshold(source_dVNC, ds_dVNC, threshold, direction='downstream')
edges[edges.overthres==True]

source_dVNC_cleaned = np.unique(edges[edges.overthres==True].upstream_pair_id)
source_dVNC_pairs = A1_adj.adj_inter.loc[(slice(None), source_dVNC_cleaned), :].index
source_dVNC_pairs = [x[2] for x in source_dVNC_pairs]
source_dVNC_pairs = Promat.extract_pairs_from_list(source_dVNC_pairs, pairs)[0]

source_dVNC_pair_paths = []
for index in tqdm(range(0, len(source_dVNC_pairs))):
    ds_dVNC = A1_adj.downstream_multihop(list(source_dVNC_pairs.loc[index]), threshold, min_members = 0, hops=5)
    source_dVNC_pair_paths.append(ds_dVNC)

# identifying ascending neurons of interest
order = [16, 0, 2, 11, 8, 1, 3, 5, 7, 12, 13, 9, 10, 15, 4, 6, 14]

ascending_layers,ascending_skids = A1_adj.layer_id(source_dVNC_pair_paths, source_dVNC_pairs.leftid, A1_ascending)
ascending_layers = ascending_layers.iloc[order, :]
ascending_skids = ascending_skids.T.iloc[order, :]

ascending_skids_allhops = []
for index in ascending_skids.index:
    skids_allhops = [x for sublist in ascending_skids.loc[index].values for x in sublist if x!='']
    ascending_skids_allhops.append(skids_allhops)

# running downstream_multihop of A1 ascendings into brain
ascendings_paths = []
for index in tqdm(range(0, len(ascending_skids_allhops))):
    ds_dVNC = VNC_adj.downstream_multihop(ascending_skids_allhops[index], threshold, min_members = 0, hops=5)
    ascendings_paths.append(ds_dVNC)

# identify neuron types
all_layers,_ = VNC_adj.layer_id(ascendings_paths, range(0, len(ascendings_paths)), A1_adj.adj.index) # include all neurons to get total number of neurons per layer
dVNC_layers,_ = VNC_adj.layer_id(ascendings_paths, range(0, len(ascendings_paths)), dVNC) 
pre_dVNC_layers,_ = VNC_adj.layer_id(ascendings_paths, range(0, len(ascendings_paths)), pre-dVNC) 


# %%
