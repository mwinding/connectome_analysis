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
