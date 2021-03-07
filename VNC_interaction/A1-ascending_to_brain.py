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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

rm = pymaid.CatmaidInstance(url, token, name, password)
adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RGN')
dVNC_to_A1 = pymaid.get_skids_by_annotation('mw dVNC to A1')
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

A1_ascending = A1_ascending + Promat.get_paired_skids(2511238, pairs) # added A00c A4 as well
A1_ascending_pairs = Promat.extract_pairs_from_list(A1_ascending, pairs)[0]

ascending_pair_paths = []
for index in tqdm(range(0, len(A1_ascending_pairs))):
    ds_ascending = VNC_adj.downstream_multihop(list(A1_ascending_pairs.loc[index]), threshold, min_members = 0, hops=hops)
    ascending_pair_paths.append(ds_ascending)

#A00c_path = VNC_adj.downstream_multihop(Promat.get_paired_skids(2511238, pairs), threshold, min_members=0, hops=hops) # added A00c A4 in (connectivity with A1 sensories/basins)
#ascending_pair_paths.append(A00c_path)

# %%
# plotting ascending paths

all_type_layers,all_type_layers_skids = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, br)
dVNC_A1_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, dVNC_to_A1)
dVNC_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, dVNC)
predVNC_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, pre_dVNC)
dSEZ_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, dSEZ)
RGN_type_layers,_ = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, RGN)

valid_ascendings = ~(all_type_layers.iloc[:, 0]==0)
all_type_layers = all_type_layers[valid_ascendings] # remove all ascendings with no strong brain connections
dVNC_A1_type_layers = dVNC_A1_type_layers[valid_ascendings]
dVNC_type_layers = dVNC_type_layers[valid_ascendings]
predVNC_type_layers = predVNC_type_layers[valid_ascendings]
dSEZ_type_layers = dSEZ_type_layers[valid_ascendings]
RGN_type_layers = RGN_type_layers[valid_ascendings]

fig, axs = plt.subplots(
    1, 4, figsize=(6, 4)
)

ax = axs[0]
annotations = all_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(all_type_layers, annot = annotations, fmt = 's', cmap = 'Greens', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('Individual A1 Ascendings')
ax.set(title='Pathway')

ax = axs[1]
annotations = dVNC_A1_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(dVNC_A1_type_layers, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='dVNC to A1')

ax = axs[2]
annotations = dVNC_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(dVNC_type_layers, annot = annotations, fmt = 's', cmap = 'Reds', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='dVNC')

ax = axs[3]
annotations = predVNC_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(predVNC_type_layers, annot = annotations, fmt = 's', cmap = 'Blues', ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='pre-dVNC')

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_ascending_paths_hops-{hops}.pdf', bbox_inches='tight')

# %%
# plot by individual ascending
plt.rcParams['font.size'] = 5

# split plot types by ascending pair

asc_pairs_order = [21250110, 4206755, 4595609, 2511238, 8059283, # direct to dVNCs
                3571478, 10929797, 3220616, 10949382, 8057753, 2123422, 4555763, 7766016] # indirect to dVNCs

layer_types = [all_type_layers, dVNC_type_layers, dVNC_A1_type_layers, dSEZ_type_layers, RGN_type_layers]
col = ['Greens', 'Reds', 'Oranges', 'Purples', 'GnBu']
#asc_pairs = all_type_layers.index
#layer_types = [all_type_layers, dVNC_type_layers, dSEZ_type_layers, RGN_type_layers]
#col = ['Greens', 'Reds', 'Purples', 'Oranges']

asc_list = []
for pair in asc_pairs_order:
    mat = np.zeros(shape=(len(layer_types), len(all_type_layers.columns)))
    for i, layer_type in enumerate(layer_types):
        mat[i, :] = layer_type.loc[pair]

    asc_list.append(mat)

# loop through pairs to plot
for i, asc in enumerate(asc_list):

    data = pd.DataFrame(asc, index = ['Total', 'dVNC', 'dVNC-A1', 'dSEZ', 'RGN'])
    #data = pd.DataFrame(asc, index = ['Total', 'dVNC', 'dSEZ', 'RGN']).iloc[:, 0:2]
    mask_list = []
    for i_iter in range(0, len(data.index)):
        mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
        mask[i_iter, :] = [False]*len(data.columns)
        mask_list.append(mask)

    fig, axs = plt.subplots(
        1, 1, figsize=(.5, .6)
    )
    for j, mask in enumerate(mask_list):
        if((j == 0)):
            vmax = 500
        if((j == 2)):
            vmax = 30
        if((j in [1,3,4])):
            vmax = 60
        ax = axs
        annotations = data.astype(int).astype(str)
        annotations[annotations=='0']=''
        sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

    plt.savefig(f'VNC_interaction/plots/individual_asc_paths/{i}_asc-{asc_pairs_order[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')

plt.rcParams['font.size'] = 6

#%%
# summary plots for main figure

dVNC_direct = [21250110, 4206755, 4595609, 2511238, 8059283]
dVNC_indirect = [3571478, 10929797, 3220616, 10949382, 8057753, 2123422, 4555763, 7766016]

asc_types_name = ['dVNC-direct', 'dVNC-indirect']

asc_types = [dVNC_direct, dVNC_indirect]
asc_types = [[Promat.get_paired_skids(x, pairs) for x in sublist] for sublist in asc_types] # convert leftid's to both skids from each pair
asc_types = [sum(x, []) for x in asc_types] # unlist nested lists

# multihop downstream
type_paths = []
for index in tqdm(range(0, len(asc_types))):
    ds_asc = VNC_adj.downstream_multihop(list(asc_types[index]), threshold, min_members = 0, hops=hops)
    type_paths.append(ds_asc)

#%%
# summary plot continued

# identifying different cell types in ascending pathways
all_type_layers_types,all_type_layers_skids_types = VNC_adj.layer_id(type_paths, asc_types_name, br) # include all neurons to get total number of neurons per layer
dVNC_type_layers_types,_ = VNC_adj.layer_id(type_paths, asc_types_name, dVNC) 
dVNC_A1_type_layers_types,_ = VNC_adj.layer_id(type_paths, asc_types_name, dVNC_to_A1)
dSEZ_type_layers_types,_ = VNC_adj.layer_id(type_paths, asc_types_name, dSEZ)
RGN_type_layers_types,_ = VNC_adj.layer_id(type_paths, asc_types_name, RGN)

# split plot types by dVNC pair
layer_types_types = [all_type_layers_types, dVNC_type_layers_types, dVNC_A1_type_layers_types, dSEZ_type_layers_types, RGN_type_layers_types]
col = ['Greens', 'Reds', 'Oranges', 'Purples', 'GnBu']

asc_list_types = []
for types in asc_types_name:
    mat = np.zeros(shape=(len(layer_types_types), len(all_type_layers_types.columns)))
    for i, layer_type in enumerate(layer_types_types):
        mat[i, :] = layer_type.loc[types]

    asc_list_types.append(mat)

plt.rcParams['font.size'] = 5

# loop through pairs to plot
for i, asc in enumerate(asc_list_types):

    data = pd.DataFrame(asc, index = ['Total', 'dVNC', 'dVNC-A1', 'dSEZ', 'RGN'])
    #data = pd.DataFrame(asc, index = ['Total', 'dVNC', 'dSEZ', 'RGN']).iloc[:, 0:2]
    mask_list = []
    for i_iter in range(0, len(data.index)):
        mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
        mask[i_iter, :] = [False]*len(data.columns)
        mask_list.append(mask)

    fig, axs = plt.subplots(
        1, 1, figsize=(.5, .6)
    )
    for j, mask in enumerate(mask_list):
        if((j == 0)):
            vmax = 800
        if((j in [2,4])):
            vmax = 60
        if((j in [1,3])):
            vmax = 100
        ax = axs
        annotations = data.astype(int).astype(str)
        annotations[annotations=='0']=''
        sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

    plt.savefig(f'VNC_interaction/plots/individual_asc_paths/Type_{i}_asc-{asc_types_name[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')

#%%
# export ds-ascending brain neurons

def readable_df(skids_list):
    max_length = max([len(x) for x in skids_list])

    df = pd.DataFrame()
    
    for i, layer in enumerate(skids_list):
    
        skids = list(layer)

        if(len(layer)==0):
            skids = ['']
        if(len(skids) != max_length):
            skids = skids + ['']*(max_length-len(skids))

        df[f'Layer {i}'] = skids

    return(df)

all_type_layers_skids.columns = [int(x) for x in all_type_layers_skids.columns]

asc_all_layers_readable = []
for column in all_type_layers_skids.columns:
    readable = readable_df(all_type_layers_skids.loc[:,column])
    asc_all_layers_readable.append(readable)

pd.concat(asc_all_layers_readable).to_csv(f'VNC_interaction/plots/individual_asc_paths/all_paths_ascending_Threshold-{threshold}.csv')
# %%
# ascending paths upstream of particular dVNC paths
from tqdm import tqdm
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
order = [16, 0, 2, 11, 1, 5, 7, 12, 13, 8, 3, 9, 10, 15, 4, 6, 14]
group1 = [0,1,2,3]
group2 = [4, 5, 6, 7, 8]
group3 = [9, 10]
group4 = [11,12,13]

ascending_layers,ascending_skids = A1_adj.layer_id(source_dVNC_pair_paths, source_dVNC_pairs.leftid, A1_ascending)
ascending_layers = ascending_layers.iloc[order, :]
ascending_skids = ascending_skids.T.iloc[order, :]

ascending_skids_allhops = []
for index in ascending_skids.index:
    skids_allhops = [x for sublist in ascending_skids.loc[index].values for x in sublist if x!='']
    ascending_skids_allhops.append(skids_allhops)
# %%
# identifying ascending_paths based on ascending IDs in dVNC paths
# not yet complete
all_ascending_layers,all_ascending_layers_skids = VNC_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, br)
all_ascending_layers_skids.columns = [int(x) for x in all_ascending_layers_skids.columns]

left_skids_list = []
ascending_dVNC_paths = []
for ascending_skids in ascending_skids_allhops:
    left_skids = Promat.extract_pairs_from_list(ascending_skids, pairs)[0].leftid.values
    path = all_ascending_layers_skids.loc[:, left_skids].values

    path_combined = []
    for layers in path:
        layers = [x for sublist in layers for x in sublist]
        #if(len(layers)>1):
        #    path_combined.append(np.concatenate(layers))
        #if(len(layers)==1):
        #    path_combined.append(layers)
        path_combined.append(layers)

    ascending_dVNC_paths.append(path_combined)
    left_skids_list.append(left_skids)


# identify neuron types
all_layers,_ = VNC_adj.layer_id(ascending_dVNC_paths, range(0, len(ascending_dVNC_paths)), VNC_adj.adj.index) # include all neurons to get total number of neurons per layer
dVNC_A1_layers,dVNC_A1_layers_skids = VNC_adj.layer_id(ascending_dVNC_paths, range(0, len(ascending_dVNC_paths)), dVNC_to_A1) 
dVNC_layers,_ = VNC_adj.layer_id(ascending_dVNC_paths, range(0, len(ascending_dVNC_paths)), dVNC)
pre_dVNC_layers,_ = VNC_adj.layer_id(ascending_dVNC_paths, range(0, len(ascending_dVNC_paths)), pre_dVNC) 

# ordered source dVNC pairs in same way
source_dVNC_pairs_ordered = source_dVNC_pairs.loc[order, :]
source_dVNC_pairs_ordered = source_dVNC_pairs_ordered.reset_index(drop=True)

dVNC_A1_layers_skids.columns = source_dVNC_pairs_ordered.leftid

contains_origin_list = []
for column in dVNC_A1_layers_skids.columns:
    path = dVNC_A1_layers_skids.loc[:, column]
    contains_origin = [True if x==column else False for sublist in path for x in sublist]
    contains_origin_list.append(sum(contains_origin))

    
# %%
