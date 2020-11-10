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
plt.rcParams['font.size'] = 6

rm = pymaid.CatmaidInstance(url, name, password, token)
adj = pd.read_csv('VNC_interaction/data/axon-dendrite.csv', header = 0, index_col = 0)
inputs = pd.read_csv('VNC_interaction/data/input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

VNC_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1_proprio = pymaid.get_skids_by_annotation('mw A1 proprio')
A1_somato = pymaid.get_skids_by_annotation('mw A1 somato')

# %%
# comparison of motorneurons contained in each path
from tqdm import tqdm
from connectome_tools.cascade_analysis import Celltype, Celltype_Analyzer

threshold = 0.01

source_dVNC, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
edges, ds_dVNC_cleaned = VNC_adj.edge_threshold(source_dVNC, ds_dVNC, threshold, direction='downstream')
edges[edges.overthres==True]

source_dVNC_cleaned = np.unique(edges[edges.overthres==True].upstream_pair_id)
source_dVNC_pairs = VNC_adj.adj_inter.loc[(slice(None), source_dVNC_cleaned), :].index
source_dVNC_pairs = [x[2] for x in source_dVNC_pairs]
source_dVNC_pairs = Promat.extract_pairs_from_list(source_dVNC_pairs, pairs)[0]

source_dVNC_pair_paths = []
for index in tqdm(range(0, len(source_dVNC_pairs))):
    ds_dVNC = VNC_adj.downstream_multihop(list(source_dVNC_pairs.loc[index]), threshold, min_members = 0, hops=5)
    source_dVNC_pair_paths.append(ds_dVNC)

order = [16, 0, 2, 11, 1, 5, 7, 12, 13, 8, 3, 9, 10, 15, 4, 6, 14]
motor_layers,motor_skids = VNC_adj.layer_id(source_dVNC_pair_paths, source_dVNC_pairs.leftid, A1_MN)
motor_layers = motor_layers.iloc[order, :]
motor_skids = motor_skids.T.iloc[order, :]

motor_skids_allhops = []
for index in motor_skids.index:
    skids_allhops = [x for sublist in motor_skids.loc[index].values for x in sublist if x!='']
    motor_skids_allhops.append(skids_allhops)

motorneuron_celltypes = [Celltype(motor_layers.index[i], skids) for i, skids in enumerate(motor_skids_allhops)]

celltypes = Celltype_Analyzer(motorneuron_celltypes)
iou_matrix = celltypes.compare_membership()
'''
fig, axs = plt.subplots(
    1, 1, figsize=(5, 5)
)

ax = axs
fig.tight_layout(pad=2.0)
sns.heatmap(iou_matrix, ax = ax, square = True)
'''
sns.clustermap(iou_matrix, figsize = (5, 5), square=True)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_dVNC_paths_MN_comparison.pdf', bbox_inches='tight')

# %%
# same with ascendings

ascending_layers,ascending_skids = VNC_adj.layer_id(source_dVNC_pair_paths, source_dVNC_pairs.leftid, A1_ascending)
ascending_layers = ascending_layers.iloc[order, :]
ascending_skids = ascending_skids.T.iloc[order, :]

ascending_skids_allhops = []
for index in motor_skids.index:
    skids_allhops = [x for sublist in ascending_skids.loc[index].values for x in sublist if x!='']
    ascending_skids_allhops.append(skids_allhops)

ascending_celltypes = [Celltype(ascending_layers.index[i], skids) for i, skids in enumerate(ascending_skids_allhops)]
ascending_celltypes = Celltype_Analyzer(ascending_celltypes)
ascending_iou_matrix = ascending_celltypes.compare_membership()
'''
fig, axs = plt.subplots(
    1, 1, figsize=(5, 5)
)

ax = axs
fig.tight_layout(pad=2.0)
sns.heatmap(ascending_iou_matrix, ax = ax, square = True)
'''
sns.clustermap(ascending_iou_matrix, figsize = (5, 5), square=True)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_dVNC_paths_ascendings_comparison.pdf', bbox_inches='tight')

# %%
# multiple-hop connectivity matrix of dVNCs to motorneurons and ascendings

A1_MN_pairs = Promat.extract_pairs_from_list(A1_MN, pairs)[0]
motor_layers,motor_skids = VNC_adj.layer_id(source_dVNC_pair_paths, source_dVNC_pairs.leftid, A1_MN)
motor_layers = motor_layers.iloc[order, :]
motor_skids = motor_skids.T.iloc[order, :]

dVNC_motor_mat, dVNC_motor_mat_plotting = VNC_adj.hop_matrix(motor_skids, source_dVNC_pairs.leftid[order], A1_MN_pairs.leftid)

annotations = dVNC_motor_mat.astype(int).astype(str)
annotations[annotations=='0']=''
sns.clustermap(dVNC_motor_mat_plotting, annot = annotations, fmt = 's', 
                row_cluster = False, cmap='Reds', figsize = (3.5, 3))
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_Hopwise_Connectivity_dVNC-motor_annots.pdf', bbox_inches='tight')

sns.clustermap(dVNC_motor_mat_plotting, 
                col_cluster = False, cmap='Reds', figsize = (3.5, 3), square = True)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_Hopwise_Connectivity_dVNC-motor.pdf', bbox_inches='tight')

# dVNC to ascendings multihop plot
ascending_pairs = Promat.extract_pairs_from_list(A1_ascending, pairs)[0]
ascending_layers,ascending_skids = VNC_adj.layer_id(source_dVNC_pair_paths, source_dVNC_pairs.leftid, A1_ascending)
ascending_layers = ascending_layers.iloc[order, :]
ascending_skids = ascending_skids.T.iloc[order, :]

dVNC_asc_mat, dVNC_asc_mat_plotting = VNC_adj.hop_matrix(ascending_skids, source_dVNC_pairs.leftid[order], ascending_pairs.leftid)
dVNC_asc_mat_plotting = dVNC_asc_mat_plotting.loc[:, (dVNC_asc_mat_plotting).sum(axis=0)>0]

sns.clustermap(dVNC_asc_mat_plotting, 
                row_cluster = False, cmap='Blues', figsize = (2, 3), square = True)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_Hopwise_Connectivity_dVNC-ascending.pdf', bbox_inches='tight')

# %%
# multihop plot of dVNCs to dVNCs via ascending neurons
# first, run 2-hop paths of each ascending
# second, identify ascendings neurons ds of each dVNC
# third, sum hops from dVNC->ascending and then ascending->new dVNC in brain


adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dVNC_pairs = Promat.extract_pairs_from_list(dVNC, pairs)[0]
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
br = pymaid.get_skids_by_annotation('mw brain neurons')

A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

br_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')

A1_ascending_pairs = Promat.extract_pairs_from_list(A1_ascending, pairs)[0]

# ascending paths
ascending_pair_paths = []
for index in tqdm(range(0, len(A1_ascending_pairs))):
    ds_asc = br_adj.downstream_multihop(list(A1_ascending_pairs.loc[index]), threshold, min_members = 0, hops=2)
    ascending_pair_paths.append(ds_asc)

asc_dVNC_layers, asc_dVNC_skids = br_adj.layer_id(ascending_pair_paths, A1_ascending_pairs.leftid, dVNC)
asc_dVNC_skids = asc_dVNC_skids.T
asc_dVNC_skids.index = [int(x) for x in asc_dVNC_skids.index]

# ascending to dVNC hop counts
asc_dVNC_mat, asc_dVNC_mat_plotting = br_adj.hop_matrix(asc_dVNC_skids, A1_ascending_pairs.leftid, dVNC_pairs.leftid)

# determine hops from dVNC to ascendings
dVNC_dVNC_mat = pd.DataFrame(np.zeros(shape=(len(source_dVNC_pairs.leftid), len(dVNC_pairs.leftid))), index = source_dVNC_pairs.leftid[order], columns = dVNC_pairs.leftid)
for i, index in enumerate(dVNC_dVNC_mat.index):
    for j, column in enumerate(dVNC_dVNC_mat.columns):
        data = dVNC_asc_mat.loc[index, :]
        ascendings = data[data>0].index
        for ascending in ascendings:
            data_asc = asc_dVNC_mat.loc[ascending, :]
            dVNCs = data_asc[data_asc>0].index
            if(column in dVNCs):
                dVNC_dVNC_mat.loc[index, column] = dVNC_asc_mat.loc[index, ascending] + asc_dVNC_mat.loc[ascending, column]

dVNC_dVNC_mat = dVNC_dVNC_mat.loc[:, dVNC_dVNC_mat.sum(axis=0)>0]

max_value = dVNC_dVNC_mat.values.max()
dVNC_dVNC_mat_plotting = dVNC_dVNC_mat.copy()

for index in dVNC_dVNC_mat_plotting.index:
    for column in dVNC_dVNC_mat_plotting.columns:
        if(dVNC_dVNC_mat_plotting.loc[index, column]>0):
            dVNC_dVNC_mat_plotting.loc[index, column] = 1 - (dVNC_dVNC_mat_plotting.loc[index, column] - max_value)

sns.clustermap(dVNC_dVNC_mat_plotting, 
                row_cluster = False, col_cluster = False, cmap='Greens', figsize = (3, 3), square = True)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_Hopwise_Connectivity_dVNC-dVNC.pdf', bbox_inches='tight')

# %%
# do any dVNC-to-A1s talk to themselves through ascendings?
dVNC_to_A1 = pymaid.get_skids_by_annotation('mw dVNC to A1')

len(np.intersect1d(dVNC_dVNC_mat.index, dVNC_to_A1))/len(dVNC_dVNC_mat.index)
len(np.intersect1d(dVNC_dVNC_mat.columns, dVNC_to_A1))/len(dVNC_dVNC_mat.columns)

dVNCA1_dVNCA1_mat = dVNC_dVNC_mat.loc[:, np.intersect1d(dVNC_dVNC_mat.columns, dVNC_to_A1)]

# %% 
# identify dVNC -> dVNC paths for schematic generation

def find_ds(mat, indices):
    all_ds_partners = []

    for index in indices:
        if(index in mat.index):
            ds_partners = mat.columns[mat.loc[index, :]>0]
            all_ds_partners.append(list(ds_partners))
        if(index not in mat.index):
            all_ds_partners.append([])
         
    #all_ds_partners = pd.DataFrame(all_ds_partners, index = indices)
    return(all_ds_partners)

layer0 = list(dVNCA1_dVNCA1_mat.index)
layer1 = find_ds(dVNCA1_dVNCA1_mat, dVNCA1_dVNCA1_mat.index)
layer2 = []
for layer in layer1:
    layer_new = find_ds(dVNCA1_dVNCA1_mat, layer)
    layer2.append(layer_new)

df = pd.DataFrame()
df['layer0'] = layer0
df['layer1'] = layer1
df['layer2'] = layer2

df_loops = df.loc[[x!=[] for x in df.layer1], :]

# plot MN hits for each of these dVNCs
import math 
MN_order = pd.read_csv('VNC_interaction/data/motorneuron-muscle-groups.csv')
MN_order = [int(x) for x in MN_order.skid_leftid if np.invert(math.isnan(x))]

for index in df_loops.index:
    indices = [df_loops.layer0[index]] + [x for x in df_loops.layer1[index] if x!=[]] + [x for sublist in df_loops.layer2[index] for x in sublist if x!=[]]
    
    height = 3
    width = 0.25*len(indices)
    fig,ax = plt.subplots(1,1,figsize=(width, height))
    sns.heatmap(dVNC_motor_mat_plotting.loc[indices, MN_order].T, cmap='Reds', cbar = False, ax=ax)
    plt.savefig(f'VNC_interaction/plots/dVNC_loops/Path-{index}_Threshold-{threshold}_Hopwise_Connectivity_dVNC-motor.pdf', bbox_inches='tight')

 
# %%
# %%
# below is incomplete 
# pathways downstream of each dVNC pair
from tqdm import tqdm

threshold = 0.01
'''
source_dVNC, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
edges, ds_dVNC_cleaned = VNC_adj.edge_threshold(source_dVNC, ds_dVNC, threshold, direction='downstream')
edges[edges.overthres==True]

source_dVNC_cleaned = np.unique(edges[edges.overthres==True].upstream_pair_id)
source_dVNC_pairs = VNC_adj.adj_inter.loc[(slice(None), source_dVNC_cleaned), :].index
source_dVNC_pairs = [x[2] for x in source_dVNC_pairs]
source_dVNC_pairs = Promat.extract_pairs_from_list(source_dVNC_pairs, pairs)[0]
'''
source_dVNC_pairs = Promat.extract_pairs_from_list(dVNC, pairs)[0]
source_MN_pairs = Promat.extract_pairs_from_list(A1_MN, pairs)[0]

dVNC_pair_paths = []
for index in tqdm(range(0, len(source_dVNC_pairs))):
    ds_dVNC = VNC_adj.downstream_multihop(list(source_dVNC_pairs.loc[index]), threshold, min_members = 0, hops=5)
    dVNC_pair_paths.append(ds_dVNC)

MN_pair_paths = []
for index in tqdm(range(0, len(source_MN_pairs))):
    us_MN = VNC_adj.upstream_multihop(list(source_MN_pairs.loc[index]), threshold, min_members = 0, hops=5)
    MN_pair_paths.append(us_MN)

# %%
# how many motorneurons downstream of each dVNC
# how many dVNCs upstream of each motorneuron


motor_layers,motor_skids = VNC_adj.layer_id(dVNC_pair_paths, source_dVNC_pairs.leftid, A1_MN)
dVNC_layers,dVNC_skids = VNC_adj.layer_id(MN_pair_paths, source_MN_pairs.leftid, dVNC)

'''
# grouped bar plot 
# too complicated
barwidth = 0.25

r1 = np.arange(len(dVNC_layers.index))
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]
r5 = [x + barwidth for x in r4]

plt.bar(r1, dVNC_layers['Layer 1'], width = barwidth, label='Layer 1')
plt.bar(r2, dVNC_layers['Layer 2'], width = barwidth, label='Layer 2')
plt.bar(r3, dVNC_layers['Layer 3'], width = barwidth, label='Layer 3')
plt.bar(r4, dVNC_layers['Layer 4'], width = barwidth, label='Layer 4')
plt.bar(r5, dVNC_layers['Layer 5'], width = barwidth, label='Layer 5')
'''