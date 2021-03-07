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

A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
br = pymaid.get_skids_by_annotation('mw brain neurons')
MBON = pymaid.get_skids_by_annotation('mw MBON')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')
LHN = pymaid.get_skids_by_annotation('mw LHN')
CN = pymaid.get_skids_by_annotation('mw CN')
KC = pymaid.get_skids_by_annotation('mw KC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC 1%')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
uPN = pymaid.get_skids_by_annotation('mw uPN')
tPN = pymaid.get_skids_by_annotation('mw tPN')
vPN = pymaid.get_skids_by_annotation('mw vPN')
mPN = pymaid.get_skids_by_annotation('mw mPN')
PN = uPN + tPN + vPN + mPN
FBN = pymaid.get_skids_by_annotation('mw FBN')
FB2N = pymaid.get_skids_by_annotation('mw FB2N')
FBN_all = FBN + FB2N

input_names = pymaid.get_annotated('mw brain inputs').name
input_names = input_names.drop(6)
general_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd']
input_skids_list = list(map(pymaid.get_skids_by_annotation, input_names))
sens_all = [x for sublist in input_skids_list for x in sublist]

asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')
asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
asc_all = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')

CN = list(np.setdiff1d(CN, LHN + FBN_all)) # 'CN' means exclusive CNs that are not FBN or LHN
pre_dVNC = list(np.setdiff1d(pre_dVNC, MBON + MBIN + LHN + CN + KC + dSEZ + dVNC + PN + FBN_all + asc_all)) # 'pre_dVNC' must have no other category assignment
dSEZ = list(np.setdiff1d(dSEZ, MBON + MBIN + LHN + CN + KC + dVNC + PN + FBN_all))

A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

br_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')

# %%
# projectome plot and ordering
# needs work; lots of conflicting ordering sections added for testing purposes

import cmasher as cmr

projectome = pd.read_csv('descending_neurons_analysis/data/projectome_adjacency.csv', index_col = 0, header = 0)
projectome.index = [str(x) for x in projectome.index]

# identify meshes
meshes = ['SEZ_left', 'SEZ_right', 'T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right', 'A1_left', 'A1_right', 'A2_left', 'A2_right', 'A3_left', 'A3_right', 'A4_left', 'A4_right', 'A5_left', 'A5_right', 'A6_left', 'A6_right', 'A7_left', 'A7_right', 'A8_left', 'A8_right']

pairOrder_dVNC = []
for skid in dVNC:
    if(skid in pairs["leftid"].values):
        pair_skid = pairs["rightid"][pairs["leftid"]==skid].iloc[0]
        pairOrder_dVNC.append(skid)
        pairOrder_dVNC.append(pair_skid)

input_projectome = projectome.loc[meshes, [str(x) for x in pairOrder_dVNC]]
output_projectome = projectome.loc[[str(x) for x in pairOrder_dVNC], meshes]

dVNC_projectome_pairs_summed_output = []
indices = []
for i in np.arange(0, len(output_projectome.index), 2):
    combined_pairs = (output_projectome.iloc[i, :] + output_projectome.iloc[i+1, :])

    combined_hemisegs = []
    for j in np.arange(0, len(combined_pairs), 2):
        combined_hemisegs.append((combined_pairs[j] + combined_pairs[j+1]))
    
    dVNC_projectome_pairs_summed_output.append(combined_hemisegs)
    indices.append(output_projectome.index[i])

dVNC_projectome_pairs_summed_output = pd.DataFrame(dVNC_projectome_pairs_summed_output, index = indices, columns = ['SEZ', 'T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'])
dVNC_projectome_pairs_summed_output = dVNC_projectome_pairs_summed_output.iloc[:, 1:len(dVNC_projectome_pairs_summed_output)]

#normalize # of presynaptic sites
dVNC_projectome_pairs_summed_output_norm = dVNC_projectome_pairs_summed_output.copy()
for i in range(len(dVNC_projectome_pairs_summed_output)):
    sum_row = sum(dVNC_projectome_pairs_summed_output_norm.iloc[i, :])
    for j in range(len(dVNC_projectome_pairs_summed_output.columns)):
        dVNC_projectome_pairs_summed_output_norm.iloc[i, j] = dVNC_projectome_pairs_summed_output_norm.iloc[i, j]/sum_row

# order based on clustering raw data
cluster = sns.clustermap(dVNC_projectome_pairs_summed_output, col_cluster = False, figsize=(6,4), rasterized=True)
row_order = cluster.dendrogram_row.reordered_ind
#fig, ax = plt.subplots(figsize=(6,4))
#sns.heatmap(dVNC_projectome_pairs_summed_output.iloc[row_order, :], rasterized=True, ax=ax)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_raw.pdf', bbox_inches='tight')

# order based on clustering normalized data
cluster = sns.clustermap(dVNC_projectome_pairs_summed_output_norm, col_cluster = False, figsize=(6,4), rasterized=True)
row_order = cluster.dendrogram_row.reordered_ind
#fig, ax = plt.subplots(figsize=(6,4))
#sns.heatmap(dVNC_projectome_pairs_summed_output_norm.iloc[row_order, :], rasterized=True, ax=ax)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_normalized.pdf', bbox_inches='tight')

# order based on counts per column
for i in range(1, 51):
    dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output_norm.copy()
    dVNC_projectome_pairs_summed_output_sort[dVNC_projectome_pairs_summed_output_sort<(i/100)]=0
    dVNC_projectome_pairs_summed_output_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order = dVNC_projectome_pairs_summed_output_sort[dVNC_projectome_pairs_summed_output_sort.sum(axis=1)>0].index

    second_sort = dVNC_projectome_pairs_summed_output_norm[dVNC_projectome_pairs_summed_output_sort.sum(axis=1)==0]
    second_sort[second_sort<.1]=0
    second_sort.sort_values(by=[i for i in reversed(['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'])], ascending=False, inplace=True)
    row_order = list(row_order) + list(second_sort.index)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(dVNC_projectome_pairs_summed_output_norm.loc[row_order, :], ax=ax, rasterized=True)
    plt.savefig(f'VNC_interaction/plots/projectome/projectome_0.{i}-sort-threshold.pdf', bbox_inches='tight')

for i in range(1, 51):
    dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output.copy()
    dVNC_projectome_pairs_summed_output_sort[dVNC_projectome_pairs_summed_output_sort<(i)]=0
    dVNC_projectome_pairs_summed_output_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order = dVNC_projectome_pairs_summed_output_sort.index

    second_sort = dVNC_projectome_pairs_summed_output[dVNC_projectome_pairs_summed_output_sort.sum(axis=1)==0]
    second_sort[second_sort<10]=0
    second_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order = list(row_order) + list(second_sort.index)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(dVNC_projectome_pairs_summed_output.loc[row_order, :], ax=ax, rasterized=True)
    plt.savefig(f'VNC_interaction/plots/projectome/projectome_{i}-sort-threshold.pdf', bbox_inches='tight')
'''
fig, ax = plt.subplots(figsize=(3,2))
sns.heatmap(dVNC_projectome_pairs_summed_output.iloc[row_order, :], ax=ax)
plt.savefig('VNC_interaction/plots/projectome/output_projectome_cluster.pdf', bbox_inches='tight', transparent = True)

# order input projectome in the same way
dVNC_projectome_pairs_summed_input = []
indices = []
for i in np.arange(0, len(input_projectome.columns), 2):
    combined_pairs = (input_projectome.iloc[:, i] + input_projectome.iloc[:, i+1])

    combined_hemisegs = []
    for j in np.arange(0, len(combined_pairs), 2):
        combined_hemisegs.append((combined_pairs[j] + combined_pairs[j+1]))
    
    dVNC_projectome_pairs_summed_input.append(combined_hemisegs)
    indices.append(input_projectome.columns[i])

dVNC_projectome_pairs_summed_input = pd.DataFrame(dVNC_projectome_pairs_summed_input, index = indices, columns = ['SEZ', 'T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'])
dVNC_projectome_pairs_summed_input = dVNC_projectome_pairs_summed_input.iloc[:, 1:len(dVNC_projectome_pairs_summed_input)]
#cluster = sns.clustermap(dVNC_projectome_pairs_summed_input, col_cluster = False, cmap = cmr.freeze, figsize=(10,10))

fig, ax = plt.subplots(figsize=(3,2))
sns.heatmap(dVNC_projectome_pairs_summed_input.iloc[row_order, :], cmap=cmr.freeze, ax=ax)
plt.savefig('VNC_interaction/plots/projectome/input_projectome_cluster.pdf', bbox_inches='tight', transparent = True)
'''
# %%
# load dVNC pairs

dVNC_pairs = Promat.extract_pairs_from_list(dVNC, pairs)[0]
#dVNC_pairs = dVNC_pairs.loc[row_order]
#dVNC_pairs.reset_index(inplace=True, drop=True)

# add a single dSEZ neuron associated with a split-GAL4 and phenotype
dVNC_pairs = dVNC_pairs.append(pd.DataFrame([[10382686, 16100103]], index=[len(dVNC_pairs)], columns = ['leftid', 'rightid']))
dVNC_pairs = dVNC_pairs.append(pd.DataFrame([[3044500, 6317793]], index=[len(dVNC_pairs)], columns = ['leftid', 'rightid']))

# %%
# paths 2-hop upstream of each dVNC
from tqdm import tqdm

hops = 2
threshold = 0.01

dVNC_pair_paths = []
for i in tqdm(range(0, len(dVNC_pairs))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_pairs.loc[i]), threshold, min_members = 0, hops=hops, strict=False)
    dVNC_pair_paths.append(us_dVNC)

dVNC_pair_paths_output = []
for i in tqdm(range(0, len(dVNC_pairs))):
    ds_dVNC = br_adj.downstream_multihop(list(dVNC_pairs.loc[i]), threshold, min_members = 0, hops=hops, strict=False)
    dVNC_pair_paths_output.append(ds_dVNC)

# %%
# plotting individual dVNC paths

# UPSTREAM
threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, list(adj.index))
dVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, dVNC)
predVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, pre_dVNC)
dSEZ_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, dSEZ)
asc_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, asc_all)
LHN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, LHN)
CN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, CN)
MBON_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, MBON)
MBIN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, MBIN)
FBN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, FBN_all)
KC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, KC)
PN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, PN)
#sens_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, sens_all)

layer_types = [all_type_layers, PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, KC_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
layer_colors = ['Greens', 'Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Blues', 'Reds', 'Purples', 'Reds']
#layer_vmax = [500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
layer_vmax = [200, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 50]
save_path = 'VNC_interaction/plots/dVNC_upstream/'

plt.rcParams['font.size'] = 5

br_adj.plot_layer_types(layer_types=layer_types, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs.leftid, figsize=(.5*hops/3, 1.5), save_path=save_path, threshold=threshold, hops=hops)

# DOWNSTREAM in brain
threshold = 0.01
all_type_layers_ds,all_type_layers_skids_ds = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, list(adj.index))
dVNC_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, dVNC)
predVNC_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, pre_dVNC)
dSEZ_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, dSEZ)
asc_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, asc_all)
LHN_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, LHN)
CN_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, CN)
MBON_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, MBON)
MBIN_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, MBIN)
FBN_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, FBN_all)
KC_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, KC)
PN_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, PN)
#sens_type_layers_ds,_ = br_adj.layer_id(dVNC_pair_paths_output, dVNC_pairs.leftid, sens_all)

layer_types_ds = [all_type_layers_ds, PN_type_layers_ds, LHN_type_layers_ds, MBIN_type_layers_ds, MBON_type_layers_ds, KC_type_layers_ds, 
                FBN_type_layers_ds, CN_type_layers_ds, asc_type_layers_ds, dSEZ_type_layers_ds, predVNC_type_layers_ds, dVNC_type_layers_ds]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
layer_colors = ['Greens', 'Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Blues', 'Reds', 'Purples', 'Reds']
#layer_vmax = [500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
layer_vmax = [200, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 50]
save_path = 'VNC_interaction/plots/dVNC_upstream/Downstream-in-brain_'

plt.rcParams['font.size'] = 5

br_adj.plot_layer_types(layer_types=layer_types_ds, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs.leftid, figsize=(.5*hops/3, 1.5), save_path=save_path, threshold=threshold, hops=hops)

# %%
# make bar plots for 1-hop and 2-hop

# UPSTREAM
ind = [x for x in range(0, len(dVNC_pair_paths))]

# determine fraction of neuron types in 1-hop and 2-hop
fraction_types = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
fraction_types = [x/layer_types[0] for x in fraction_types]
fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
colors = ['#1D79B7', '#D4E29E', '#FF8734', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC', '#D88052', '#E0B1AD', '#A52A2A']

# summary plot of 1st order upstream of dVNCs
plt.bar(ind, fraction_types[0].iloc[:, 0], color = colors[0])
bottom = fraction_types[0].iloc[:, 0]

for i in range(1, len(fraction_types)):
    plt.bar(ind, fraction_types[i].iloc[:, 0], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types[i].iloc[:, 0]

remaining = [1 for x in range(0, len(dVNC_pair_paths))] - bottom
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_1st_order.pdf', format='pdf', bbox_inches='tight')

# summary plot of 2nd order upstream of dVNCs
plt.bar(ind, fraction_types[0].iloc[:, 1], color = colors[0])
bottom = fraction_types[0].iloc[:, 1]

for i in range(1, len(fraction_types)):
    plt.bar(ind, fraction_types[i].iloc[:, 1], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types[i].iloc[:, 1]

remaining = [1 for x in range(0, len(dVNC_pair_paths))] - bottom
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_2nd_order.pdf', format='pdf', bbox_inches='tight')
# %%
# make bar plots for 1-hop and 2-hop

# DOWNSTREAM in brain
ind = [x for x in range(0, len(dVNC_pair_paths_output))]

# determine fraction of neuron types in 1-hop and 2-hop
fraction_types_ds = [PN_type_layers_ds, LHN_type_layers_ds, MBIN_type_layers_ds, MBON_type_layers_ds, 
                FBN_type_layers_ds, CN_type_layers_ds, asc_type_layers_ds, dSEZ_type_layers_ds, predVNC_type_layers_ds, dVNC_type_layers_ds]
fraction_types_ds = [(x/layer_types_ds[0]).fillna(-1) for x in fraction_types_ds]
fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
colors = ['#1D79B7', '#D4E29E', '#FF8734', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC', '#D88052', '#E0B1AD', '#A52A2A']

# summary plot of 1st order downstream of dVNCs in brain
plt.bar(ind, fraction_types_ds[0].iloc[:, 0], color = colors[0])
bottom = fraction_types_ds[0].iloc[:, 0]

for i in range(1, len(fraction_types_ds)):
    plt.bar(ind, fraction_types_ds[i].iloc[:, 0], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types_ds[i].iloc[:, 0]

bottom[bottom==-10]=1
remaining = [1 for x in range(0, len(dVNC_pair_paths_output))] - bottom
#remaining = [x if x!=-1 else 0.0 for x in remaining]
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.ylim(0,1)
plt.savefig('VNC_interaction/plots/dVNC_upstream/downstream_summary_plot_1st_order.pdf', format='pdf', bbox_inches='tight')

# summary plot of 2nd order downstream of dVNCs in brain
plt.bar(ind, fraction_types_ds[0].iloc[:, 1], color = colors[0])
bottom = fraction_types_ds[0].iloc[:, 1]

for i in range(1, len(fraction_types_ds)):
    plt.bar(ind, fraction_types_ds[i].iloc[:, 1], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types_ds[i].iloc[:, 1]

bottom[bottom==-10]=1
remaining = [1 for x in range(0, len(dVNC_pair_paths_output))] - bottom
#remaining = [x if x!=-1 else 0.0 for x in remaining]
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.ylim(0,1)
plt.savefig('VNC_interaction/plots/dVNC_upstream/downstream_summary_plot_2nd_order.pdf', format='pdf', bbox_inches='tight')

# %%
# combine all data types for dVNCs: us1o, us2o, ds1o, ds2o, projectome

fraction_cell_types_1o_us = pd.DataFrame([x.iloc[:, 0] for x in fraction_types], index = fraction_types_names).T
fraction_cell_types_1o_us.columns = [f'1o_us_{x}' for x in fraction_cell_types_1o_us.columns]
unk_col = 1-fraction_cell_types_1o_us.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_1o_us['1o_us_unk']=unk_col

fraction_cell_types_2o_us = pd.DataFrame([x.iloc[:, 1] for x in fraction_types], index = fraction_types_names).T
fraction_cell_types_2o_us.columns = [f'2o_us_{x}' for x in fraction_cell_types_2o_us.columns]
unk_col = 1-fraction_cell_types_2o_us.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_2o_us['2o_us_unk']=unk_col

fraction_cell_types_1o_ds = pd.DataFrame([x.iloc[:, 0] for x in fraction_types_ds], index = fraction_types_names).T
fraction_cell_types_1o_ds.columns = [f'1o_ds_{x}' for x in fraction_cell_types_1o_ds.columns]
unk_col = 1-fraction_cell_types_1o_ds.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_1o_ds['1o_ds_unk']=unk_col
fraction_cell_types_1o_ds[fraction_cell_types_1o_ds==-1]=0

fraction_cell_types_2o_ds = pd.DataFrame([x.iloc[:, 1] for x in fraction_types_ds], index = fraction_types_names).T
fraction_cell_types_2o_ds.columns = [f'2o_ds_{x}' for x in fraction_cell_types_2o_ds.columns]
unk_col = 1-fraction_cell_types_2o_ds.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_2o_ds['2o_ds_unk']=unk_col
fraction_cell_types_2o_ds[fraction_cell_types_2o_ds==-1]=0

all_data = dVNC_projectome_pairs_summed_output_norm.copy()
all_data.index = [int(x) for x in all_data.index]

all_data = pd.concat([fraction_cell_types_1o_us, fraction_cell_types_2o_us, all_data, fraction_cell_types_1o_ds, fraction_cell_types_2o_ds], axis=1)
all_data.fillna(0, inplace=True)

# clustered version of all_data combined
cluster = sns.clustermap(all_data, col_cluster = False, figsize=(30,30), rasterized=True)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_all_data.pdf', bbox_inches='tight')
order = cluster.dendrogram_row.reordered_ind
fig,ax=plt.subplots(1,1,figsize=(6,4))
sns.heatmap(all_data.iloc[order, :].drop(list(fraction_cell_types_1o_us.columns) + list(fraction_cell_types_2o_us.columns) + list(fraction_cell_types_1o_ds.columns) + list(fraction_cell_types_2o_ds.columns), axis=1), ax=ax, rasterized=True)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_all_data_same_size.pdf', bbox_inches='tight')

cluster = sns.clustermap(all_data.drop(['1o_us_pre-dVNC', '2o_us_pre-dVNC'], axis=1), col_cluster = False, figsize=(20,15), rasterized=True)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_all_data_removed_us-pre-dVNCs.pdf', bbox_inches='tight')

# decreasing sort of all_data but with feedback and non-feedback dVNC clustered
for i in range(1, 50):
    dVNCs_with_FB = all_data.loc[:, list(fraction_cell_types_1o_ds.columns) + list(fraction_cell_types_2o_ds.columns)].sum(axis=1)
    dVNCs_FB_true_skids = dVNCs_with_FB[dVNCs_with_FB>0].index
    dVNCs_FB_false_skids = dVNCs_with_FB[dVNCs_with_FB==0].index

    dVNC_projectome_pairs_summed_output_sort = all_data.copy()
    dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output_sort.loc[:, ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']]
    dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output_sort.loc[dVNCs_FB_true_skids]
    dVNC_projectome_pairs_summed_output_sort[dVNC_projectome_pairs_summed_output_sort<(i/100)]=0
    dVNC_projectome_pairs_summed_output_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order_FB_true = dVNC_projectome_pairs_summed_output_sort.index

    second_sort = all_data.copy()
    second_sort = second_sort.loc[:, ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']]
    second_sort = second_sort.loc[dVNCs_FB_false_skids]
    second_sort[second_sort<(i/100)]=0
    second_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order_FB_false = second_sort.index
    row_order = list(row_order_FB_true) + list(row_order_FB_false)
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(all_data.loc[row_order, :], ax=ax, rasterized=True)
    plt.savefig(f'VNC_interaction/plots/projectome/splitFB_projectome_0.{i}-sort-threshold.pdf', bbox_inches='tight')
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(all_data.loc[row_order, ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax=ax, rasterized=True)
    plt.savefig(f'VNC_interaction/plots/projectome/splitFB_same-size_projectome_0.{i}-sort-threshold.pdf', bbox_inches='tight')
# %%
# what fraction of us and ds neurons are from different cell types per hop?

fraction_cell_types_1o_us = pd.DataFrame([x.iloc[:, 0] for x in fraction_types], index = fraction_types_names)
fraction_cell_types_1o_us = fraction_cell_types_1o_us.fillna(0) # one dVNC with no inputs

fraction_cell_types_2o_us = pd.DataFrame([x.iloc[:, 1] for x in fraction_types], index = fraction_types_names)
fraction_cell_types_2o_us = fraction_cell_types_2o_us.fillna(0) # one dVNC with no inputs

fraction_cell_types_1o_us_scatter = []
for j in range(1, len(fraction_cell_types_1o_us.columns)):
    for i in range(0, len(fraction_cell_types_1o_us.index)):
        fraction_cell_types_1o_us_scatter.append([fraction_cell_types_1o_us.iloc[i, j], fraction_cell_types_1o_us.index[i]]) 

fraction_cell_types_1o_us_scatter = pd.DataFrame(fraction_cell_types_1o_us_scatter, columns = ['fraction', 'cell_type'])

fraction_cell_types_2o_us_scatter = []
for j in range(1, len(fraction_cell_types_2o_us.columns)):
    for i in range(0, len(fraction_cell_types_2o_us.index)):
        fraction_cell_types_2o_us_scatter.append([fraction_cell_types_2o_us.iloc[i, j], fraction_cell_types_2o_us.index[i]]) 

fraction_cell_types_2o_us_scatter = pd.DataFrame(fraction_cell_types_2o_us_scatter, columns = ['fraction', 'cell_type'])

fig, ax = plt.subplots(1, 1, figsize=(1.25,1))
sns.stripplot(x='cell_type', y='fraction', data=fraction_cell_types_1o_us_scatter, ax=ax, size=.5, jitter=0.2)
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-0.05,1))
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-1o.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(1.25,1))
sns.stripplot(x='cell_type', y='fraction', data=fraction_cell_types_2o_us_scatter, ax=ax, size=.5, jitter=0.2)
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-0.05,1))
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-2o.pdf', format='pdf', bbox_inches='tight')

# %%
# number of us and ds neurons are from different cell types per hop?

layer_types_ds_counts = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]

cell_types_1o_us = pd.DataFrame([x.iloc[:, 0] for x in layer_types_ds_counts], index = fraction_types_names)
cell_types_1o_us = cell_types_1o_us.fillna(0) # one dVNC with no inputs

cell_types_2o_us = pd.DataFrame([x.iloc[:, 1] for x in layer_types_ds_counts], index = fraction_types_names)
cell_types_2o_us = cell_types_2o_us.fillna(0) # one dVNC with no inputs

cell_types_1o_us_scatter = []
for j in range(1, len(cell_types_1o_us.columns)):
    for i in range(0, len(cell_types_1o_us.index)):
        cell_types_1o_us_scatter.append([cell_types_1o_us.iloc[i, j], cell_types_1o_us.index[i]]) 

cell_types_1o_us_scatter = pd.DataFrame(cell_types_1o_us_scatter, columns = ['counts', 'cell_type'])

cell_types_2o_us_scatter = []
for j in range(1, len(cell_types_2o_us.columns)):
    for i in range(0, len(cell_types_2o_us.index)):
        cell_types_2o_us_scatter.append([cell_types_2o_us.iloc[i, j], cell_types_2o_us.index[i]]) 

cell_types_2o_us_scatter = pd.DataFrame(cell_types_2o_us_scatter, columns = ['counts', 'cell_type'])

fig, ax = plt.subplots(1, 1, figsize=(1.25,1.25))
sns.stripplot(x='cell_type', y='counts', hue='cell_type', data=cell_types_1o_us_scatter, palette = colors, ax=ax, size=.5, jitter=0.25, alpha=0.9, edgecolor="none")
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-2, 80), xlabel='')
ax.get_legend().remove()

median_width = 0.6

for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"

    # calculate the median value for all replicates of either X or Y
    median_val = cell_types_1o_us_scatter[cell_types_1o_us_scatter['cell_type']==sample_name].counts.median()

    # plot horizontal lines across the column, centered on the tick
    ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val], lw=0.25, color='k', zorder=100)

plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-1o_counts.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(1.25,1.25))
sns.stripplot(x='cell_type', y='counts', hue='cell_type', data=cell_types_2o_us_scatter, palette = colors, ax=ax, size=.5, jitter=0.25, alpha=0.9, edgecolor="none")
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-2, 80), xlabel='')
ax.get_legend().remove()

median_width = 0.6

for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"

    # calculate the median value for all replicates of either X or Y
    median_val = cell_types_2o_us_scatter[cell_types_2o_us_scatter['cell_type']==sample_name].counts.median()

    # plot horizontal lines across the column, centered on the tick
    ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val], lw=0.25, color='k', zorder=100)

plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-2o_counts.pdf', format='pdf', bbox_inches='tight')

# %%
# how many dVNCs have different cell types per hop?

cell_types = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
cell_types_names = ['PN', 'LHN', 'MBIN', 'MBON', 'MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']

counts = []
for i in range(0, len(cell_types)):
    counts.append([f'{cell_types_names[i]}' , sum(cell_types[i].iloc[:, 0]>0), sum(cell_types[i].iloc[:, 1]>0)])

counts = pd.DataFrame(counts, columns = ['cell_type', 'number_1o', 'number_2o']).set_index('cell_type')

plt.rcParams['font.size'] = 5
x = np.arange(len(counts))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(2,1.5))
order1 = ax.bar(x - width/2, counts.number_1o, width, label='Directly Upstream')
order2 = ax.bar(x + width/2, counts.number_2o, width, label='2-Hop Upstream')

ax.set_ylabel('Number of dVNCs')
ax.set_xticks(x)
ax.set_xticklabels(counts.index)
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_1o_2o_counts.pdf', format='pdf', bbox_inches='tight')

# %%
# histogram of number of cell types per dVNC at 1o and 2o upstream and downstream
# work in progress
from matplotlib import pyplot as py
'''
fig, ax = plt.subplots(figsize=(3,3))
bins_2o = np.arange(max(cell_types_2o_us.T.PN)/2) - 0.5
sns.distplot(cell_types_2o_us.T.PN/2, bins=bins_2o, kde=False, ax=ax, color='blue')
sns.distplot(cell_types_1o_us.T.PN/2, bins=bins_2o, kde=False, ax=ax, color='blue')
ax.set(xlim=(-1, max(cell_types_2o_us.T.PN)/2), xticks=(range(0, 20, 1)))
'''

fig, axs = plt.subplots(2, 2, figsize=(4,4))

data_1o = cell_types_1o_us.T.PN/2
data_2o = cell_types_2o_us.T.PN/2
max_value = int(max(list(data_2o) + list(data_1o)))
bins_2o = np.arange(max_value+2) - 0.5

ax = axs[0,0] 
ax.hist(data_1o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))

ax = axs[0,1] 
ax.hist(data_2o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))


data_1o = cell_types_1o_us.T.LHN/2
data_2o = cell_types_2o_us.T.LHN/2
max_value = int(max(list(data_2o) + list(data_1o)))
bins_2o = np.arange(max_value+2) - 0.5

ax = axs[1,0] 
ax.hist(data_1o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))

ax = axs[1,1] 
ax.hist(data_2o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))

# %%
# 3D Plot

ct_1o_us = cell_types_1o_us.T/2
ct_2o_us = cell_types_2o_us.T/2

data_1o_3d_list = []
for i in range(0, len(ct_1o_us.columns)):
    data_1o = ct_1o_us.iloc[:, i]
    data_1o_3d = [[i, sum(data_1o==i)] for i in range(int(max(data_1o+1)))]
    data_1o_3d = pd.DataFrame(data_1o_3d, columns=['number', 'height'])
    data_1o_3d_list.append(data_1o_3d)

data_2o_3d_list = []
for i in range(0, len(ct_2o_us.columns)):
    data_2o = ct_2o_us.iloc[:, i]
    data_2o_3d = [[i, sum(data_2o==i)] for i in range(int(max(data_2o+1)))]
    data_2o_3d = pd.DataFrame(data_2o_3d, columns=['number', 'height'])
    data_2o_3d_list.append(data_2o_3d)

fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
for zs_values, i in zip([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], range(len(data_1o_3d_list))):
    ax.bar(data_1o_3d_list[i].number, data_1o_3d_list[i].height, zs = zs_values, zdir='y')

fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
for zs_values, i in zip([0,1,3,4,6,7,9,10,12,13], range(len(data_2o_3d_list))):
    ax.bar(data_2o_3d_list[i].number, data_2o_3d_list[i].height, zs = zs_values, zdir='y')

# %%
# violinplot
cell_types_1o_us_scatter['order']=['1st-order']*len(cell_types_1o_us_scatter)
cell_types_2o_us_scatter['order']=['2nd-order']*len(cell_types_2o_us_scatter)

celltypes_1o_2o_us = cell_types_1o_us_scatter.append(cell_types_2o_us_scatter)

fig, axs = plt.subplots(figsize=(10,5))
sns.boxenplot(y='counts', x='cell_type', hue='order', data =celltypes_1o_2o_us, ax=axs, outlier_prop=0)
#sns.boxenplot(y='counts', x='cell_type', data=celltypes_1o_2o_us, ax=axs)
plt.savefig('VNC_interaction/plots/dVNC_upstream/boxenplot.pdf', bbox_inches='tight', transparent = True)

# %%
# Ridgeline plot
import joypy
#fig, axes = joypy.joyplot(cell_types_2o_us_scatter, by="cell_type", overlap=4) #, hist=True, bins=int(max(cell_types_2o_us_scatter.counts)))
joypy.joyplot(fraction_cell_types_1o_us_scatter, by="cell_type", overlap=4)
#fig, axes = joypy.joyplot(cell_types_1o_us_scatter, by="cell_type")

# %%
# multi-hop matrix of all cell types to dVNCs
# incomplete

threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, br)
dVNC_type_layers,dVNC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, dVNC)
predVNC_type_layers,predVNC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, pre_dVNC)
dSEZ_type_layers,dSEZ_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, dSEZ)
LHN_type_layers,LHN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, LHN)
CN_type_layers,CN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, CN)
MBON_type_layers,MBON_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, MBON)
MBIN_type_layers,MBIN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, MBIN)
FBN_type_layers,FBN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, FBN_all)
KC_type_layers,KC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, KC)
PN_type_layers,PN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, PN)

#sns.barplot(x = [1, 2], y = [sum(MBON_type_layers.iloc[:, 0]>0)/len(MBON_type_layers.iloc[:, 0]), sum(MBON_type_layers.iloc[:, 1]>0)/len(MBON_type_layers.iloc[:, 1])])
MBON_dVNC, MBON_dVNC_plotting = br_adj.hop_matrix(MBON_type_skids.T, dVNC_pairs.leftid, Promat.extract_pairs_from_list(MBON, pairs)[0].leftid)

# %%
# specific interactions between dVNCs with phenotypes and a couple selected dVNCs

from connectome_tools.cascade_analysis import Celltype, Celltype_Analyzer

dVNC_important = [[17353986, np.where(dVNC_pairs.leftid==17353986)[0][0], 'backup', 'unpublished', 'dVNC'],
                    [10728328, np.where(dVNC_pairs.leftid==10728328)[0][0], 'backup', 'published', 'dVNC'],
                    [10728333, np.where(dVNC_pairs.leftid==10728333)[0][0], 'backup', 'published', 'dVNC'],
                    [6446394, np.where(dVNC_pairs.leftid==6446394)[0][0], 'stop', 'published', 'dVNC'],
                    [10382686, np.where(dVNC_pairs.leftid==10382686)[0][0], 'stop', 'unpublished', 'dSEZ'],
                    [16851496, np.where(dVNC_pairs.leftid==16851496)[0][0], 'cast', 'unpublished', 'dVNC'],
                    [10553248, np.where(dVNC_pairs.leftid==10553248)[0][0], 'cast', 'unpublished', 'dVNC'],
                    [3044500, np.where(dVNC_pairs.leftid==3044500)[0][0], 'cast_onset_offset', 'unpublished', 'dSEZ'],
                    [3946166, np.where(dVNC_pairs.leftid==3946166)[0][0], 'cast_onset_offset', 'unpublished', 'dVNC']]

dVNC_important = pd.DataFrame(dVNC_important, columns=['leftid', 'index', 'behavior', 'status', 'celltype'])
#dVNC_exclusive = dVNC_important.loc[dVNC_important.celltype=='dVNC']
#dVNC_exclusive.reset_index(inplace=True, drop=True)
#dVNC_important_us = all_type_layers_skids.loc[:, dVNC_exclusive.leftid]

#dVNC_important_us.iloc[0, :]

# check overlap between us networks
us_cts = []
for i in range(len(dVNC_important_us.index)):
    for j in range(len(dVNC_important_us.columns)):
        cts = Celltype(f'{dVNC_exclusive.behavior[j]} {dVNC_important_us.columns[j]} {i+1}-order', dVNC_important_us.iloc[i, j])
        us_cts.append(cts)

cta = Celltype_Analyzer(us_cts)
sns.heatmap(cta.compare_membership(), annot=True, fmt='.0%')

# number of neurons in us networks
us_1order = [x for sublist in dVNC_important_us.loc[0] for x in sublist]
us_2order = [x for sublist in dVNC_important_us.loc[1] for x in sublist]

us_1order_unique = np.unique(us_1order)
us_2order_unique = np.unique(us_2order)

pymaid.add_annotations(dVNC_important.leftid.values, 'mw dVNC important')
pymaid.add_annotations([pairs[pairs.leftid==x].rightid.values[0] for x in dVNC_important.leftid.values], 'mw dVNC important')
pymaid.add_annotations(us_1order_unique, 'mw dVNC important 1st-order upstream')

for i, us in enumerate(dVNC_important_us.loc[0]):
    pymaid.add_annotations(us, f'mw dVNC upstream-1o {dVNC_important_us.columns[i]}')
    pymaid.add_meta_annotations(f'mw dVNC upstream-1o {dVNC_important_us.columns[i]}', 'mw dVNC upstream-1o')
# %%
