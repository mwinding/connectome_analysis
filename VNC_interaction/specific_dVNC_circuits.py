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

rm = pymaid.CatmaidInstance(url, name, password, token)
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
pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC')
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

cluster = sns.clustermap(dVNC_projectome_pairs_summed_output, col_cluster = False, figsize=(10,10))
row_order = cluster.dendrogram_row.reordered_ind

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
# %%
# dVNC skeletons with associated split-GAL4 lines

dVNC_pairs_line = [[18604097, 17353986, pymaid.get_names(18604097)[str(18604097)]],
                    [3979181, 5794678, pymaid.get_names(3979181)[str(3979181)]],
                    [10382686, 16100103, pymaid.get_names(10382686)[str(10382686)]],
                    [3044500, 6317793, pymaid.get_names(3044500)[str(3044500)]],
                    [3946166, 3620633, pymaid.get_names(3946166)[str(3946166)]],
                    [6446394, 5462159, pymaid.get_names(6446394)[str(6446394)]],
                    [10553248, 8496618, pymaid.get_names(10553248)[str(10553248)]],
                    [16851496, 16339338, pymaid.get_names(16851496)[str(16851496)]]]

dVNC_pairs_line = pd.DataFrame(dVNC_pairs_line, columns = ['leftid', 'rightid', 'leftname'])

dVNC_pairs = Promat.extract_pairs_from_list(dVNC, pairs)[0]
dVNC_pairs = dVNC_pairs.loc[row_order]
dVNC_pairs.reset_index(inplace=True, drop=True)
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

#dVNC_pair_paths.pop(81) # removes one descending neuron with no inputs from the brain
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
# multi-hop matrix of all cell types to dVNCs

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
