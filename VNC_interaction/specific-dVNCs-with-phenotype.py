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
plt.rcParams['font.size'] = 5

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
general_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd']
input_skids_list = list(map(pymaid.get_skids_by_annotation, input_names))
sens_all = [x for sublist in input_skids_list for x in sublist]

asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')
asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
asc_all = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')

LHN = list(np.setdiff1d(LHN, FBN_all))
CN = list(np.setdiff1d(CN, LHN + FBN_all)) # 'CN' means exclusive CNs that are not FBN or LHN
dSEZ = list(np.setdiff1d(dSEZ, MBON + MBIN + LHN + CN + KC + dVNC + PN + FBN_all))
pre_dVNC = list(np.setdiff1d(pre_dVNC, MBON + MBIN + LHN + CN + KC + dSEZ + dVNC + PN + FBN_all + asc_all)) # 'pre_dVNC' must have no other category assignment

A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

br_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')

# %%
# load dVNCs with phenotypes

dVNC_important = [[17353986, pairs[pairs.leftid==17353986].rightid.values[0], 'backup', 'unpublished', 'dVNC'],
                    [10728328, pairs[pairs.leftid==10728328].rightid.values[0], 'backup', 'published', 'dVNC'],
                    [10728333, pairs[pairs.leftid==10728333].rightid.values[0], 'backup', 'published', 'dVNC'],
                    [6446394, pairs[pairs.leftid==6446394].rightid.values[0], 'stop', 'published', 'dVNC'],
                    [10382686, pairs[pairs.leftid==10382686].rightid.values[0], 'stop', 'unpublished', 'dSEZ'],
                    [16851496, pairs[pairs.leftid==16851496].rightid.values[0], 'cast', 'unpublished', 'dVNC'],
                    [10553248, pairs[pairs.leftid==10553248].rightid.values[0], 'cast', 'unpublished', 'dVNC'],
                    [3044500, pairs[pairs.leftid==3044500].rightid.values[0], 'cast_onset_offset', 'unpublished', 'dSEZ'],
                    [3946166, pairs[pairs.leftid==3946166].rightid.values[0], 'cast_onset_offset', 'unpublished', 'dVNC']]

dVNC_important = pd.DataFrame(dVNC_important, columns=['leftid', 'rightid', 'behavior', 'status', 'celltype'])
dVNC_important['names'] = [f'{dVNC_important.leftid[i]}-{dVNC_important.behavior[i]}' for i in range(len(dVNC_important))]
order = [0,1,2,6,5,4,3,7,8]
#order = dVNC_important.iloc[order, :].leftid.values
dVNC_important = dVNC_important.loc[order, :]
dVNC_important.reset_index(inplace=True)
# %%
# projectome plot and ordering

import cmasher as cmr

projectome = pd.read_csv('descending_neurons_analysis/data/projectome_adjacency.csv', index_col = 0, header = 0)
projectome.index = [str(x) for x in projectome.index]

# identify meshes
meshes = ['SEZ_left', 'SEZ_right', 'T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right', 'A1_left', 'A1_right', 'A2_left', 'A2_right', 'A3_left', 'A3_right', 'A4_left', 'A4_right', 'A5_left', 'A5_right', 'A6_left', 'A6_right', 'A7_left', 'A7_right', 'A8_left', 'A8_right']

pairOrder = []
for skid in (dVNC_important.loc[:, ['leftid', 'rightid']].values.flatten()):
    if(skid in pairs["leftid"].values):
        pair_skid = pairs["rightid"][pairs["leftid"]==skid].iloc[0]
        pairOrder.append(skid)
        pairOrder.append(pair_skid)

input_projectome = projectome.loc[meshes, [str(x) for x in pairOrder]]
output_projectome = projectome.loc[[str(x) for x in pairOrder], meshes]

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

#normalize # of presynaptic sites
dVNC_projectome_pairs_summed_output_norm = dVNC_projectome_pairs_summed_output.copy()
for i in range(len(dVNC_projectome_pairs_summed_output)):
    sum_row = sum(dVNC_projectome_pairs_summed_output_norm.iloc[i, :])
    for j in range(len(dVNC_projectome_pairs_summed_output.columns)):
        dVNC_projectome_pairs_summed_output_norm.iloc[i, j] = dVNC_projectome_pairs_summed_output_norm.iloc[i, j]/sum_row

dVNC_projectome_pairs_summed_output_norm.index = [int(x) for x in dVNC_projectome_pairs_summed_output_norm.index]
dVNC_projectome_pairs_summed_output.index = [int(x) for x in dVNC_projectome_pairs_summed_output.index]

# plot projectome
fig, ax = plt.subplots(1,1, figsize=(2, 1.375))
annotations = dVNC_projectome_pairs_summed_output.copy()
#annotations[annotations<10]=0
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(dVNC_projectome_pairs_summed_output, annot=annotations, fmt='s', ax=ax, vmax=100)
plt.xticks(rotation=45, ha='right')
plt.savefig('VNC_interaction/plots/dVNC_phenotype/projectome_output_plot.pdf', format='pdf', bbox_inches='tight')
# %%
# paths 2-hop upstream of each dVNC/dSEZ with phenotype
from tqdm import tqdm

hops = 2
threshold = 0.01

dVNC_pair_paths = []
for i in tqdm(range(0, len(dVNC_important))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_important.loc[i, ['leftid', 'rightid']]), threshold, min_members = 0, hops=hops, strict=False)
    dVNC_pair_paths.append(us_dVNC)

dVNC_pair_paths_output = []
for i in tqdm(range(0, len(dVNC_important))):
    ds_dVNC = br_adj.downstream_multihop(list(list(dVNC_important.loc[i, ['leftid', 'rightid']])), threshold, min_members = 0, hops=hops, strict=False)
    dVNC_pair_paths_output.append(ds_dVNC)
# %%
# individual dVNC paths

# UPSTREAM
threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, list(adj.index))
dVNC_type_layers,dVNC_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, dVNC)
predVNC_type_layers,predVNC_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, pre_dVNC)
dSEZ_type_layers,dSEZ_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, dSEZ)
asc_type_layers,asc_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, asc_all)
LHN_type_layers,LHN_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, LHN)
CN_type_layers,CN_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, CN)
MBON_type_layers,MBON_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, MBON)
MBIN_type_layers,MBIN_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, MBIN)
FBN_type_layers,FBN_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, FBN_all)
KC_type_layers,KC_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, KC)
PN_type_layers,PN_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_important.leftid, PN)

layer_skids = [all_type_layers_skids, dVNC_type_layers_skids, predVNC_type_layers_skids, dSEZ_type_layers_skids, 
                asc_type_layers_skids, LHN_type_layers_skids, CN_type_layers_skids, MBON_type_layers_skids, MBIN_type_layers_skids,
                FBN_type_layers_skids, KC_type_layers_skids, PN_type_layers_skids]
layer_types = [all_type_layers, PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, KC_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']


# DOWNSTREAM in brain
threshold = 0.01
all_type_layers_ds,all_type_layers_skids_ds = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, list(adj.index))
dVNC_type_layers_ds,dVNC_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, dVNC)
predVNC_type_layers_ds,predVNC_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, pre_dVNC)
dSEZ_type_layers_ds,dSEZ_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, dSEZ)
asc_type_layers_ds,asc_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, asc_all)
LHN_type_layers_ds,LHN_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, LHN)
CN_type_layers_ds,CN_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, CN)
MBON_type_layers_ds,MBON_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, MBON)
MBIN_type_layers_ds,MBIN_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, MBIN)
FBN_type_layers_ds,FBN_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, FBN_all)
KC_type_layers_ds,KC_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, KC)
PN_type_layers_ds,PN_type_layers_ds_skids = br_adj.layer_id(dVNC_pair_paths_output, dVNC_important.leftid, PN)

layer_types_ds = [all_type_layers_ds, PN_type_layers_ds, LHN_type_layers_ds, MBIN_type_layers_ds, MBON_type_layers_ds, KC_type_layers_ds, 
                FBN_type_layers_ds, CN_type_layers_ds, asc_type_layers_ds, dSEZ_type_layers_ds, predVNC_type_layers_ds, dVNC_type_layers_ds]

# %%
# make bar plots for 1-hop and 2-hop
plt.rcParams['font.size'] = 5

# UPSTREAM
ind = [x for x in range(0, len(dVNC_pair_paths))]

# determine fraction of neuron types in 1-hop and 2-hop
fraction_types = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
fraction_types = [x/layer_types[0] for x in fraction_types]
fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
colors = ['#1D79B7', '#D4E29E', '#FF8734', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC', '#D88052', '#E0B1AD', '#A52A2A']

# summary plot of 1st order upstream of dVNCs
fig, ax = plt.subplots(figsize=(2,1.5))
plt.bar(ind, fraction_types[0].iloc[:, 0], color = colors[0])
bottom = fraction_types[0].iloc[:, 0]

for i in range(1, len(fraction_types)):
    plt.bar(ind, fraction_types[i].iloc[:, 0], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types[i].iloc[:, 0]

remaining = [1 for x in range(0, len(dVNC_pair_paths))] - bottom
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.xticks(rotation=45, ha='right')
ax.set(xticklabels = dVNC_important.names, xticks=np.arange(0, 10, 1))
plt.ylim(0,1)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/summary_plot_1st_order.pdf', format='pdf', bbox_inches='tight')

# summary plot of 2nd order upstream of dVNCs
fig, ax = plt.subplots(figsize=(2,1.5))
plt.bar(ind, fraction_types[0].iloc[:, 1], color = colors[0])
bottom = fraction_types[0].iloc[:, 1]

for i in range(1, len(fraction_types)):
    plt.bar(ind, fraction_types[i].iloc[:, 1], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types[i].iloc[:, 1]

remaining = [1 for x in range(0, len(dVNC_pair_paths))] - bottom
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.xticks(rotation=45, ha='right')
ax.set(xticklabels = dVNC_important.names, xticks=np.arange(0, 10, 1))
plt.ylim(0,1)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/summary_plot_2nd_order.pdf', format='pdf', bbox_inches='tight')

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
fig, ax = plt.subplots(figsize=(2,1.5))
plt.bar(ind, fraction_types_ds[0].iloc[:, 0], color = colors[0])
bottom = fraction_types_ds[0].iloc[:, 0]

for i in range(1, len(fraction_types_ds)):
    plt.bar(ind, fraction_types_ds[i].iloc[:, 0], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types_ds[i].iloc[:, 0]

bottom[bottom==-10]=1
remaining = [1 for x in range(0, len(dVNC_pair_paths_output))] - bottom
#remaining = [x if x!=-1 else 0.0 for x in remaining]
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.xticks(rotation=45, ha='right')
ax.set(xticklabels = dVNC_important.names, xticks=np.arange(0, 10, 1))
plt.ylim(0,1)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/downstream_summary_plot_1st_order.pdf', format='pdf', bbox_inches='tight')

# summary plot of 2nd order downstream of dVNCs in brain
fig, ax = plt.subplots(figsize=(2,1.5))
plt.bar(ind, fraction_types_ds[0].iloc[:, 1], color = colors[0])
bottom = fraction_types_ds[0].iloc[:, 1]

for i in range(1, len(fraction_types_ds)):
    plt.bar(ind, fraction_types_ds[i].iloc[:, 1], bottom = bottom, color = colors[i])
    bottom = bottom + fraction_types_ds[i].iloc[:, 1]

bottom[bottom==-10]=1
remaining = [1 for x in range(0, len(dVNC_pair_paths_output))] - bottom
#remaining = [x if x!=-1 else 0.0 for x in remaining]
plt.bar(ind, remaining, bottom = bottom, color = 'tab:grey')
plt.xticks(rotation=45, ha='right')
ax.set(xticklabels = dVNC_important.names, xticks=np.arange(0, 10, 1))
plt.ylim(0,1)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/downstream_summary_plot_2nd_order.pdf', format='pdf', bbox_inches='tight')

# %%
# cosine similarity of all inputs and outputs
import cmasher as cmr 

# cosine similarity function
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)

    return(cos)

all_skids_us = np.unique([x for sublist in all_type_layers_skids.values.flatten() for x in sublist])
all_skids_us_paired = Promat.extract_pairs_from_list(all_skids_us, pairs)[0]
all_skids_ds = np.unique([x for sublist in all_type_layers_skids_ds.values.flatten() for x in sublist])
all_skids_ds_paired = Promat.extract_pairs_from_list(all_skids_ds, pairs)[0]

ds = br_adj.hop_matrix(all_type_layers_skids_ds.T, dVNC_important.leftid.values, all_skids_ds_paired.leftid.values)
us = br_adj.hop_matrix(all_type_layers_skids.T, dVNC_important.leftid.values, all_skids_us_paired.leftid.values)

cos_mat_us = np.zeros(shape=(len(us[0]), len(us[0])))
for i in range(len(us[0])):
    for j in range(len(us[0])):
        cos_mat_us[i, j] = cosine_similarity(us[0].iloc[i].values, us[0].iloc[j].values)

cos_mat_us = pd.DataFrame(cos_mat_us, index = dVNC_important.leftid.values, 
                            columns = dVNC_important.leftid.values)
annotations = cos_mat_us.copy()
annotations[annotations<.2]=0
annotations = annotations.round(2).astype(str)
annotations[annotations=='0.0']=''
fig, ax = plt.subplots(1,1, figsize=(2, 2))
sns.heatmap(cos_mat_us, annot=annotations, fmt='s', ax=ax, cmap = cmr.freeze, cbar=False)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/cosine_similarity_upstream.pdf', format='pdf', bbox_inches='tight')


cos_mat_ds = np.zeros(shape=(len(ds[0]), len(ds[0])))
for i in range(len(ds[0])):
    for j in range(len(ds[0])):
        cos_mat_ds[i, j] = cosine_similarity(ds[0].iloc[i].values, ds[0].iloc[j].values)

cos_mat_ds = pd.DataFrame(cos_mat_ds, index = dVNC_important.leftid.values, 
                            columns = dVNC_important.leftid.values)
cos_mat_ds.fillna(0, inplace=True)
annotations = cos_mat_ds.copy()
annotations[annotations<.2]=0
annotations = annotations.round(2).astype(str)
annotations[annotations=='0.0']=''
fig, ax = plt.subplots(1,1, figsize=(2, 2))
sns.heatmap(cos_mat_ds, annot=annotations, fmt='s', ax=ax, cmap = cmr.ember, cbar=False)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/cosine_similarity_feedback.pdf', format='pdf', bbox_inches='tight')

us_temp = us[0]
ds_temp = ds[0]
us_temp.columns = [f'{x}-us' for x in us_temp.columns]
ds_temp.columns = [f'{x}-ds' for x in ds_temp.columns]
combined = pd.concat([us[0], ds[0]], axis=1)

cos_mat = np.zeros(shape=(len(ds[0]), len(ds[0])))
for i in range(len(combined)):
    for j in range(len(combined)):
        cos_mat[i, j] = cosine_similarity(combined.iloc[i].values, combined.iloc[j].values)

cos_mat = pd.DataFrame(cos_mat, index = dVNC_important.leftid.values,
                        columns = dVNC_important.leftid)
annotations = cos_mat.copy()
annotations[annotations<.2]=0
annotations = annotations.round(2).astype(str)
annotations[annotations=='0.0']=''
fig, ax = plt.subplots(1,1, figsize=(2, 2))
sns.heatmap(cos_mat, annot=annotations, fmt='s', ax=ax, cmap = cmr.rainforest, cbar=False)
plt.savefig('VNC_interaction/plots/dVNC_phenotype/cosine_similar_upstream_feedback.pdf', format='pdf', bbox_inches='tight')

# %%
# hop matrices per PN type, per MBON type
# todo: rows colored by valence

us = br_adj.hop_matrix(all_type_layers_skids.T, dVNC_important.leftid.values, all_skids_us_paired.leftid.values)
us = us[0].T

MBONavers = pymaid.get_skids_by_annotation('mw MBON subclass_aversive')
MBONapp = pymaid.get_skids_by_annotation('mw MBON subclass_appetitive')

MBONexc_app = pymaid.get_skids_by_annotation('mw MBON excit Appet')
MBONinb_av = pymaid.get_skids_by_annotation('mw MBON inhib Avers')
MBONexc_av = pymaid.get_skids_by_annotation('mw MBON excit Avers')
MBONinb_app = pymaid.get_skids_by_annotation('mw MBON inhib Appet')
MBON_other = np.setdiff1d(MBON, MBONexc_app + MBONinb_av + MBONexc_av + MBONinb_app)

# inputs from MBONs
cts_names = ['MBON-app excitatory', 'MBON-avers inhibitory', 'MBON-avers excitatory', 'MBON-app inhibitory', 'MBON-other']
cts_plot_1o = []
cts_plot_2o = []
for celltype in [MBONexc_app, MBONinb_av, MBONexc_av, MBONinb_app, MBON_other]:
    data = us.loc[np.intersect1d(us.index, celltype)]
    cts_plot_1o.append((data==1).sum(axis=0)) # count number of neurons upstream 1o
    cts_plot_2o.append((data==2).sum(axis=0)) # count number of neurons upstream 2o

cts_plot_1o = pd.concat(cts_plot_1o, axis=1).T
cts_plot_2o = pd.concat(cts_plot_2o, axis=1).T
cts_plot_1o.index = cts_names
cts_plot_2o.index = cts_names

fig, axs = plt.subplots(1,2, figsize=(2, .75), sharey=True)
ax = axs[0]
annotations = cts_plot_1o.copy()
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(cts_plot_1o,  annot=annotations, fmt='s', ax=ax, cbar=False)
ax.set(title = 'First-order Upstream')

ax = axs[1]
annotations = cts_plot_2o.copy()
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(cts_plot_2o,  annot=annotations, fmt='s', ax=ax, cbar=False)
ax.set(title = 'Second-order Upstream')
plt.savefig('VNC_interaction/plots/dVNC_phenotype/MBON_inputs.pdf', format='pdf', bbox_inches='tight')

asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')

# inputs from PNs/ascendings
cts_names = ['uPN', 'tPN', 'vPN', 'mPN', 'asc_noci', 'asc_classII_III', 'asc_mechano', 'asc_proprio']
cts_plot_1o = []
cts_plot_2o = []
for celltype in [uPN, tPN, vPN, mPN, asc_noci, asc_classII_III, asc_mechano, asc_proprio]:
    data = us.loc[np.intersect1d(us.index, celltype)]
    cts_plot_1o.append((data==1).sum(axis=0)) # count number of neurons upstream 1o
    cts_plot_2o.append((data==2).sum(axis=0)) # count number of neurons upstream 2o

cts_plot_1o = pd.concat(cts_plot_1o, axis=1).T
cts_plot_2o = pd.concat(cts_plot_2o, axis=1).T
cts_plot_1o.index = cts_names
cts_plot_2o.index = cts_names

fig, axs = plt.subplots(1,2, figsize=(2, .75), sharey=True)
ax = axs[0]
annotations = cts_plot_1o.copy()
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(cts_plot_1o,  annot=annotations, fmt='s', ax=ax, cbar=False)
ax.set(title = 'First-order Upstream')

ax = axs[1]
annotations = cts_plot_2o.copy()
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(cts_plot_2o,  annot=annotations, fmt='s', ax=ax, cbar=False)
ax.set(title = 'Second-order Upstream')
plt.savefig('VNC_interaction/plots/dVNC_phenotype/sens_inputs.pdf', format='pdf', bbox_inches='tight')


# inputs from each type
cts_names = ['uPN', 'tPN', 'vPN', 'mPN', 'asc_noci', 'asc_classII_III', 'asc_mechano', 'asc_proprio', 
            'MBON-app excitatory', 'MBON-avers inhibitory', 'MBON-avers excitatory', 'MBON-app inhibitory', 'MBON-other']
cts_plot_1o = []
cts_plot_2o = []
for celltype in [uPN, tPN, vPN, mPN, asc_noci, asc_classII_III, asc_mechano, asc_proprio,
                    MBONexc_app, MBONinb_av, MBONexc_av, MBONinb_app, MBON_other]:
    data = us.loc[np.intersect1d(us.index, celltype)]
    cts_plot_1o.append((data==1).sum(axis=0)) # count number of neurons upstream 1o
    cts_plot_2o.append((data==2).sum(axis=0)) # count number of neurons upstream 2o

cts_plot_1o = pd.concat(cts_plot_1o, axis=1).T
cts_plot_2o = pd.concat(cts_plot_2o, axis=1).T
cts_plot_1o.index = cts_names
cts_plot_2o.index = cts_names

fig, axs = plt.subplots(1,2, figsize=(2, 1.5), sharey=True)
ax = axs[0]
annotations = cts_plot_1o.copy()
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(cts_plot_1o,  annot=annotations, fmt='s', ax=ax, cbar=False)
ax.set(title = 'First-order Upstream')

ax = axs[1]
annotations = cts_plot_2o.copy()
annotations = annotations.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(cts_plot_2o,  annot=annotations, fmt='s', ax=ax, cbar=False)
ax.set(title = 'Second-order Upstream')
plt.savefig('VNC_interaction/plots/dVNC_phenotype/all_inputs_by_hops.pdf', format='pdf', bbox_inches='tight')

# %%
# cascades from each sensory modality

sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

from src.traverse import Cascade, to_transmission_matrix
from src.traverse import to_markov_matrix, RandomWalk
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
from joblib import Parallel, delayed

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

def run_cascade(i, cdispatch):
    return(cdispatch.multistart(start_nodes = i))

brain_outputs = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
brain_outputs = [x for sublist in brain_outputs for x in sublist]
brain_outputs_indices = np.where([x in brain_outputs for x in mg.meta.index])[0]

brain_inputs_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs and ascending').name))

brain_inputs_indices_list = []
for skids in brain_inputs_list:
    indices = np.where([x in skids for x in mg.meta.index])[0]
    brain_inputs_indices_list.append(indices)

p = 0.05
max_hops = 11
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj=adj.values, p=p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = brain_outputs_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

brain_inputs_hit_hist_list = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch) for i in brain_inputs_indices_list)

# %%
# plot number visits per DNs with phenotype

from connectome_tools.cascade_analysis import Cascade_Analyzer, Celltype, Celltype_Analyzer

dVNC_important['index_left'] = np.where([x in dVNC_important.leftid.values for x in mg.meta.index])[0]
dVNC_important['index_right'] = np.where([x in dVNC_important.rightid.values for x in mg.meta.index])[0]

sns.heatmap(brain_inputs_hit_hist_list[0][dVNC_important.loc[:, ['index_left', 'index_right']].values.flatten()])