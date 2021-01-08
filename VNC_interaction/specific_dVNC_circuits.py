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
dVNC_pairs_line = dVNC_pairs
# %%
# paths 2-hop upstream of each dVNC
from tqdm import tqdm

hops = 2
threshold = 0.01

dVNC_pair_paths = []
for index in tqdm(range(0, len(dVNC_pairs))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_pairs.loc[index, ['leftid', 'rightid']]), threshold, min_members = 0, hops=hops, strict=False)
    dVNC_pair_paths.append(us_dVNC)

#dVNC_pair_paths.pop(81) # removes one descending neuron with no inputs from the brain
# %%
# plotting us of dVNC paths
# for 

threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, list(adj.index))
dVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, dVNC)
predVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, pre_dVNC)
dSEZ_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, dSEZ)
asc_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, asc_all)
LHN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, LHN)
CN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, CN)
MBON_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, MBON)
MBIN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, MBIN)
FBN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, FBN_all)
KC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, KC)
PN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, PN)

layer_types = [all_type_layers, PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, KC_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
layer_colors = ['Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Blues', 'Reds', 'Purples', 'Reds']
layer_vmax = [500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
save_path = 'VNC_interaction/plots/dVNC_upstream/'

plt.rcParams['font.size'] = 5

br_adj.plot_layer_types(layer_types=layer_types, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs_line.leftid, figsize=(.5*hops/3, 1.5), save_path=save_path, threshold=threshold, hops=hops)

# %%
# make bar plots for 1-hop and 2-hop

ind = [x for x in range(0, len(dVNC_pair_paths))]

# determine fraction of neuron types in 1-hop and 2-hop
fraction_types = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
fraction_types = [x/layer_types[0] for x in fraction_types]
fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']
colors = ['blue', 'tab:green', 'orange', 'yellow', 'purple', 'olive', 'tab:blue', 'tab:brown', 'salmon', 'brown']

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
# multi-hop matrix of all cell types to dVNCs

threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, br)
dVNC_type_layers,dVNC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, dVNC)
predVNC_type_layers,predVNC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, pre_dVNC)
dSEZ_type_layers,dSEZ_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, dSEZ)
LHN_type_layers,LHN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, LHN)
CN_type_layers,CN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, CN)
MBON_type_layers,MBON_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, MBON)
MBIN_type_layers,MBIN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, MBIN)
FBN_type_layers,FBN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, FBN_all)
KC_type_layers,KC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, KC)
PN_type_layers,PN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, PN)

#sns.barplot(x = [1, 2], y = [sum(MBON_type_layers.iloc[:, 0]>0)/len(MBON_type_layers.iloc[:, 0]), sum(MBON_type_layers.iloc[:, 1]>0)/len(MBON_type_layers.iloc[:, 1])])
MBON_dVNC, MBON_dVNC_plotting = br_adj.hop_matrix(MBON_type_skids.T, dVNC_pairs_line.leftid, Promat.extract_pairs_from_list(MBON, pairs)[0].leftid)

# %%
