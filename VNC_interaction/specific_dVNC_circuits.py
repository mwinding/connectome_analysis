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
                    [10553248, 8496618, pymaid.get_names(10553248)[str(10553248)]],
                    [6446394, 5462159, pymaid.get_names(6446394)[str(6446394)]]]

dVNC_pairs_line = pd.DataFrame(dVNC_pairs_line, columns = ['leftid', 'rightid', 'leftname'])

dVNC_pairs = Promat.extract_pairs_from_list(dVNC, pairs)[0]
# %%
# paths 3-hop upstream of each dVNC
from tqdm import tqdm

hops = 2
threshold = 0.01

dVNC_pair_paths = []
for index in tqdm(range(0, len(dVNC_pairs_line))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_pairs_line.loc[index, ['leftid', 'rightid']]), threshold, min_members = 0, hops=hops, strict=False, use_edges=True)
    dVNC_pair_paths.append(us_dVNC)

MBONa1 = pymaid.get_skids_by_annotation('MBON-a1')
ds_MBON = br_adj.downstream_multihop(MBONa1, threshold, min_members=0, hops=hops, strict=False, use_edges=True)
ds_MBON_old = br_adj.downstream_multihop(MBONa1, threshold, min_members=0, hops=hops, strict=False, use_edges=False)

MBONa1 = pymaid.get_skids_by_annotation('MBON-a1')
us_MBON = br_adj.upstream_multihop(MBONa1, threshold, min_members=0, hops=hops, strict=False, use_edges=True)
us_MBON_old = br_adj.upstream_multihop(MBONa1, threshold, min_members=0, hops=hops, strict=False, use_edges=False)

'''
threshold = 0.02

dVNC_pair_paths_medium = []
for index in tqdm(range(0, len(dVNC_pairs_line))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_pairs_line.loc[index, ['leftid', 'rightid']]), threshold, min_members = 0, hops=hops, strict=True)
    dVNC_pair_paths_medium.append(us_dVNC)

threshold = 0.03

dVNC_pair_paths_strong = []
for index in tqdm(range(0, len(dVNC_pairs_line))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_pairs_line.loc[index, ['leftid', 'rightid']]), threshold, min_members = 0, hops=hops, strict=True)
    dVNC_pair_paths_strong.append(us_dVNC)

threshold = 0.04

dVNC_pair_paths_super_strong = []
for index in tqdm(range(0, len(dVNC_pairs_line))):
    us_dVNC = br_adj.upstream_multihop(list(dVNC_pairs_line.loc[index, ['leftid', 'rightid']]), threshold, min_members = 0, hops=hops, strict=True)
    dVNC_pair_paths_super_strong.append(us_dVNC)
'''

# %%
# plotting us dVNC paths

threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, br)
dVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, dVNC)
predVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, pre_dVNC)
dSEZ_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, dSEZ)
LHN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, LHN)
CN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, CN)
MBON_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, MBON)
MBIN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, MBIN)
FBN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, FBN_all)
KC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, KC)
PN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs_line.leftid, PN)

layer_types = [all_type_layers, PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, KC_type_layers, 
                FBN_type_layers, CN_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','FBN', 'CN', 'dSEZ', 'pre-dVNC', 'dVNC']
layer_colors = ['Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Reds', 'Purples', 'Reds']
layer_vmax = [300, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
save_path = 'VNC_interaction/plots/dVNC_upstream/'

plt.rcParams['font.size'] = 5

br_adj.plot_layer_types(layer_types=layer_types, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs_line.leftid, figsize=(.5, 1.5), save_path=save_path, threshold=threshold)
'''
threshold = 0.02
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, br)
dVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, dVNC)
predVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, pre_dVNC)
dSEZ_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, dSEZ)
LHN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, LHN)
CN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, CN)
MBON_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, MBON)
MBIN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, MBIN)
FBN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, FBN_all)
KC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, KC)
PN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_medium, dVNC_pairs_line.leftid, PN)

layer_types = [all_type_layers, PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, KC_type_layers, 
                FBN_type_layers, CN_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','FBN', 'CN', 'dSEZ', 'pre-dVNC', 'dVNC']
layer_colors = ['Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Reds', 'Purples', 'Reds']
layer_vmax = [300, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
save_path = 'VNC_interaction/plots/dVNC_upstream/'

plt.rcParams['font.size'] = 5

br_adj.plot_layer_types(layer_types=layer_types, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs_line.leftid, figsize=(.5, 1.5), save_path=save_path, threshold=threshold)

threshold = 0.03
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, br)
dVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, dVNC)
predVNC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, pre_dVNC)
dSEZ_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, dSEZ)
LHN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, LHN)
CN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, CN)
MBON_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, MBON)
MBIN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, MBIN)
FBN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, FBN_all)
KC_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, KC)
PN_type_layers,_ = br_adj.layer_id(dVNC_pair_paths_strong, dVNC_pairs_line.leftid, PN)

layer_types = [all_type_layers, PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, KC_type_layers, 
                FBN_type_layers, CN_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
layer_names = ['Total', 'PN', 'LHN', 'MBIN', 'MBON', 'KC','FBN', 'CN', 'dSEZ', 'pre-dVNC', 'dVNC']
layer_colors = ['Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Reds', 'Purples', 'Reds']
layer_vmax = [300, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
save_path = 'VNC_interaction/plots/dVNC_upstream/'

plt.rcParams['font.size'] = 5

br_adj.plot_layer_types(layer_types=layer_types, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs_line.leftid, figsize=(.5, 1.5), save_path=save_path, threshold=threshold)
'''
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
