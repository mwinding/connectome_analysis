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
adj = pd.read_csv('VNC_interaction/data/axon-dendrite.csv', header = 0, index_col = 0)
inputs = pd.read_csv('VNC_interaction/data/input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-09-22.csv', header = 0) # import pairs

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

order = [16, 0, 2, 11, 8, 1, 3, 5, 7, 12, 13, 9, 10, 15, 4, 6, 14]
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