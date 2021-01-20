#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass


from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random

import cmasher as cmr

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, name, password, token)

#adj = mg.adj  # adjacency matrix from the "mg" object
adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

# remove A1 except for ascendings
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

# load inputs and pair data
inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

# load projectome 
projectome = pd.read_csv('data/projectome.csv')

# load cluster data
clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)

# separate meta file with median_node_visits from sensory for each node
# determined using iterative random walks
meta_with_order = pd.read_csv('data/meta_data_w_order.csv', index_col = 0, header = 0)

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

order_df = []
for key in lvl7.groups:
    skids = lvl7.groups[key]
    node_visits = meta_with_order.loc[skids, :].median_node_visits
    order_df.append([key, np.nanmean(node_visits)])

order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
order_df = order_df.sort_values(by = 'node_visit_order')

order = list(order_df.cluster)

# %%
# ipsi/contra per cluster

# these annotations includes brain neurons, sensories, and some others
ipsi = pymaid.get_skids_by_annotation('mw ipsilateral')
contra = pymaid.get_skids_by_annotation('mw contralateral')

# integration types per cluster
cluster_lvl7 = [[key, lvl7.groups[key].values] for key in lvl7.groups.keys()]
cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'skids'])
cluster_lvl7.set_index('key', inplace=True)

ipsi_contra_clusters = []
for key in cluster_lvl7.index:
    ipsi_sum = len(np.intersect1d(cluster_lvl7.loc[key].skids, ipsi))/len(cluster_lvl7.loc[key].skids)
    contra_sum = len(np.intersect1d(cluster_lvl7.loc[key].skids, contra))/len(cluster_lvl7.loc[key].skids)
    ipsi_frac = ipsi_sum/(ipsi_sum+contra_sum)
    contra_frac = contra_sum/(ipsi_sum+contra_sum)
    ipsi_contra_clusters.append([key, ipsi_frac, contra_frac])

ipsi_contra_clusters = pd.DataFrame(ipsi_contra_clusters, columns = ['key', 'ipsi', 'contra'])
ipsi_contra_clusters.set_index('key', inplace=True)
ipsi_contra_clusters = ipsi_contra_clusters.loc[order, :]

ind = [x for x in range(0, len(cluster_lvl7))]
fig, ax = plt.subplots(1,1, figsize=(3,2))
plt.bar(ind, ipsi_contra_clusters.ipsi.values)
plt.bar(ind, ipsi_contra_clusters.contra.values, bottom = ipsi_contra_clusters.ipsi, color='violet')
fig.savefig('interhemisphere/plots/ipsi_contra_makeup_clusters.pdf', format='pdf', bbox_inches='tight')
# %%
# amount of ipsi/contra per cell type

# set cell types
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
RGN = pymaid.get_skids_by_annotation('mw RGN')
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
input_skids_list = list(map(pymaid.get_skids_by_annotation, input_names))
sens_all = [x for sublist in input_skids_list for x in sublist]
A00c = pymaid.get_skids_by_annotation('mw A00c')

asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')
asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
asc_all = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')

LHN = list(np.setdiff1d(LHN, FBN_all))
CN = list(np.setdiff1d(CN, LHN + FBN_all)) # 'CN' means exclusive CNs that are not FBN or LHN
dSEZ = list(np.setdiff1d(dSEZ, MBON + MBIN + LHN + CN + KC + dVNC + PN + FBN_all))
pre_dVNC = list(np.setdiff1d(pre_dVNC, MBON + MBIN + LHN + CN + KC + dSEZ + dVNC + PN + FBN_all + asc_all + sens_all + A00c)) # 'pre_dVNC' must have no other category assignment

celltypes = [list(np.setdiff1d(br, RGN)), sens_all, PN, LHN, MBIN, KC, MBON, FBN_all, CN, pre_dVNC, dSEZ, dVNC]
celltype_names = ['Total', 'Sens', 'PN', 'LHN', 'MBIN', 'KC', 'MBON', 'MB-FBN', 'CN', 'pre-dVNC', 'dSEZ', 'dVNC']

ipsi_contra_celltypes = []
unknown_list = []
for i, celltype in enumerate(celltypes):
    ipsi_sum = len(np.intersect1d(celltype, ipsi))
    contra_sum = len(np.intersect1d(celltype, contra))
    ipsi_frac = ipsi_sum/len(celltype)
    contra_frac = contra_sum/len(celltype)

    ipsi_contra_celltypes.append([celltype_names[i], ipsi_frac, contra_frac])

    unknown = np.setdiff1d(celltype, (ipsi+contra))
    unknown_list.append(unknown)

ipsi_contra_celltypes = pd.DataFrame(ipsi_contra_celltypes, columns = ['celltype', 'ipsi', 'contra'])

ind = [x for x in range(0, len(ipsi_contra_celltypes))]
fig, ax = plt.subplots(1,1, figsize=(2,2))
plt.bar(ind, ipsi_contra_celltypes.ipsi.values)
plt.bar(ind, ipsi_contra_celltypes.contra.values, bottom = ipsi_contra_celltypes.ipsi, color='violet')
plt.xticks(rotation=45, ha='right')
ax.set(xticklabels = celltype_names, xticks=np.arange(0, len(ipsi_contra_celltypes), 1))
fig.savefig('interhemisphere/plots/ipsi_contra_makeup_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# contra/bilateral character plot

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

projectome = pd.read_csv('interhemisphere/data/projectome_split.csv', index_col = 0, header = 0)

is_left = []
for skid in projectome.skeleton:
    if(skid in left):
        is_left.append([skid, 1])
    if(skid in right):
        is_left.append([skid, 0])
    if((skid not in right) & (skid not in left)):
        is_left.append([skid, -1])
is_left = pd.DataFrame(is_left, columns = ['skid', 'is_left'])

projectome['is_left']=is_left.is_left.values
proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['Brain Hemisphere left', 'Brain Hemisphere right'].sum()

right_contra_axon_outputs = proj_group.loc[(contra, 0, 1, 0), :] # right side, axon outputs
left_contra_axon_outputs = proj_group.loc[(contra, 1, 1, 0), :] # left side, axon outputs

right_den_inputs = proj_group.loc[(slice(None), 0, 0, 1), :] # right side, dendrite inputs
left_den_inputs = proj_group.loc[(slice(None), 1, 0, 1), :] # left side, dendrite inputs

right_contra_axon_outputs['ratio'] = right_contra_axon_outputs['Brain Hemisphere right']/right_contra_axon_outputs['Brain Hemisphere left']