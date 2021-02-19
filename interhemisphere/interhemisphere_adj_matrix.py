#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

import pymaid as pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, name, password, token)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import connectome_tools.process_matrix as pm
from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

sns.set_context("talk")

mg_ad = load_metagraph("Gad", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_ad.calculate_degrees(inplace=True)

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
pairs.drop(1121, inplace=True) # remove duplicate rightid

# load cluster data
clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)

# separate meta file with median_node_visits from sensory for each node
# determined using iterative random walks
meta_with_order = pd.read_csv('data/meta_data_w_order.csv', index_col = 0, header = 0)

def cluster_order(lvl_label_str, meta_with_order):
    lvl = clusters.groupby(lvl_label_str)
    order_df = []
    skids_df = []
    for key in lvl.groups:
        skids = lvl.groups[key]
        node_visits = meta_with_order.loc[skids, :].median_node_visits
        order_df.append([key, np.nanmean(node_visits)])
        skids_df.append([x for x in zip(skids, [key]*len(skids))])

    order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
    order_df = order_df.sort_values(by = 'node_visit_order')

    skids_df = pd.DataFrame([x for sublist in skids_df for x in sublist], columns = ['skid', 'cluster'])
    skids_df = skids_df.set_index('cluster', drop=False)#.loc[order_7]
    skids_df.index = range(0, len(skids_df.index))

    return(lvl, list(order_df.cluster), skids_df)

lvl7, order_7, skids_lvl7 = cluster_order('lvl7_labels', meta_with_order)


# %%
# order adj properly
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')
br_neurons = pymaid.get_skids_by_annotation('mw brain neurons')

ipsi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), adj.index))
bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), adj.index))
contra = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), adj.index))

ipsi_left = list(np.intersect1d(ipsi, left))
bilateral_left = list(np.intersect1d(bilateral, left))
contra_left = list(np.intersect1d(contra, left))

ipsi_right = list(np.intersect1d(ipsi, right))
bilateral_right = list(np.intersect1d(bilateral, right))
contra_right = list(np.intersect1d(contra, right))

adj = adj.loc[ipsi_left + bilateral_left + contra_left + contra_right + bilateral_right + ipsi_right, ipsi_left + bilateral_left + contra_left + contra_right + bilateral_right + ipsi_right]

meta_test = []
for i in range(len(adj.index)):
    if(adj.index[i] in br_neurons):
        meta_test.append(True)
    if(adj.index[i] not in br_neurons):
        meta_test.append(False)

meta = pd.DataFrame([True]*len(adj.index), columns = ['brain_neurons'], index = adj.index)

cell_type = []
for i in range(len(adj.index)):
    if(adj.index[i] in ipsi_left):
        cell_type.append('1_ipsi')
    if(adj.index[i] in bilateral_left):
        cell_type.append('2_bilateral')
    if(adj.index[i] in contra_left):
        cell_type.append('3_contra')
    if(adj.index[i] in ipsi_right):
        cell_type.append('6_ipsi')
    if(adj.index[i] in bilateral_right):
        cell_type.append('5_bilateral')
    if(adj.index[i] in contra_right):
        cell_type.append('4_contra')

meta['class'] = cell_type

cluster_type = []
for i in range(len(adj.index)):
    cluster_id = skids_lvl7[skids_lvl7.skid==adj.index[i]].cluster.values
    if(len(cluster_id)==1):
        cluster_type.append(cluster_id[0])
    if(len(cluster_id)==0):
        cluster_type.append('unknown')

meta['cluster'] = cluster_type

hemisphere_type = []
for i in range(len(adj.index)):
    if(adj.index[i] in left):
        hemisphere_type.append('L')
    if(adj.index[i] in right):
        hemisphere_type.append('R')

meta['hemisphere'] = hemisphere_type

pair_id = []
for skid in meta.index:
    if(meta.loc[skid].hemisphere=='L'):
        pair_id.append(skid)
    if((meta.loc[skid].hemisphere=='R') & (skid in (list(pairs.leftid) + list(pairs.rightid)))):
        pair_id.append(pm.Promat.identify_pair(skid, pairs))
    if((meta.loc[skid].hemisphere=='R') & (skid not in (list(pairs.leftid) + list(pairs.rightid)))):
        pair_id.append(skid)

meta['pair_id'] = pair_id

# %%
#

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
test = adjplot(
    adj.values,
    meta=meta,  
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order=['pair_id'],  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)
plt.savefig('interhemisphere/plots/raw_adj_matrix.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
test = adjplot(
    adj.values,
    meta=meta,
    sort_class='class',  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order=['cluster', 'pair_id'],  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)
plt.savefig('interhemisphere/plots/adj_matrix_sorted-class-cluster-pair.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
test = adjplot(
    adj.values,
    meta=meta,
    sort_class='cluster',  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order=['class', 'pair_id'],  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)
plt.savefig('interhemisphere/plots/adj_matrix_sorted-cluster-class-pair.pdf', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(15, 15))
test = adjplot(
    adj.values,
    meta=meta,
    sort_class='class',  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order=['pair_id'],  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)
plt.savefig('interhemisphere/plots/adj_matrix_sorted-class-pair.pdf', bbox_inches='tight')

# %%
# 

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

all_edges_combined_split = []
for i in range(len(all_edges_combined.index)):
    row = all_edges_combined.iloc[i]
    if((row.upstream_status=='paired') & (row.downstream_status=='paired')):
        if(row.type=='ipsilateral'):
            all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.type=='contralateral'):
            all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.left, row.type, row.upstream_status, row.downstream_status])
            all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])

    if((row.upstream_status=='nonpaired') & (row.downstream_status=='paired')):
        if(row.upstream_pair_id in left):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.upstream_pair_id in right):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])

    if((row.upstream_status=='paired') & (row.downstream_status=='nonpaired')):
        if(row.downstream_pair_id in left):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.downstream_pair_id in right):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'right', row.left, row.type, row.upstream_status, row.downstream_status])

    #if((row.upstream_status=='nonpaired') & (row.downstream_status=='nonpaired')):

all_edges_combined_split = pd.DataFrame(all_edges_combined_split, columns = ['upstream_pair_id', 'downstream_pair_id', 'upstream_side', 'downstream_side', 'edge_weight', 'type', 'upstream_status', 'downstream_status'])

# %%
# order matrix
immature = pymaid.get_skids_by_annotation('mw brain few synapses')

ipsi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), adj.index))
bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), adj.index))
contra = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), adj.index))

ipsi = list(np.setdiff1d(ipsi, immature))
bilateral = list(np.setdiff1d(bilateral, immature))
contra = list(np.setdiff1d(contra, immature))

ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)
bilateral_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)
contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)


ipsi_order_left = list(ipsi_pairs[0].leftid) + list(np.intersect1d(ipsi_pairs[2].nonpaired, left))
ipsi_order_right = list(ipsi_pairs[0].rightid) + list(np.intersect1d(ipsi_pairs[2].nonpaired, right))

bilateral_order_left = list(bilateral_pairs[0].leftid) + list(np.intersect1d(bilateral_pairs[2].nonpaired, left))
bilateral_order_right = list(bilateral_pairs[0].rightid) + list(np.intersect1d(bilateral_pairs[2].nonpaired, right))

contra_order_left = list(contra_pairs[0].leftid) + list(np.intersect1d(contra_pairs[2].nonpaired, left))
contra_order_right = list(contra_pairs[0].rightid) + list(np.intersect1d(contra_pairs[2].nonpaired, right))

contra_order_right.reverse()
bilateral_order_right.reverse()
ipsi_order_right.reverse()

order = ipsi_order_left + bilateral_order_left + contra_order_left + contra_order_right + bilateral_order_right + ipsi_order_right

adj = adj.loc[order, order]
meta_test = pd.DataFrame([True]*len(adj.index), columns = ['brain_neurons'], index = adj.index)

# %%
#


fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj.values,
    meta=meta_test,
    sort_class=None,  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(0.5, 3),  # min and max sizes for dots, so this is effectively binarizing
    item_order=None,  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)

# %%
