#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 6})

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)
adj = mg.adj  # adjacency matrix from the "mg" object

# repeat for other connection types
mg_aa = load_metagraph("Gaa", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_aa.calculate_degrees(inplace=True)
adj_aa = mg_aa.adj

mg_dd = load_metagraph("Gdd", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_dd.calculate_degrees(inplace=True)
adj_dd = mg_dd.adj

mg_da = load_metagraph("Gda", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_da.calculate_degrees(inplace=True)
adj_da = mg_da.adj

clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)
lvl7 = clusters.groupby('lvl7_labels')

# separate meta file with median_node_visits from sensory for each node
# determined using iterative random walks
meta_with_order = pd.read_csv('data/meta_data_w_order.csv', index_col = 0, header = 0)

order_df = []
for key in lvl7.groups:
    skids = lvl7.groups[key]
    node_visits = meta_with_order.loc[skids, :].median_node_visits
    order_df.append([key, np.nanmean(node_visits)])

order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
order_df = order_df.sort_values(by = 'node_visit_order')

order = list(order_df.cluster)

# %%
# pull sensory annotations and then pull associated skids
input_names = pymaid.get_annotated('mw brain inputs').name
input_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))
input_skids = [val for sublist in input_skids_list for val in sublist]

output_order = [1, 0, 2]
output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

output_names_reordered = [output_names[i] for i in output_order]
output_skids_list_reordered = [output_skids_list[i] for i in output_order]

# level 7 clusters
clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)
lvl7 = clusters.groupby('lvl7_labels')

meta_with_order = pd.read_csv('data/meta_data_w_order.csv', index_col = 0, header = 0)

# ordering by mean node visit from sensory
order_df = []
for key in lvl7.groups:
    skids = lvl7.groups[key]
    node_visits = meta_with_order.loc[skids, :].median_node_visits
    order_df.append([key, np.nanmean(node_visits)])

order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
order_df = order_df.sort_values(by = 'node_visit_order')

order = list(order_df.cluster)

# getting skids of each cluster
cluster_lvl7 = []
for key in order:
    cluster_lvl7.append(lvl7.groups[key].values)

# %%
## cascades from each cluster, ending at brain inputs/outputs
# maybe should switch to sensory second-order?

def skid_to_index(skid, mg):
    index_match = np.where(mg.meta.index == skid)[0]
    if(len(index_match)==1):
        return(index_match[0])
    if(len(index_match)!=1):
        print('Not one match for skid %i!' %skid)

from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert cluster skids to indices
cluster_lvl7_indices_list = []
for skids in cluster_lvl7:
    indices = []
    for skid in skids:
        index = skid_to_index(skid, mg)
        indices.append(index)
    cluster_lvl7_indices_list.append(indices)

#all_input_indices = np.where([x in input_skids for x in mg.meta.index])[0]
#all_output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = [],
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

cluster_hit_hist_list = []
for indices in cluster_lvl7_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = indices)
    cluster_hit_hist_list.append(hit_hist)

# aa
transition_probs = to_transmission_matrix(adj_aa, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = [],
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

cluster_hit_hist_list_aa = []
for indices in cluster_lvl7_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = indices)
    cluster_hit_hist_list_aa.append(hit_hist)

# dd
transition_probs = to_transmission_matrix(adj_dd, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = [],
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

cluster_hit_hist_list_dd = []
for indices in cluster_lvl7_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = indices)
    cluster_hit_hist_list_dd.append(hit_hist)

# da
transition_probs = to_transmission_matrix(adj_da, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = [],
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

cluster_hit_hist_list_da = []
for indices in cluster_lvl7_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = indices)
    cluster_hit_hist_list_da.append(hit_hist)

# %%
# categorize neurons in each cluster cascade by cluster

def hit_hist_to_clusters(hit_hist_list, lvl7, order):
    # breaking signal cascades into cluster groups
    output_hit_hist_lvl7 = []
    for hit_hist in hit_hist_list:
        clustered_hist = []

        for key in order:
            skids = lvl7.groups[key]
            indices = np.where([x in skids for x in mg.meta.index])[0]
            cluster_hist = hit_hist[indices]
            cluster_hist = pd.DataFrame(cluster_hist, index = indices)

            clustered_hist.append(cluster_hist)
        
        output_hit_hist_lvl7.append(clustered_hist)
    
    return(output_hit_hist_lvl7)

def sum_cluster_hit_hist(hit_hist_cluster, order):
    # summed signal cascades per cluster group (hops remain intact)
    summed_hist = []
    for hit_hist in hit_hist_cluster:
        sum_hist = []
        for i, cluster in enumerate(hit_hist):
            sum_cluster = cluster.sum(axis = 0)/(len(cluster.index)) # normalize by number of neurons in cluster
            sum_hist.append(sum_cluster)

        sum_hist = pd.DataFrame(sum_hist) # column names will be hop number
        sum_hist.index = order # uses cluster name for index of each summed cluster row
        summed_hist.append(sum_hist)

    return(summed_hist)

def alt_sum_cluster(summed_hops_hist_lvl7):
    
    alt_summed_hops_hist_lvl7 = []
    for hop in summed_hops_hist_lvl7[0].columns:
        summed_hist_lvl7 = []
        for hit_hist in summed_hops_hist_lvl7:
            summed_hist_lvl7.append(hit_hist.iloc[:, hop])

        summed_hist_lvl7 = pd.DataFrame(summed_hist_lvl7, index = summed_hops_hist_lvl7[0].index).T
        alt_summed_hops_hist_lvl7.append(summed_hist_lvl7)
    
    return(alt_summed_hops_hist_lvl7)

cluster_hit_hist_lvl7 = hit_hist_to_clusters(cluster_hit_hist_list, lvl7, order)
summed_hops_hist_lvl7 = sum_cluster_hit_hist(cluster_hit_hist_lvl7, order)
alt_summed_hops_hist_lvl7 = alt_sum_cluster(summed_hops_hist_lvl7)

cluster_hit_hist_lvl7_aa = hit_hist_to_clusters(cluster_hit_hist_list_aa, lvl7, order)
summed_hops_hist_lvl7_aa = sum_cluster_hit_hist(cluster_hit_hist_lvl7_aa, order)
alt_summed_hops_hist_lvl7_aa = alt_sum_cluster(summed_hops_hist_lvl7_aa)

cluster_hit_hist_lvl7_dd = hit_hist_to_clusters(cluster_hit_hist_list_dd, lvl7, order)
summed_hops_hist_lvl7_dd = sum_cluster_hit_hist(cluster_hit_hist_lvl7_dd, order)
alt_summed_hops_hist_lvl7_dd = alt_sum_cluster(summed_hops_hist_lvl7_dd)

cluster_hit_hist_lvl7_da = hit_hist_to_clusters(cluster_hit_hist_list_da, lvl7, order)
summed_hops_hist_lvl7_da = sum_cluster_hit_hist(cluster_hit_hist_lvl7_da, order)
alt_summed_hops_hist_lvl7_da = alt_sum_cluster(summed_hops_hist_lvl7_da)
# %%
# plot visits to different groups, normalized to group size

# plot only first 3 hops
fig, axs = plt.subplots(
    1, 1, figsize = (8, 7)
)
ax = axs

sns.heatmap(alt_summed_hops_hist_lvl7[0] + alt_summed_hops_hist_lvl7[1] + alt_summed_hops_hist_lvl7[2], ax = ax, rasterized = True, square=True)
ax.set_ylabel('Individual Clusters')
ax.set_xlabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);

fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_clusters_3hops_ad.pdf', bbox_inches='tight')


# plot only first 3 hops
fig, axs = plt.subplots(
    1, 1, figsize = (8, 7)
)
ax = axs

sns.heatmap(alt_summed_hops_hist_lvl7_aa[0] + alt_summed_hops_hist_lvl7_aa[1] + alt_summed_hops_hist_lvl7_aa[2], ax = ax, rasterized = True, square=True)
ax.set_ylabel('Individual Clusters')
ax.set_xlabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);

fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_clusters_3hops_aa.pdf', bbox_inches='tight')


# plot only first 3 hops
fig, axs = plt.subplots(
    1, 1, figsize = (8, 7)
)
ax = axs

sns.heatmap(alt_summed_hops_hist_lvl7_dd[0] + alt_summed_hops_hist_lvl7_dd[1] + alt_summed_hops_hist_lvl7_dd[2], vmax = 20, ax = ax, rasterized = True, square=True)
ax.set_ylabel('Individual Clusters')
ax.set_xlabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);

fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_clusters_3hops_dd.pdf', bbox_inches='tight')


# plot only first 3 hops
fig, axs = plt.subplots(
    1, 1, figsize = (8, 7)
)
ax = axs
sns.heatmap(alt_summed_hops_hist_lvl7_da[0] + alt_summed_hops_hist_lvl7_da[1] + alt_summed_hops_hist_lvl7_da[2], vmax = 20, ax = ax, rasterized = True, square=True)
ax.set_ylabel('Individual Clusters')
ax.set_xlabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);

fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_clusters_3hops_da.pdf', bbox_inches='tight')

# %%
# plot visits to different groups, normalized to group size, displaying all hops

# plot only first 3 hops
panel_width = 10
panel_height = 9
fig, axs = plt.subplots(
    panel_width, panel_height, figsize = (30, 30), sharey = True
)

for x in range(len(summed_hops_hist_lvl7)):
    for j in range(panel_height):
        for i in range(panel_width):
            ax = axs[i, j]
            sns.heatmap(summed_hops_hist_lvl7[x], ax = ax, rasterized = True, cbar = False)
            ax.set_xlabel('Hops from source')
            ax.set_ylabel('Individual Clusters')
            ax.set_yticks([]);

fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_clusters_allhops.pdf', bbox_inches='tight')


# %%
# some examples

fig, axs = plt.subplots(
    1, 1, figsize = (5, 5)
)
sns.heatmap(summed_hops_hist_lvl7[0], rasterized = True, ax = axs)
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster0.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (5, 5)
)
sns.heatmap(summed_hops_hist_lvl7[40], rasterized = True, ax = axs)
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster40.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (5, 5)
)
sns.heatmap(summed_hops_hist_lvl7[50], rasterized = True, ax = axs)
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster50.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (5, 5)
)
sns.heatmap(summed_hops_hist_lvl7[86], rasterized = True, ax = axs)
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster86.pdf', bbox_inches='tight')

# %%
