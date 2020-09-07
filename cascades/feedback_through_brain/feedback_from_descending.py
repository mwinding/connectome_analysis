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

#%%
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

pre_output_names = list(pymaid.get_annotated('mw pre-brain outputs').name)
pre_output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw pre-brain outputs').name))

#%%
# cascades from each output type, ending at brain inputs 
# maybe should switch to senosry second-order?
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import to_markov_matrix, RandomWalk
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
from joblib import Parallel, delayed

def run_cascade(i, cdispatch):
    return(cdispatch.multistart(start_nodes = i))

# convert skids to indices
output_indices_list = []
for skids in output_skids_list_reordered:
    indices = np.where([x in skids for x in mg.meta.index])[0]
    output_indices_list.append(indices)

pre_output_indices_list = []
for skids in pre_output_skids_list:
    indices = np.where([x in skids for x in mg.meta.index])[0]
    pre_output_indices_list.append(indices)

all_input_indices = np.where([x in input_skids for x in mg.meta.index])[0]
all_output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = all_input_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

output_hit_hist_list = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch) for i in output_indices_list)
pre_output_hit_hist_list = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch) for i in pre_output_indices_list)

'''
output_hit_hist_list = []
for indices in output_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = indices)
    output_hit_hist_list.append(hit_hist)

pre_output_hit_hist_list = []
for indices in pre_output_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = indices)
    pre_output_hit_hist_list.append(hit_hist)
'''

'''
# Random Walk comparison
max_hops = 10
n_init = 100
simultaneous = False
transition_probs = to_markov_matrix(adj)

cdispatch = TraverseDispatcher(
    RandomWalk,
    transition_probs,
    stop_nodes = all_input_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

output_hit_hist_list_rw = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch) for i in output_indices_list)
'''

# %%
# grouping cascade indices by cluster type

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

# cluster order and number of neurons per cluster
cluster_lvl7 = []
for key in lvl7.groups.keys():
    cluster_lvl7.append([key, len(lvl7.groups[key])])

cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'num_cluster'])

def hit_hist_to_clusters(hit_hist_list, lvl7):
    # breaking signal cascades into cluster groups
    output_hit_hist_lvl7 = []
    for hit_hist in hit_hist_list:
        clustered_hist = []

        for key in lvl7.groups.keys():
            skids = lvl7.groups[key]
            indices = np.where([x in skids for x in mg.meta.index])[0]
            cluster_hist = hit_hist[indices]
            cluster_hist = pd.DataFrame(cluster_hist, index = indices)

            clustered_hist.append(cluster_hist)
        
        output_hit_hist_lvl7.append(clustered_hist)
    
    return(output_hit_hist_lvl7)

def sum_cluster_hit_hist(hit_hist_cluster):
    # summed signal cascades per cluster group (hops remain intact)
    summed_hist = []
    for hit_hist in hit_hist_cluster:
        sum_hist = []
        for i, cluster in enumerate(hit_hist):
            sum_cluster = cluster.sum(axis = 0)/(len(cluster.index)) # normalize by number of neurons in cluster
            sum_hist.append(sum_cluster)

        sum_hist = pd.DataFrame(sum_hist) # column names will be hop number
        sum_hist.index = cluster_lvl7.key # uses cluster name for index of each summed cluster row
        summed_hist.append(sum_hist)

    return(summed_hist)

output_hit_hist_lvl7 = hit_hist_to_clusters(output_hit_hist_list, lvl7)
pre_output_hit_hist_lvl7 = hit_hist_to_clusters(pre_output_hit_hist_list, lvl7)

output_summed_hist_lvl7 = sum_cluster_hit_hist(output_hit_hist_lvl7)
pre_output_summed_hist_lvl7= sum_cluster_hit_hist(pre_output_hit_hist_lvl7)

#output_hit_hist_lvl7_rw = hit_hist_to_clusters(output_hit_hist_list_rw, lvl7)
#output_summed_hist_lvl7_rw = sum_cluster_hit_hist(output_hit_hist_lvl7_rw)


# number of neurons per cluster group over threshold (hops remain intact)
threshold = 50

num_hist_lvl7 = []
for hit_hist in output_hit_hist_lvl7:
    num_hist = []
    for i, cluster in enumerate(hit_hist):
        num_cluster = (cluster>threshold).sum(axis = 0) 
        num_hist.append(num_cluster)

    num_hist = pd.DataFrame(num_hist) # column names will be hop number
    num_hist.index = cluster_lvl7.key # uses cluster name for index of each summed cluster row
    num_hist_lvl7.append(num_hist)
# %%
# plot signal of all outputs through clusters

fig, axs = plt.subplots(
    1, 1, figsize=(5, 5)
)

vmax = 300

ax = axs
sns.heatmap(sum(output_summed_hist_lvl7).loc[order, 0:7], ax = ax, vmax = vmax, rasterized=True, cbar_kws={'label': 'Visits from sensory signal'})
ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_xlabel('Hops from output signal')

plt.savefig('cascades/feedback_through_brain/plots/summed_output_feedback_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize=(5, 5)
)

vmax = 300

ax = axs
sns.heatmap(sum(pre_output_summed_hist_lvl7).loc[order, 0:7], ax = ax, vmax = vmax, rasterized=True, cbar_kws={'label': 'Visits from sensory signal'})
ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_xlabel('Hops from pre-output signal')

plt.savefig('cascades/feedback_through_brain/plots/summed_pre-output_feedback_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

# plotting number of neurons downstream of each sensory modality (with threshold)
fig, axs = plt.subplots(
    1, 1, figsize=(5, 5)
)

ax = axs
sns.heatmap(sum(num_hist_lvl7).loc[order, 0:7], ax = ax, rasterized=True, cbar_kws={'label': 'Number of Neurons Downstream'})
ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_xlabel('Hops from output signal')

plt.savefig('cascades/feedback_through_brain/plots/summed_output_feedback_through_clusters_lvl7_num.pdf', format='pdf', bbox_inches='tight')
# %%
# plot signal of each output type through clusters
width = 2
height = 2
fig, axs = plt.subplots(
    2, 3, figsize=(width, height)
)

#fig.tight_layout(pad=1)
vmax = n_init

for i in range(0, len(output_names_reordered)):
    ax = axs[0, i]
    
    cbar = False
    sns.heatmap(output_summed_hist_lvl7[i].loc[order, 0:3], ax = ax, vmax = vmax, cbar = cbar, cbar_kws={'label': 'Average Number of Visits'})
    ax.set_ylabel('')
    if(i==0):
        ax.set_ylabel('Individual Clusters')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('%s' %output_names_reordered[i][3:])

    #sns.heatmap(summed_hist_lvl7[1].loc[sort], ax = ax, rasterized=True)

for i in range(0, len(pre_output_names)):
    ax = axs[1, i]
    cbar = False
    if(i==(len(pre_output_names)-1)):
        cbar = False
    sns.heatmap(pre_output_summed_hist_lvl7[i].loc[order, 0:3], ax = ax, vmax = vmax, cbar = cbar, cbar_kws={'label': 'Average Number of Visits'})
    ax.set_ylabel('')
    if(i==0):
        ax.set_ylabel('Individual Clusters')
    ax.set_yticks([])
    ax.set_xlabel('Hops')
    ax.set_title('%s' %pre_output_names[i][3:])

    #sns.heatmap(summed_hist_lvl7[1].loc[sort], ax = ax, rasterized=True)

plt.savefig('cascades/feedback_through_brain/plots/output_feedback_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

# %%
# single neuron perspective

# amount of neurons that receive feedback from dVNCs
#sum(output_hit_hist_list[0][:, 0:5].sum(axis=1)>50)

feedback_from_dVNC = np.where((output_hit_hist_list[0][:, 1:4]).sum(axis=1)>50)[0]
feedback_from_dSEZ = np.where((output_hit_hist_list[1][:, 1:4]).sum(axis=1)>50)[0]
feedback_from_RG = np.where((output_hit_hist_list[2][:, 1:4]).sum(axis=1)>50)[0]

feedback_from_pre_dVNC = np.where((pre_output_hit_hist_list[0][:, 1:4]).sum(axis=1)>50)[0]
feedback_from_pre_dSEZ = np.where((pre_output_hit_hist_list[1][:, 1:4]).sum(axis=1)>50)[0]
feedback_from_pre_RG = np.where((pre_output_hit_hist_list[2][:, 1:4]).sum(axis=1)>50)[0]

feedback_from_pre_dVNC = np.delete(feedback_from_pre_dVNC, all_output_indices)
feedback_from_pre_dSEZ = np.delete(feedback_from_pre_dSEZ, all_output_indices)
feedback_from_pre_RG = np.delete(feedback_from_pre_RG, all_output_indices)

iou_data = [feedback_from_dVNC, feedback_from_dSEZ, feedback_from_pre_dVNC, feedback_from_pre_dSEZ, feedback_from_pre_RG]
iou_matrix = np.zeros((len(iou_data), len(iou_data)))

for i in range(len(iou_data)):
    for j in range(len(iou_data)):
        if(len(np.union1d(iou_data[i], iou_data[j])) > 0):
            iou = len(np.intersect1d(iou_data[i], iou_data[j]))/len(np.union1d(iou_data[i], iou_data[j]))
            iou_matrix[i, j] = iou

iou_matrix = pd.DataFrame(iou_matrix, index = ['dVNC (%i)' %len(feedback_from_dVNC), 
                                                'dSEZ (%i)' %len(feedback_from_dSEZ),
                                                'pre-dVNC (%i)' %len(feedback_from_pre_dVNC), 
                                                'pre-dSEZ (%i)' %len(feedback_from_pre_dSEZ), 
                                                'pre-RG (%i)' %len(feedback_from_pre_RG)], columns = ['dVNC', 'dSEZ', 'pre-dVNC', 'pre-dSEZ', 'pre-RG'])

new_order = [0, 4, 1, 2, 3]
iou_matrix = iou_matrix.iloc[new_order, new_order]

fig, axs = plt.subplots(
    1, 1, figsize=(2.5, 2.5)
)

ax = axs
fig.tight_layout(pad=2.0)
sns.heatmap(iou_matrix, ax = ax, rasterized = True, square = True)
ax.set_title('Intersection of Feedback Types')

plt.savefig('cascades/feedback_through_brain/plots/output_FBN_centers.pdf', format='pdf', bbox_inches='tight')

# %%
# export feedback types

def index_to_skid(index, mg):
    return(mg.meta.iloc[index, :].name)

feedback_from_dVNC_skids = pd.DataFrame([index_to_skid(x, mg) for x in feedback_from_dVNC], columns = ['skid'])
feedback_from_dSEZ_skids = pd.DataFrame([index_to_skid(x, mg) for x in feedback_from_dSEZ], columns = ['skid'])
feedback_from_pre_dVNC_skids = pd.DataFrame([index_to_skid(x, mg) for x in feedback_from_pre_dVNC], columns = ['skid'])
feedback_from_pre_dSEZ_skids = pd.DataFrame([index_to_skid(x, mg) for x in feedback_from_pre_dSEZ], columns = ['skid'])
feedback_from_pre_RG_skids = pd.DataFrame([index_to_skid(x, mg) for x in feedback_from_pre_RG], columns = ['skid'])

feedback_from_dVNC_skids.to_csv('cascades/feedback_through_brain/plots/feedback_from_dVNC.csv', index = False)
feedback_from_dSEZ_skids.to_csv('cascades/feedback_through_brain/plots/feedback_from_dSEZ.csv', index = False)
feedback_from_pre_dVNC_skids.to_csv('cascades/feedback_through_brain/plots/feedback_from_predVNC.csv', index = False)
feedback_from_pre_dSEZ_skids.to_csv('cascades/feedback_through_brain/plots/feedback_from_predSEZ.csv', index = False)
feedback_from_pre_RG_skids.to_csv('cascades/feedback_through_brain/plots/feedback_from_preRGN.csv', index = False)


# %%
