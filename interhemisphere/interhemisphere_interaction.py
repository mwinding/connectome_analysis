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

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

import cmasher as cmr

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
#mg = load_metagraph("G", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

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

#%%
# pull brain input skids and then divide into left and right hemisphere

input_names = list(pymaid.get_annotated('mw brain inputs and ascending').name)
brain_inputs_list = list(map(pymaid.get_skids_by_annotation, input_names))
input_skids = [val for sublist in brain_inputs_list for val in sublist]
input_names_format = ['ORN', 'thermo', 'visual', 'AN', 'MN', 'vtd', 'asc-proprio', 'asc-mechano', 'asc-classII_III', 'asc-noci']

left_annot = pymaid.get_skids_by_annotation('mw left')
right_annot = pymaid.get_skids_by_annotation('mw right')

# need to switch several ascending neurons because they ascending contralateral
# including those in annotations: ['mw A1 ascending noci', 'mw A1 ascending proprio', 'mw A1 ascending mechano']
#   excluding: [2123422, 2784471]
neurons_to_flip = list(map(pymaid.get_skids_by_annotation, ['mw A1 ascending noci', 'mw A1 ascending proprio', 'mw A1 ascending mechano']))
neurons_to_flip = [x for sublist in neurons_to_flip for x in sublist]
neurons_to_flip = list(np.setdiff1d(neurons_to_flip, [2123422, 2784471]))
neurons_to_flip_left = [skid for skid in neurons_to_flip if skid in left_annot]
neurons_to_flip_right = [skid for skid in neurons_to_flip if skid in right_annot]

# removing neurons_to_flip and adding to the other side
left = list(np.setdiff1d(left_annot, neurons_to_flip_left)) + neurons_to_flip_right
right = list(np.setdiff1d(right_annot, neurons_to_flip_right)) + neurons_to_flip_left

# loading output neurons
output_names = list(pymaid.get_annotated('mw brain outputs').name)
brain_outputs_list = list(map(pymaid.get_skids_by_annotation, output_names))
output_skids = [val for sublist in brain_outputs_list for val in sublist]

# identify left and right side for each skid category
def split_hemilateral_to_indices(skids, left, right, skids_order):
    intersect_left = np.intersect1d(skids, left)
    indices_left = np.where([x in intersect_left for x in skids_order])[0]
    intersect_right = np.intersect1d(skids, right)
    indices_right = np.where([x in intersect_right for x in skids_order])[0]

    return(indices_left, indices_right, intersect_left, intersect_right)

# split according to left/right input type and identify indices of adj for cascade
inputs_split = [split_hemilateral_to_indices(skids, left, right, adj.index) for skids in brain_inputs_list]
outputs_split = [split_hemilateral_to_indices(skids, left, right, adj.index) for skids in brain_outputs_list]

#ORN_indices_left, ORN_indices_right, ORN_left, ORN_right = split_hemilateral_to_indices(ORN_skids, left, right, mg)
input_indices_left, input_indices_right, input_left, input_right = split_hemilateral_to_indices(input_skids, left, right, adj.index)
output_indices = np.where([x in output_skids for x in adj.index])[0]

#%%
# cascades from left and right ORNs
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
from joblib import Parallel, delayed

def run_cascade(i, cdispatch):
    return(cdispatch.multistart(start_nodes = i))

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj.values, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = output_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

all_inputs_hit_hist_left, all_inputs_hit_hist_right = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch) for i in [input_indices_left, input_indices_right])

inputs_hit_hist_list_left = Parallel(n_jobs=-1)(delayed(run_cascade)(i[0], cdispatch) for i in inputs_split)
inputs_hit_hist_list_right = Parallel(n_jobs=-1)(delayed(run_cascade)(i[1], cdispatch) for i in inputs_split)

# %%
# signal through ipsilateral and contralateral structures
ipsi = pymaid.get_skids_by_annotation('mw brain ipsilateral')
contra = pymaid.get_skids_by_annotation('mw brain contralateral')

ipsi_indices_left, ipsi_indices_right, ipsi_left, ipsi_right = split_hemilateral_to_indices(ipsi, left_annot, right_annot, adj.index)
contra_indices_left, contra_indices_right, contra_left, contra_right = split_hemilateral_to_indices(contra, left_annot, right_annot, adj.index)

# add sensories to ipsi and contral as appropriate
ipsi_indices_left = list(ipsi_indices_left) + [x for i, indices in enumerate(inputs_split) for x in indices[0] if i in [0,1,2,3,4,5,8]] + list(np.where([x in [2123422] for x in adj.index])[0])
ipsi_indices_right = list(ipsi_indices_right) + [x for i, indices in enumerate(inputs_split) for x in indices[1] if i in [0,1,2,3,4,5,8]] + list(np.where([x in [2784471] for x in adj.index])[0])

contra_indices_left = list(contra_indices_left) + list(np.where([x in neurons_to_flip_left for x in adj.index])[0])
contra_indices_right = list(contra_indices_right) + list(np.where([x in neurons_to_flip_right for x in adj.index])[0])

# %%
# number of ipsi or contra neurons visited per hop
# shows nicely the flow of information through two hemispheres
# folds left and right ipsilateral and left and right contralateral together

fig, axs = plt.subplots(
    len(inputs_hit_hist_list_left), 1, figsize=(2, 10)
)
threshold = n_init/2
fig.tight_layout(pad=2.5)

for i in range(len(inputs_hit_hist_list_left)):
    ipsi_same = (inputs_hit_hist_list_left[i][ipsi_indices_left]>threshold).sum(axis=0) + (inputs_hit_hist_list_right[i][ipsi_indices_right]>threshold).sum(axis=0)
    contra_same = (inputs_hit_hist_list_left[i][contra_indices_left]>threshold).sum(axis=0) + (inputs_hit_hist_list_right[i][contra_indices_right]>threshold).sum(axis=0)
    ipsi_opposite = (inputs_hit_hist_list_left[i][ipsi_indices_right]>threshold).sum(axis=0) + (inputs_hit_hist_list_right[i][ipsi_indices_left]>threshold).sum(axis=0)
    contra_opposite = (inputs_hit_hist_list_left[i][contra_indices_right]>threshold).sum(axis=0) + (inputs_hit_hist_list_right[i][contra_indices_left]>threshold).sum(axis=0)

    data = pd.DataFrame([ipsi_same, contra_same, ipsi_opposite, contra_opposite], 
                        index = ['Ipsilateral same side', 'Contralateral same side', 
                                'Ipsilateral opposite side', 'Contralateral opposite side'])

    ax = axs[i]
    ax.set_title(f'{input_names[i]}')
    sns.heatmap(data, ax = ax, annot=True, fmt="d", cbar = False)

fig.savefig('interhemisphere/plots/num_ipsicontra_ds_each_sensory.pdf', format='pdf', bbox_inches='tight')

# plot same with cascade from all sensories

fig, ax = plt.subplots(
    1, 1, figsize=(2, 1)
)
threshold = n_init/2
fig.tight_layout(pad=2.5)

ipsi_same = (all_inputs_hit_hist_left[ipsi_indices_left]>threshold).sum(axis=0) + (all_inputs_hit_hist_right[ipsi_indices_right]>threshold).sum(axis=0)
contra_same = (all_inputs_hit_hist_left[contra_indices_left]>threshold).sum(axis=0) + (all_inputs_hit_hist_right[contra_indices_right]>threshold).sum(axis=0)
ipsi_opposite = (all_inputs_hit_hist_left[ipsi_indices_right]>threshold).sum(axis=0) + (all_inputs_hit_hist_right[ipsi_indices_left]>threshold).sum(axis=0)
contra_opposite = (all_inputs_hit_hist_left[contra_indices_right]>threshold).sum(axis=0) + (all_inputs_hit_hist_right[contra_indices_left]>threshold).sum(axis=0)

data = pd.DataFrame([ipsi_same, contra_same, ipsi_opposite, contra_opposite], 
                    index = ['Ipsilateral same side', 'Contralateral same side', 
                            'Ipsilateral opposite side', 'Contralateral opposite side'])

sns.heatmap(data, ax = ax, annot=True, fmt="d", cbar = False)
fig.savefig('interhemisphere/plots/num_ipsicontra_all-sensory_folded-sides.pdf', format='pdf', bbox_inches='tight')

# %%
# identify integration neurons
# plot cascades with integration overlay

def intersect_stats(hit_hist1, hit_hist2, threshold, hops):
    intersect_hops = []
    total_hops = []

    for i in np.arange(0, hops):
        intersect = np.logical_and(hit_hist1[:,i]>threshold, hit_hist2[:,i]>threshold)
        total = np.logical_or(hit_hist1[:,i]>threshold, hit_hist2[:,i]>threshold)
        intersect_hops.append(intersect)
        total_hops.append(total)

    percent = []
    for i in np.arange(0, hops):
        percent.append(sum(intersect_hops[i])/sum(total_hops[i]))

    return(np.array(intersect_hops), np.array(total_hops), percent)

threshold = 50
hops = 10

inputs_intersect_list = [intersect_stats(inputs_hit_hist_list_left[i], inputs_hit_hist_list_right[i], threshold, hops) for i in range(len(inputs_hit_hist_list_left))]
all_inputs_intersect, all_inputs_total, all_inputs_percent = intersect_stats(all_inputs_hit_hist_left, all_inputs_hit_hist_right, threshold, hops)

# plot results
fig, axs = plt.subplots(
    3, 1, figsize=(1.5, 2.5), sharex=True
)
threshold = n_init/2
fig.tight_layout(pad=2.5)

ax = axs[0]
ipsi_left = (all_inputs_hit_hist_left[ipsi_indices_left]>threshold).sum(axis=0)
contra_left = (all_inputs_hit_hist_left[contra_indices_left]>threshold).sum(axis=0)
ipsi_right = (all_inputs_hit_hist_left[ipsi_indices_right]>threshold).sum(axis=0)
contra_right = (all_inputs_hit_hist_left[contra_indices_right]>threshold).sum(axis=0)

data_left = pd.DataFrame([ipsi_left, contra_left, contra_right, ipsi_right], index = ['Ipsi(L)', 'Contra(L)', 'Contra(R)', 'Ipsi(R)'])
sns.heatmap(data_left.iloc[:, 0:6], ax = ax, annot=True, fmt="d", cbar = False)

ax = axs[1]
ipsi_right = (all_inputs_hit_hist_right[ipsi_indices_right]>threshold).sum(axis=0)
contra_right = (all_inputs_hit_hist_right[contra_indices_right]>threshold).sum(axis=0)
ipsi_left = (all_inputs_hit_hist_right[ipsi_indices_left]>threshold).sum(axis=0)
contra_left = (all_inputs_hit_hist_right[contra_indices_left]>threshold).sum(axis=0)

data_right = pd.DataFrame([ipsi_left, contra_left, contra_right, ipsi_right], index = ['Ipsi(R)', 'Contra(R)', 'Contra(L)', 'Ipsi(L)'])
sns.heatmap(data_right.iloc[:, 0:6], ax = ax, annot=True, fmt="d", cbar = False)

ax = axs[2]
ipsi_left = all_inputs_intersect.T[ipsi_indices_left].sum(axis=0)/all_inputs_total.T[ipsi_indices_left].sum(axis=0)
contra_left = all_inputs_intersect.T[contra_indices_left].sum(axis=0)/all_inputs_total.T[contra_indices_left].sum(axis=0)
contra_right = all_inputs_intersect.T[contra_indices_right].sum(axis=0)/all_inputs_total.T[contra_indices_right].sum(axis=0)
ipsi_right = all_inputs_intersect.T[ipsi_indices_right].sum(axis=0)/all_inputs_total.T[ipsi_indices_right].sum(axis=0)

data = pd.DataFrame([ipsi_left, contra_left, contra_right, ipsi_right], index = ['Ipsi(L)', 'Contra(L)', 'Contra(R)', 'Ipsi(R)'])
data = data.fillna(0)
sns.heatmap(data.iloc[:, 0:6], ax = ax, annot=True, fmt=".0%", cbar = False, cmap = cmr.lavender)
fig.savefig('interhemisphere/plots/summary_intersect-plot.pdf', format='pdf', bbox_inches='tight')

# %%
# cascades and integration through clusters

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels_side')

# integration types per cluster
cluster_lvl7 = [[key, lvl7.groups[key].values] for key in lvl7.groups.keys()]
cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'num_cluster'])
cluster_lvl7.set_index('key', inplace=True)

def hit_hist_to_clusters(hit_hist_list, lvl7, adj):
    # breaking signal cascades into cluster groups
    output_hit_hist_lvl7 = []
    for hit_hist in hit_hist_list:
        clustered_hist = []
        for key in lvl7.groups.keys():
            skids = lvl7.groups[key]
            indices = np.where([x in skids for x in adj.index])[0]
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
        for i, cluster in enumerate(hit_hist): # removed raw normalization
            sum_cluster = cluster.sum(axis = 0)#/(len(cluster.index)) # normalize by number of neurons in cluster
            sum_hist.append(sum_cluster)

        sum_hist = pd.DataFrame(sum_hist) # column names will be hop number
        sum_hist.index = cluster_lvl7.index # uses cluster name for index of each summed cluster row
        summed_hist.append(sum_hist)

    return(summed_hist)

hops = 4

interhemi_hit_hist_lvl7 = hit_hist_to_clusters([all_inputs_hit_hist_left, all_inputs_hit_hist_right], lvl7, adj)
interhemi_hit_hist_lvl7_summed = sum_cluster_hit_hist(interhemi_hit_hist_lvl7)

interhemi_hit_hist_lvl7_summed = [hit_hist.iloc[:, 0:hops+1].sum(axis=1) for hit_hist in interhemi_hit_hist_lvl7_summed]

interhemi_hit_hist_lvl7_summed_total = interhemi_hit_hist_lvl7_summed[0] + interhemi_hit_hist_lvl7_summed[1]
interhemi_hit_hist_lvl7_norm = [(hit_hist/interhemi_hit_hist_lvl7_summed_total).fillna(0) for hit_hist in interhemi_hit_hist_lvl7_summed]

order_left = [x + 'L' for x in order]
order_right = [x + 'R' for x in order]
order_both = [[x + 'R', x + 'L'] for x in order]
order_both_unlisted = [x for sublist in order_both for x in sublist]

fig, ax = plt.subplots(1,1, figsize=(2,2))
ind = [x for x in range(0, len(interhemi_hit_hist_lvl7_summed[0]))]
plt.bar(ind, interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted])
plt.bar(ind, interhemi_hit_hist_lvl7_summed[1].loc[order_both_unlisted], bottom=interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted])
fig.savefig('interhemisphere/plots/left-right-visits_clusters.pdf', format='pdf', bbox_inches='tight')

control_raw_left = interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted]
control_raw_right = interhemi_hit_hist_lvl7_summed[1].loc[order_both_unlisted]

fig, ax = plt.subplots(1,1, figsize=(2,2))
ind = [x for x in range(0, len(interhemi_hit_hist_lvl7_summed[0]))]
plt.bar(ind, interhemi_hit_hist_lvl7_norm[0].loc[order_both_unlisted])
plt.bar(ind, interhemi_hit_hist_lvl7_norm[1].loc[order_both_unlisted], bottom=interhemi_hit_hist_lvl7_norm[0].loc[order_both_unlisted])
fig.savefig('interhemisphere/plots/left-right-visits_clusters_norm.pdf', format='pdf', bbox_inches='tight')

control_left = interhemi_hit_hist_lvl7_norm[0].loc[order_both_unlisted]
control_right = interhemi_hit_hist_lvl7_norm[1].loc[order_both_unlisted]
# %%
# plot cluster flow by each sensory modality

for i in range(len(inputs_hit_hist_list_left)):
    interhemi_hit_hist_lvl7 = hit_hist_to_clusters([inputs_hit_hist_list_left[i], inputs_hit_hist_list_right[i]], lvl7, adj)
    interhemi_hit_hist_lvl7_summed = sum_cluster_hit_hist(interhemi_hit_hist_lvl7)

    interhemi_hit_hist_lvl7_summed = [hit_hist.iloc[:, 0:hops+1].sum(axis=1) for hit_hist in interhemi_hit_hist_lvl7_summed]

    interhemi_hit_hist_lvl7_summed_total = interhemi_hit_hist_lvl7_summed[0] + interhemi_hit_hist_lvl7_summed[1]
    interhemi_hit_hist_lvl7_norm = [(hit_hist/interhemi_hit_hist_lvl7_summed_total).fillna(0) for hit_hist in interhemi_hit_hist_lvl7_summed]

    order_left = [x + 'L' for x in order]
    order_right = [x + 'R' for x in order]
    order_both = [[x + 'R', x + 'L'] for x in order]
    order_both_unlisted = [x for sublist in order_both for x in sublist]

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    ind = [x for x in range(0, len(interhemi_hit_hist_lvl7_summed[0]))]
    plt.bar(ind, interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted])
    plt.bar(ind, interhemi_hit_hist_lvl7_summed[1].loc[order_both_unlisted], bottom=interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted])
    fig.savefig(f'interhemisphere/plots/raw-{input_names_format[i]}_left-right-visits_clusters.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    ind = [x for x in range(0, len(interhemi_hit_hist_lvl7_summed[0]))]
    plt.bar(ind, interhemi_hit_hist_lvl7_norm[0].loc[order_both_unlisted])
    plt.bar(ind, interhemi_hit_hist_lvl7_norm[1].loc[order_both_unlisted], bottom=interhemi_hit_hist_lvl7_norm[0].loc[order_both_unlisted])
    fig.savefig(f'interhemisphere/plots/norm-{input_names_format[i]}left-right-visits_clusters.pdf', format='pdf', bbox_inches='tight')

# %%
# number of ipsi vs contra per cluster
# for all sensory modalities
# for individual sensory modalities

# added hit_hists from all sensories (left/right) to set of individual modalities
# this simplifies code 

import connectome_tools.cascade_analysis as casc
from connectome_tools.process_matrix import Promat
contra = pymaid.get_skids_by_annotation('mw contralateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')

threshold = n_init/2
hops=6

inputs_hit_hist_list_left.append(all_inputs_hit_hist_left)
inputs_hit_hist_list_right.append(all_inputs_hit_hist_right)

# has to be overthreshold for a particular hop
intersection_list = []
for i in range(len(inputs_hit_hist_list_left)):
    integrators = ((inputs_hit_hist_list_left[i]>threshold).sum(axis=1)>0) & ((inputs_hit_hist_list_right[i]>threshold).sum(axis=1)>0)
    integrators = np.where(integrators==True)[0]
    integrators = list(adj.index[integrators]) # converts to skid
    integrators = np.intersect1d(integrators, contra + bilateral) # only neurons that cross commissure
    integrators = Promat.extract_pairs_from_list(integrators, pairs)[0]
    intersection_list.append([x for sublist in integrators.values for x in sublist])

# has to be overthreshold for summed hops
intersection_summed_list = []
for i in range(len(inputs_hit_hist_list_left)):
    integrators = (inputs_hit_hist_list_left[i][:, 1:hops+1].sum(axis=1)>threshold) & (inputs_hit_hist_list_right[i][:, 1:hops+1].sum(axis=1)>threshold)
    integrators = np.where(integrators==True)[0]
    integrators = list(adj.index[integrators]) # converts to skid
    integrators = np.intersect1d(integrators, contra + bilateral) # only neurons that cross commissure
    integrators = Promat.extract_pairs_from_list(integrators, pairs)[0]
    intersection_summed_list.append([x for sublist in integrators.values for x in sublist])

inter_celltypes = list(map(lambda pair: casc.Celltype(pair[0], pair[1]), zip(input_names, intersection_list)))
inter_analyzer = casc.Celltype_Analyzer(inter_celltypes)
inter_iou = inter_analyzer.compare_membership('dice')
sns.clustermap(inter_iou, figsize=(5,5), annot=True, vmin=0)

inter_summed_celltypes = list(map(lambda pair: casc.Celltype(pair[0], pair[1]), zip(input_names, intersection_summed_list)))
inter_summed_analyzer = casc.Celltype_Analyzer(inter_summed_celltypes)
inter_summed_iou = inter_summed_analyzer.compare_membership('dice')
sns.clustermap(inter_summed_iou, figsize=(5,5), annot=True, vmin=0)

# %%
# how much signal is transmissed to contralateral side by bilaterals vs contralaterals?

def excise_cascade(excised_skids, input_skids, output_skids, brain_inputs_list, left, right, adj):
    adj_excised = adj.loc[np.setdiff1d(adj.index, excised_skids), np.setdiff1d(adj.index, excised_skids)]
    excised_input_indices_left, excised_input_indices_right, excised_input_left, excised_input_right = split_hemilateral_to_indices(input_skids, left, right, adj_excised.index)
    excised_output_indices = np.where([x in output_skids for x in adj_excised.index])[0]
    excised_inputs_split = [split_hemilateral_to_indices(skids, left, right, adj_excised.index) for skids in brain_inputs_list]

    p = 0.05
    max_hops = 6
    n_init = 100
    simultaneous = True

    transition_probs = to_transmission_matrix(adj_excised.values, p)
    cdispatch_excised = TraverseDispatcher(
        Cascade,
        transition_probs,
        stop_nodes = excised_output_indices,
        max_hops=max_hops,
        allow_loops = False,
        n_init=n_init,
        simultaneous=simultaneous,
    )

    excised_all_inputs_hit_hist_left, excised_all_inputs_hit_hist_right = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch_excised) for i in [excised_input_indices_left, excised_input_indices_right])
    excised_inputs_hit_hist_list_left = Parallel(n_jobs=-1)(delayed(run_cascade)(i[0], cdispatch_excised) for i in excised_inputs_split)
    excised_inputs_hit_hist_list_right = Parallel(n_jobs=-1)(delayed(run_cascade)(i[1], cdispatch_excised) for i in excised_inputs_split)

    return(excised_all_inputs_hit_hist_left, excised_all_inputs_hit_hist_right, excised_inputs_hit_hist_list_left, excised_inputs_hit_hist_list_right, adj_excised)

# make new adj matrices with excised nodes
contra_br = list(np.setdiff1d(contra, input_skids + output_skids))
bilateral_br = list(np.setdiff1d(bilateral, input_skids + output_skids))

# random set of neurons corresponding to number of contra/bi removed
br_interneurons = np.setdiff1d(adj.index, input_skids + output_skids + contra_br + bilateral_br)
np.random.seed(0)
random_nums = np.random.choice(len(br_interneurons), len(contra_br), replace = False)
random_contra_set = list(br_interneurons[random_nums])

np.random.seed(1)
random_nums = np.random.choice(len(br_interneurons), len(bilateral_br), replace = False)
random_bilateral_set = list(br_interneurons[random_nums])

dC_all_inputs_hit_hist_left, dC_all_inputs_hit_hist_right, dC_inputs_hit_hist_list_left, dC_inputs_hit_hist_list_right, adj_dContra = excise_cascade(contra_br, input_skids, output_skids, brain_inputs_list, left, right, adj)
dB_all_inputs_hit_hist_left, dB_all_inputs_hit_hist_right, dB_inputs_hit_hist_list_left, dB_inputs_hit_hist_list_right, adj_dBi = excise_cascade(bilateral_br, input_skids, output_skids, brain_inputs_list, left, right, adj)
dB_dC_all_inputs_hit_hist_left, dB_dC_all_inputs_hit_hist_right, dB_dC_inputs_hit_hist_list_left, dB_dC_inputs_hit_hist_list_right, adj_dB_dC = excise_cascade(contra_br + bilateral_br, input_skids, output_skids, brain_inputs_list, left, right, adj)
control_all_inputs_hit_hist_left, control_all_inputs_hit_hist_right, control_inputs_hit_hist_list_left, control_inputs_hit_hist_list_right, _ = excise_cascade([], input_skids, output_skids, brain_inputs_list, left, right, adj)

con_dC_all_inputs_hit_hist_left, con_dC_all_inputs_hit_hist_right, con_dC_inputs_hit_hist_list_left, con_dC_inputs_hit_hist_list_right, con_adj_dContra = excise_cascade(random_contra_set, input_skids, output_skids, brain_inputs_list, left, right, adj)
con_dB_all_inputs_hit_hist_left, con_dB_all_inputs_hit_hist_right, con_dB_inputs_hit_hist_list_left, con_dB_inputs_hit_hist_list_right, con_adj_dBi = excise_cascade(random_bilateral_set, input_skids, output_skids, brain_inputs_list, left, right, adj)
con_dB_dC_all_inputs_hit_hist_left, con_dB_dC_all_inputs_hit_hist_right, con_dB_dC_inputs_hit_hist_list_left, con_dB_dC_inputs_hit_hist_list_right, con_adj_dB_dC = excise_cascade(random_contra_set + random_bilateral_set, input_skids, output_skids, brain_inputs_list, left, right, adj)

# %%
# plot all sensory left/right cascades after removing contra or bilateral neurons

# plot cascades with excised contralateral brain neurons
def plot_process_excise_cascades(all_inputs_hit_hist_left, all_inputs_hit_hist_right, adj_excised, lvl7, hops, path, order=order):

    interhemi_hit_hist_lvl7 = hit_hist_to_clusters([all_inputs_hit_hist_left, all_inputs_hit_hist_right], lvl7, adj_excised)
    interhemi_hit_hist_lvl7_summed = sum_cluster_hit_hist(interhemi_hit_hist_lvl7)

    interhemi_hit_hist_lvl7_summed = [hit_hist.iloc[:, 0:hops+1].sum(axis=1) for hit_hist in interhemi_hit_hist_lvl7_summed]

    interhemi_hit_hist_lvl7_summed_total = interhemi_hit_hist_lvl7_summed[0] + interhemi_hit_hist_lvl7_summed[1]
    interhemi_hit_hist_lvl7_norm = [(hit_hist/interhemi_hit_hist_lvl7_summed_total).fillna(0) for hit_hist in interhemi_hit_hist_lvl7_summed]

    order_left = [x + 'L' for x in order]
    order_right = [x + 'R' for x in order]
    order_both = [[x + 'R', x + 'L'] for x in order]
    order_both_unlisted = [x for sublist in order_both for x in sublist]

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    ind = [x for x in range(0, len(interhemi_hit_hist_lvl7_summed[0]))]
    plt.bar(ind, interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted])
    plt.bar(ind, interhemi_hit_hist_lvl7_summed[1].loc[order_both_unlisted], bottom=interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted])
    fig.savefig(path + '.pdf', format='pdf', bbox_inches='tight')

    excised_contra_raw_left = interhemi_hit_hist_lvl7_summed[0].loc[order_both_unlisted]
    excised_contra_raw_right = interhemi_hit_hist_lvl7_summed[1].loc[order_both_unlisted]

    left_hits = interhemi_hit_hist_lvl7_norm[0].loc[order_both_unlisted]
    right_hits = interhemi_hit_hist_lvl7_norm[1].loc[order_both_unlisted]
    hits = (left_hits>0)|(right_hits>0)

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    ind = [x for x in range(0, len(left_hits[hits]))]
    plt.bar(ind, left_hits[hits])
    plt.bar(ind, right_hits[hits], bottom=left_hits[hits])
    fig.savefig(path + '_norm.pdf', format='pdf', bbox_inches='tight')

    excised_contra_data_left = left_hits[hits]
    excised_contra_data_right = right_hits[hits]

    return(excised_contra_raw_left, excised_contra_raw_right, excised_contra_data_left, excised_contra_data_right)

hops=4

excised_contra_raw_left, excised_contra_raw_right, excised_contra_data_left, excised_contra_data_right = plot_process_excise_cascades(dC_all_inputs_hit_hist_left, dC_all_inputs_hit_hist_right, adj_dContra, lvl7, hops, 'interhemisphere/plots/excised-Contra_left-right-visits_clusters')
excised_bi_raw_left, excised_bi_raw_right, excised_bi_data_left, excised_bi_data_right = plot_process_excise_cascades(dB_all_inputs_hit_hist_left, dB_all_inputs_hit_hist_right, adj_dBi, lvl7, hops, 'interhemisphere/plots/excised-bilateral_left-right-visits_clusters')
excised_bi_contra_raw_left, excised_bi_contra_raw_right, excised_bi_contra_data_left, excised_bi_contra_data_right = plot_process_excise_cascades(dB_dC_all_inputs_hit_hist_left, dB_dC_all_inputs_hit_hist_right, adj_dB_dC, lvl7, hops, 'interhemisphere/plots/excised-bilateral-contralateral_left-right-visits_clusters')
excised_control2_raw_left, excised_control2_raw_right, excised_control2_data_left, excised_control2_data_right = plot_process_excise_cascades(control_all_inputs_hit_hist_left, control_all_inputs_hit_hist_right, adj, lvl7, hops, 'interhemisphere/plots/excised-control_left-right-visits_clusters')

excised_controlContra_raw_left, excised_controlContra_raw_right, excised_controlContra_data_left, excised_controlContra_data_right = plot_process_excise_cascades(con_dC_all_inputs_hit_hist_left, con_dC_all_inputs_hit_hist_right, con_adj_dContra, lvl7, hops, 'interhemisphere/plots/excised-control-Contra_left-right-visits_clusters')
excised_controlBi_raw_left, excised_controlBi_raw_right, excised_controlBi_data_left, excised_controlBi_data_right = plot_process_excise_cascades(con_dB_all_inputs_hit_hist_left, con_dB_all_inputs_hit_hist_right, con_adj_dBi, lvl7, hops, 'interhemisphere/plots/excised-control-Bilateral_left-right-visits_clusters')
excised_controlBiContra_raw_left, excised_controlBiContra_raw_right, excised_controlBiContra_data_left, excised_controlBiContra_data_right = plot_process_excise_cascades(con_dB_dC_all_inputs_hit_hist_left, con_dB_dC_all_inputs_hit_hist_right, con_adj_dB_dC, lvl7, hops, 'interhemisphere/plots/excised-control-Bilateral-Contra_left-right-visits_clusters')

# %%
# Difference in signal between L/R clusters

diff_contra = abs(excised_contra_data_left - excised_contra_data_right)
diff_bi = abs(excised_bi_data_left - excised_bi_data_right)
diff_bi_contra = abs(excised_bi_contra_data_left - excised_bi_contra_data_right)
diff_control = abs(control_left - control_right)
diff_contra_control = abs(excised_controlContra_data_left - excised_controlContra_data_right)
diff_bi_control = abs(excised_controlBi_data_left - excised_controlBi_data_right)
diff_bi_contra_control = abs(excised_controlBiContra_data_left - excised_controlBiContra_data_right)

diff_contra = pd.DataFrame(zip(diff_contra, ['contra']*len(diff_contra)), columns = ['hemisphere_segregation', 'excised_type'])
diff_bi = pd.DataFrame(zip(diff_bi, ['bi']*len(diff_bi)), columns = ['hemisphere_segregation', 'excised_type'])
diff_bi_contra = pd.DataFrame(zip(diff_bi_contra, ['contra-bi']*len(diff_bi_contra)), columns = ['hemisphere_segregation', 'excised_type'])
diff_control = pd.DataFrame(zip(diff_control, ['control']*len(diff_control)), columns = ['hemisphere_segregation', 'excised_type'])
diff_contra_control = pd.DataFrame(zip(diff_contra_control, ['control_contra']*len(diff_contra_control)), columns = ['hemisphere_segregation', 'excised_type'])
diff_bi_control = pd.DataFrame(zip(diff_bi_control, ['control_bi']*len(diff_bi_control)), columns = ['hemisphere_segregation', 'excised_type'])
diff_bi_contra_control = pd.DataFrame(zip(diff_bi_contra_control, ['control_contra-bi']*len(diff_bi_contra_control)), columns = ['hemisphere_segregation', 'excised_type'])

data = pd.concat([diff_control, diff_contra_control, diff_contra, diff_bi_control, diff_bi, diff_bi_contra_control, diff_bi_contra])
fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.boxplot(x='excised_type', y='hemisphere_segregation', data=data, ax=ax, fliersize = 0.1, linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
fig.savefig('interhemisphere/plots/hemisphere_segregation_Excised-neurons_expanded.pdf', format='pdf', bbox_inches='tight')

data = pd.concat([diff_control, diff_contra, diff_bi, diff_bi_contra])
fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.boxplot(x='excised_type', y='hemisphere_segregation', data=data, ax=ax, fliersize = 0.1, linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
fig.savefig('interhemisphere/plots/hemisphere_segregation_Excised-neurons.pdf', format='pdf', bbox_inches='tight')

# %%
# Effect on total signal transmission

excised_bi_transmission = list(excised_bi_raw_right/control_raw_right) + list(excised_bi_raw_left/control_raw_left)
excised_contra_transmission = list(excised_contra_raw_right/control_raw_right) + list(excised_contra_raw_left/control_raw_left)
excised_bi_contra_transmission = list(excised_bi_contra_raw_right/control_raw_right) + list(excised_bi_contra_raw_left/control_raw_left)
control_transmission = list(excised_control2_raw_right/control_raw_right) + list(excised_control2_raw_left/control_raw_left)
control_bi_transmission = list(excised_controlBi_raw_right/control_raw_right) + list(excised_controlBi_raw_left/control_raw_left)
control_contra_transmission = list(excised_controlContra_raw_right/control_raw_right) + list(excised_controlContra_raw_left/control_raw_left)
control_bi_contra_transmission = list(excised_controlBiContra_raw_right/control_raw_right) + list(excised_controlBiContra_raw_left/control_raw_left)

excised_contra_transmission = pd.DataFrame(zip(excised_contra_transmission, ['contra']*len(excised_contra_transmission)), columns = ['transmission', 'excised_type'])
excised_bi_transmission = pd.DataFrame(zip(excised_bi_transmission, ['bi']*len(excised_bi_transmission)), columns = ['transmission', 'excised_type'])
excised_bi_contra_transmission = pd.DataFrame(zip(excised_bi_contra_transmission, ['bi_contra']*len(excised_bi_contra_transmission)), columns = ['transmission', 'excised_type'])
control_transmission = pd.DataFrame(zip(control_transmission, ['control']*len(control_transmission)), columns = ['transmission', 'excised_type'])
control_bi_transmission = pd.DataFrame(zip(control_bi_transmission, ['control-bi']*len(control_bi_transmission)), columns = ['transmission', 'excised_type'])
control_contra_transmission = pd.DataFrame(zip(control_contra_transmission, ['control-contra']*len(excised_bi_contra_transmission)), columns = ['transmission', 'excised_type'])
control_bi_contra_transmission = pd.DataFrame(zip(control_bi_contra_transmission, ['control-bi-contra']*len(excised_bi_contra_transmission)), columns = ['transmission', 'excised_type'])

data = pd.concat([control_contra_transmission, excised_contra_transmission, control_bi_transmission, excised_bi_transmission, control_bi_contra_transmission, excised_bi_contra_transmission])

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.boxplot(x='excised_type', y='transmission', data=data, ax=ax, fliersize = 0.1, linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(-0.05, 1.5))
fig.savefig('interhemisphere/plots/transmission_strength_Excised-neurons_expanded.pdf', format='pdf', bbox_inches='tight')

data = pd.concat([control_transmission, excised_contra_transmission, excised_bi_transmission, excised_bi_contra_transmission])

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.boxplot(x='excised_type', y='transmission', data=data, ax=ax, fliersize = 0.1, linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
#sns.barplot(x='excised_type', y='transmission', data=data, ax=ax, linewidth = 0.5, ci='sd')

# %%
# block bilateral neurons ipsi or contra connections

bilateral_br_left = list(np.intersect1d(bilateral_br, left))
bilateral_br_right = list(np.intersect1d(bilateral_br, right))

adj_left = list(np.intersect1d(adj.index, left))
adj_right = list(np.intersect1d(adj.index, right))

excised_bi_ipsi_adj = adj.copy()
excised_bi_ipsi_adj.loc[bilateral_br_left, adj_left] = 0
excised_bi_ipsi_adj.loc[bilateral_br_right, adj_right] = 0

excised_bi_contra_adj = adj.copy()
excised_bi_contra_adj.loc[bilateral_br_left, adj_right] = 0
excised_bi_contra_adj.loc[bilateral_br_right, adj_left] = 0

bi_dIpsi_all_inputs_left, bi_dIpsi_all_inputs_right, bi_dIpsi_hit_list_left, bi_dIpsi_hit_list_right, _ = excise_cascade([], input_skids, output_skids, brain_inputs_list, left, right, excised_bi_ipsi_adj)
bi_dContra_all_inputs_left, bi_dContra_all_inputs_right, bi_dContra_hit_list_left, bi_dContra_hit_list_right, _ = excise_cascade([], input_skids, output_skids, brain_inputs_list, left, right, excised_bi_contra_adj)
con_bi_dIpsi_all_inputs_left, con_bi_dIpsi_all_inputs_right, con_bi_dIpsi_hit_list_left, con_bi_dIpsi_hit_list_right, con_excised_bi_ipsi_adj = excise_cascade(contra_br, input_skids, output_skids, brain_inputs_list, left, right, excised_bi_ipsi_adj)
con_bi_dContra_all_inputs_left, con_bi_dContra_all_inputs_right, con_bi_dContra_hit_list_left, con_bi_dContra_hit_list_right, con_excised_bi_contra_adj = excise_cascade(contra_br, input_skids, output_skids, brain_inputs_list, left, right, excised_bi_contra_adj)

excised_bi_dIpsi_raw_left, excised_bi_dIpsi_raw_right, excised_bi_dIpsi_data_left, excised_bi_dIpsi_data_right = plot_process_excise_cascades(bi_dIpsi_all_inputs_left, bi_dIpsi_all_inputs_right, excised_bi_ipsi_adj, lvl7, hops, 'interhemisphere/plots/excised-ipsi-edges_Bilateral_left-right-visits_clusters')
excised_bi_dContra_raw_left, excised_bi_dContra_raw_right, excised_bi_dContra_data_left, excised_bi_dContra_data_right = plot_process_excise_cascades(bi_dContra_all_inputs_left, bi_dContra_all_inputs_right, excised_bi_contra_adj, lvl7, hops, 'interhemisphere/plots/excised-contra-edges_Bilateral_left-right-visits_clusters')
excised_con_bi_dIpsi_raw_left, excised_con_bi_dIpsi_raw_right, excised_con_bi_dIpsi_data_left, excised_con_bi_dIpsi_data_right = plot_process_excise_cascades(con_bi_dIpsi_all_inputs_left, con_bi_dIpsi_all_inputs_right, con_excised_bi_ipsi_adj, lvl7, hops, 'interhemisphere/plots/excised-Contra_ipsi-edges_Bilateral_left-right-visits_clusters')
excised_con_bi_dContra_raw_left, excised_con_bi_dContra_raw_right, excised_con_bi_dContra_data_left, excised_con_bi_dContra_data_right = plot_process_excise_cascades(con_bi_dContra_all_inputs_left, con_bi_dContra_all_inputs_right, con_excised_bi_contra_adj, lvl7, hops, 'interhemisphere/plots/excised-Contra_contra-edges_Bilateral_left-right-visits_clusters')

# additional plots

diff_bi_dIpsi = abs(excised_bi_dIpsi_data_left - excised_bi_dIpsi_data_right)
diff_bi_dContra = abs(excised_bi_dContra_data_left - excised_bi_dContra_data_right)
diff_con_bi_dIpsi = abs(excised_con_bi_dIpsi_data_left - excised_con_bi_dIpsi_data_right)
diff_con_bi_dContra = abs(excised_con_bi_dContra_data_left - excised_con_bi_dContra_data_right)

diff_bi_dIpsi = pd.DataFrame(zip(diff_bi_dIpsi, ['bi-dIpsi']*len(diff_bi_dIpsi)), columns = ['hemisphere_segregation', 'excised_type'])
diff_bi_dContra = pd.DataFrame(zip(diff_bi_dContra, ['bi-dContra']*len(diff_bi_dContra)), columns = ['hemisphere_segregation', 'excised_type'])
diff_con_bi_dIpsi = pd.DataFrame(zip(diff_con_bi_dIpsi, ['con + bi-dIpsi']*len(diff_con_bi_dIpsi)), columns = ['hemisphere_segregation', 'excised_type'])
diff_con_bi_dContra = pd.DataFrame(zip(diff_con_bi_dContra, ['con + bi-dContra']*len(diff_con_bi_dContra)), columns = ['hemisphere_segregation', 'excised_type'])

data = pd.concat([diff_control, diff_bi_dIpsi, diff_bi_dContra, 
                diff_contra, diff_con_bi_dIpsi, diff_con_bi_dContra, diff_bi])
fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.boxplot(x='excised_type', y='hemisphere_segregation', data=data, ax=ax, fliersize = 0.1, linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
fig.savefig('interhemisphere/plots/hemisphere_segregation_excised-ipsi-contra-edges_Bilateral.pdf', format='pdf', bbox_inches='tight')


excised_bi_dIpsi_transmission = list(excised_bi_dIpsi_raw_right/control_raw_right) + list(excised_bi_dIpsi_raw_left/control_raw_left)
excised_bi_dContra_transmission = list(excised_bi_dContra_raw_right/control_raw_right) + list(excised_bi_dContra_raw_left/control_raw_left)
excised_con_bi_dIpsi_transmission = list(excised_con_bi_dIpsi_raw_right/control_raw_right) + list(excised_con_bi_dIpsi_raw_left/control_raw_left)
excised_con_bi_dContra_transmission = list(excised_con_bi_dContra_raw_right/control_raw_right) + list(excised_con_bi_dContra_raw_left/control_raw_left)
excised_bi_dIpsi_transmission = pd.DataFrame(zip(excised_bi_dIpsi_transmission, ['bi-dIpsi']*len(excised_bi_dIpsi_transmission)), columns = ['transmission', 'excised_type'])
excised_bi_dContra_transmission = pd.DataFrame(zip(excised_bi_dContra_transmission, ['bi-dContra']*len(excised_bi_dContra_transmission)), columns = ['transmission', 'excised_type'])
excised_con_bi_dIpsi_transmission = pd.DataFrame(zip(excised_con_bi_dIpsi_transmission, ['con + bi-dIpsi']*len(excised_con_bi_dIpsi_transmission)), columns = ['transmission', 'excised_type'])
excised_con_bi_dContra_transmission = pd.DataFrame(zip(excised_con_bi_dContra_transmission, ['con + bi-dContra']*len(excised_con_bi_dContra_transmission)), columns = ['transmission', 'excised_type'])

data = pd.concat([control_transmission, excised_bi_dIpsi_transmission, excised_bi_dContra_transmission, 
                    excised_contra_transmission, excised_con_bi_dIpsi_transmission, excised_con_bi_dContra_transmission, excised_bi_transmission])

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.boxplot(x='excised_type', y='transmission', data=data, ax=ax, fliersize = 0.1, linewidth = 0.5)
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(-0.05, 1.5))
fig.savefig('interhemisphere/plots/transmission_strength_Excised-neurons_expanded.pdf', format='pdf', bbox_inches='tight')


## *** CONTINUE HERE

# %%
# signal to left and right descending
# not super obvious differences when all descending are lumped together
# signal clearly goes to both sides of the brain equally

#sns.heatmap([ORN_hit_hist_left[dVNC_indices_left].sum(axis = 0), ORN_hit_hist_left[dVNC_indices_right].sum(axis = 0)])
#sns.heatmap([(ORN_hit_hist_left[dVNC_indices_left]>50).sum(axis = 0), (ORN_hit_hist_left[dVNC_indices_right]>50).sum(axis = 0)])
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
#output_dVNC = pd.DataFrame([ORN_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            ORN_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            ORN_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            AN_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            AN_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            AN_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            MN_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            MN_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            MN_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            A00c_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            A00c_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            thermo_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            thermo_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            photo_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            photo_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            photo_hit_hist_right[dVNC_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> dVNC_left', 'ORN_left -> dVNC_right', 
                                    'ORN_right -> dVNC_left', 'ORN_right -> dVNC_right',
                                    'AN_left -> dVNC_left', 'AN_left -> dVNC_right', 
                                    'AN_right -> dVNC_left', 'AN_right -> dVNC_right',
                                    'MN_left -> dVNC_left', 'MN_left -> dVNC_right', 
                                    'MN_right -> dVNC_left', 'MN_right -> dVNC_right',
                                    'A00c_left -> dVNC_left', 'A00c_left -> dVNC_right', 
                                    'A00c_right -> dVNC_left', 'A00c_right -> dVNC_right',
                                    'thermo_left -> dVNC_left', 'thermo_left -> dVNC_right', 
                                    'thermo_right -> dVNC_left', 'thermo_right -> dVNC_right',
                                    'photo_left -> dVNC_left', 'photo_left -> dVNC_right', 
                                    'photo_right -> dVNC_left', 'photo_right -> dVNC_right'])

sns.heatmap(output_dVNC, ax = axs, rasterized = True)

# %%
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
output_dSEZ = pd.DataFrame([ORN_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            ORN_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            ORN_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            AN_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            AN_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            AN_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            MN_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            MN_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            MN_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            A00c_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            A00c_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            thermo_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            thermo_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            photo_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            photo_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            photo_hit_hist_right[dSEZ_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> dSEZ_left', 'ORN_left -> dSEZ_right', 
                                    'ORN_right -> dSEZ_left', 'ORN_right -> dSEZ_right',
                                    'AN_left -> dSEZ_left', 'AN_left -> dSEZ_right', 
                                    'AN_right -> dSEZ_left', 'AN_right -> dSEZ_right',
                                    'MN_left -> dSEZ_left', 'MN_left -> dSEZ_right', 
                                    'MN_right -> dSEZ_left', 'MN_right -> dSEZ_right',
                                    'A00c_left -> dSEZ_left', 'A00c_left -> dSEZ_right', 
                                    'A00c_right -> dSEZ_left', 'A00c_right -> dSEZ_right',
                                    'thermo_left -> dSEZ_left', 'thermo_left -> dSEZ_right', 
                                    'thermo_right -> dSEZ_left', 'thermo_right -> dSEZ_right',
                                    'photo_left -> dSEZ_left', 'photo_left -> dSEZ_right', 
                                    'photo_right -> dSEZ_left', 'photo_right -> dSEZ_right'])

sns.heatmap(output_dSEZ, ax = axs, rasterized = True)

# %%
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
output_RG = pd.DataFrame([ORN_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[RG_indices_right].sum(axis = 0),
                            ORN_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            ORN_hit_hist_right[RG_indices_right].sum(axis = 0),
                            AN_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[RG_indices_right].sum(axis = 0),
                            AN_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            AN_hit_hist_right[RG_indices_right].sum(axis = 0),
                            MN_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[RG_indices_right].sum(axis = 0),
                            MN_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            MN_hit_hist_right[RG_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[RG_indices_right].sum(axis = 0),
                            A00c_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            A00c_hit_hist_right[RG_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[RG_indices_right].sum(axis = 0),
                            thermo_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            thermo_hit_hist_right[RG_indices_right].sum(axis = 0),
                            photo_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[RG_indices_right].sum(axis = 0),
                            photo_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            photo_hit_hist_right[RG_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> RG_left', 'ORN_left -> RG_right', 
                                    'ORN_right -> RG_left', 'ORN_right -> RG_right',
                                    'AN_left -> RG_left', 'AN_left -> RG_right', 
                                    'AN_right -> RG_left', 'AN_right -> RG_right',
                                    'MN_left -> RG_left', 'MN_left -> RG_right', 
                                    'MN_right -> RG_left', 'MN_right -> RG_right',
                                    'A00c_left -> RG_left', 'A00c_left -> RG_right', 
                                    'A00c_right -> RG_left', 'A00c_right -> RG_right',
                                    'thermo_left -> RG_left', 'thermo_left -> RG_right', 
                                    'thermo_right -> RG_left', 'thermo_right -> RG_right',
                                    'photo_left -> RG_left', 'photo_left -> RG_right', 
                                    'photo_right -> RG_left', 'photo_right -> RG_right'])

sns.heatmap(output_RG, ax = axs, rasterized = True)

# %%
#
x_axis_labels = [0,1,2,3,4,5,6]
x_label = 'Hops from Sensory'

fig, axs = plt.subplots(
    4, 2, figsize=(5, 8)
)

fig.tight_layout(pad=2.5)
#cbar_ax = axs.add_axes([3, 7, .1, .75])

ax = axs[0, 0] 
ax.set_title('Signal from ORN')
ax.set(xticks=[0, 1, 2, 3, 4, 5])
sns.heatmap(ORN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[1, 0]
ax.set_title('Signal from AN')
sns.heatmap(AN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[2, 0]
ax.set_title('Signal from MN')
sns.heatmap(MN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[3, 0]
ax.set_title('Signal from A00c')
sns.heatmap(A00c_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)
ax.set_xlabel(x_label)

ax = axs[0, 1]
ax.set_title('Signal from vtd')
sns.heatmap(vtd_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[1, 1]
ax.set_title('Signal from thermo')
sns.heatmap(thermo_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[2, 1]
ax.set_title('Signal from photo')
sns.heatmap(photo_profile.T.iloc[:,0:7], ax = ax, cbar_ax = axs[3, 1], xticklabels = x_axis_labels, rasterized=True)
ax.set_xlabel(x_label)

ax = axs[3, 1]
ax.set_xlabel('Number of Visits\nfrom Sensory Signal')
#ax.axis("off")

plt.savefig('cascades/plots/sensory_integration_per_hop.pdf', format='pdf', bbox_inches='tight')

# %%
# visits in ipsi or contra neurons per hop
# initial exploratory chunk
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
ipsi_contra_flow = pd.DataFrame([ORN_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[contra_indices_left].sum(axis = 0),
                            ORN_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            ORN_hit_hist_left[contra_indices_right].sum(axis = 0),
                            AN_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[contra_indices_left].sum(axis = 0),
                            AN_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            AN_hit_hist_left[contra_indices_right].sum(axis = 0),
                            MN_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[contra_indices_left].sum(axis = 0),
                            MN_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            MN_hit_hist_left[contra_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[contra_indices_left].sum(axis = 0),
                            A00c_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            A00c_hit_hist_left[contra_indices_right].sum(axis = 0),
                            vtd_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            vtd_hit_hist_left[contra_indices_left].sum(axis = 0),
                            vtd_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            vtd_hit_hist_left[contra_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[contra_indices_left].sum(axis = 0),
                            thermo_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            thermo_hit_hist_left[contra_indices_right].sum(axis = 0),
                            photo_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[contra_indices_left].sum(axis = 0),
                            photo_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            photo_hit_hist_left[contra_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> ipsi_indices_left', 'ORN_left -> contra_indices_left', 
                                    'ORN_left -> ipsi_indices_right', 'ORN_left -> contra_indices_right',
                                    'AN_left -> ipsi_indices_left', 'AN_left -> contra_indices_left', 
                                    'AN_left -> ipsi_indices_right', 'AN_left -> contra_indices_right',
                                    'MN_left -> ipsi_indices_left', 'MN_left -> contra_indices_left', 
                                    'MN_left -> ipsi_indices_right', 'MN_left -> contra_indices_right',
                                    'A00c_left -> ipsi_indices_left', 'A00c_left -> contra_indices_left', 
                                    'A00c_left -> ipsi_indices_right', 'A00c_left -> contra_indices_right',
                                    'vtd_left -> ipsi_indices_left', 'vtd_left -> contra_indices_left', 
                                    'vtd_left -> ipsi_indices_right', 'vtd_left -> contra_indices_right',
                                    'thermo_left -> ipsi_indices_left', 'thermo_left -> contra_indices_left', 
                                    'thermo_left -> ipsi_indices_right', 'thermo_left -> contra_indices_right',
                                    'photo_left -> ipsi_indices_left', 'photo_left -> contra_indices_left', 
                                    'photo_left -> ipsi_indices_right', 'photo_left -> contra_indices_right'])

sns.heatmap(ipsi_contra_flow, ax = axs, rasterized = True)

# %%
# number of ipsi or contra neurons visited >50 per hop
# initial exploratory chunk
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
threshold = n_init/2

ipsi_contra_flow_num_neurons = pd.DataFrame([(ORN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (ORN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (AN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (AN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (MN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (MN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (A00c_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (A00c_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (vtd_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (vtd_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (vtd_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (vtd_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (thermo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (thermo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (photo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (photo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0)], 
                            index = ['ORN_left -> ipsi_indices_left', 'ORN_left -> contra_indices_left', 
                                    'ORN_left -> ipsi_indices_right', 'ORN_left -> contra_indices_right',
                                    'AN_left -> ipsi_indices_left', 'AN_left -> contra_indices_left', 
                                    'AN_left -> ipsi_indices_right', 'AN_left -> contra_indices_right',
                                    'MN_left -> ipsi_indices_left', 'MN_left -> contra_indices_left', 
                                    'MN_left -> ipsi_indices_right', 'MN_left -> contra_indices_right',
                                    'A00c_left -> ipsi_indices_left', 'A00c_left -> contra_indices_left', 
                                    'A00c_left -> ipsi_indices_right', 'A00c_left -> contra_indices_right',
                                    'vtd_left -> ipsi_indices_left', 'vtd_left -> contra_indices_left', 
                                    'vtd_left -> ipsi_indices_right', 'vtd_left -> contra_indices_right',
                                    'thermo_left -> ipsi_indices_left', 'thermo_left -> contra_indices_left', 
                                    'thermo_left -> ipsi_indices_right', 'thermo_left -> contra_indices_right',
                                    'photo_left -> ipsi_indices_left', 'photo_left -> contra_indices_left', 
                                    'photo_left -> ipsi_indices_right', 'photo_left -> contra_indices_right'])

sns.heatmap(ipsi_contra_flow_num_neurons, ax = axs, rasterized = True)

