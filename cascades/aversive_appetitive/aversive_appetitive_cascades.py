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

aversive_skids_list = list(map(pymaid.get_skids_by_annotation, ['mw MBON subclass_aversive', 'mw thermosensories', 'mw photoreceptors', 'mw A00c']))
appetitive_skids_list = list(map(pymaid.get_skids_by_annotation, ['mw MBON subclass_appetitive', 'mw ORN']))

aversive_skids = [val for sublist in aversive_skids_list for val in sublist]
appetitive_skids = [val for sublist in appetitive_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

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
aversive_indices = np.where([x in aversive_skids for x in mg.meta.index])[0]
appetitive_indices = np.where([x in appetitive_skids for x in mg.meta.index])[0]
all_output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

# end cascades at outputs and prevent loops
cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = all_output_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

# run cascades from aversive or appetitive nodes
aversive_hit_hist = cdispatch.multistart(start_nodes = aversive_indices)
appetitive_hit_hist = cdispatch.multistart(start_nodes = appetitive_indices)

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

aversive_hit_hist_lvl7 = hit_hist_to_clusters([aversive_hit_hist], lvl7)
appetitive_hit_hist_lvl7 = hit_hist_to_clusters([appetitive_hit_hist], lvl7)

aversive_summed_hist_lvl7 = sum_cluster_hit_hist(aversive_hit_hist_lvl7)[0]
appetitive_summed_hist_lvl7= sum_cluster_hit_hist(appetitive_hit_hist_lvl7)[0]

# how many neurons are shared between activation patterns?

threshold = 50
hops = 4 # how many hops are considered for threshold

indices_over_thres_appetitive = []
for i, cluster in enumerate(appetitive_hit_hist_lvl7[0]):
    bool_list = (cluster.iloc[:, 1:(hops+1)]).sum(axis = 1)>threshold
    indices = cluster.index[np.where(bool_list)[0]] # identify neuron(s) overthreshold and link to adj matrix indices 
    indices_over_thres_appetitive.append(indices)

indices_over_thres_aversive = []
for i, cluster in enumerate(aversive_hit_hist_lvl7[0]):
    bool_list = (cluster.iloc[:, 1:(hops+1)]).sum(axis = 1)>threshold # summed first 4 hops and checked if over threshold
    indices = cluster.index[np.where(bool_list)[0]] # identify neuron(s) overthreshold and link to adj matrix indices 
    indices_over_thres_aversive.append(indices)

# identify intersection between aversive and appetitive skids
aversive_appetitive_skids = []
num_intersections = []
intersect_clusters = []

for i, key in enumerate(lvl7.groups):
    skids = lvl7.groups[key]
    intersect = np.intersect1d(indices_over_thres_appetitive[i], indices_over_thres_aversive[i])
    aversive_appetitive_skids.append(intersect)

    intersect_clusters.append(len(intersect)/len(skids))
    num_intersections.append(len(intersect))

intersect_clusters = pd.DataFrame(intersect_clusters, index = list(lvl7.groups), columns = ['Both'])
num_intersections = pd.DataFrame(num_intersections, index = list(lvl7.groups))


# identify exclusive aversive or appetitive skids
aversive_skids = []
appetitive_skids = []
aversive_clusters = []
appetitive_clusters = []
num_aversive_clusters = []
num_appetitive_clusters = []

for i, key in enumerate(lvl7.groups):
    skids = lvl7.groups[key]
    aversive = np.setdiff1d(indices_over_thres_aversive[i], indices_over_thres_appetitive[i])
    appetitive = np.setdiff1d(indices_over_thres_appetitive[i], indices_over_thres_aversive[i])
    
    aversive_skids.append(aversive)
    appetitive_skids.append(appetitive)

    num_aversive_clusters.append(len(aversive))
    num_appetitive_clusters.append(len(appetitive))

    aversive_clusters.append(len(aversive)/len(skids))
    appetitive_clusters.append(len(appetitive)/len(skids))


appetitive_clusters = pd.DataFrame(appetitive_clusters, index = list(lvl7.groups), columns = ['Aversive'])
aversive_clusters = pd.DataFrame(aversive_clusters, index = list(lvl7.groups), columns = ['Appetitive'])
num_appetitive_clusters = pd.DataFrame(num_appetitive_clusters, index = list(lvl7.groups))
num_aversive_clusters = pd.DataFrame(num_aversive_clusters, index = list(lvl7.groups))

# %%
# plotting aversive and appetitive cascades; intersections and exclusive signal
import cmasher as cmr

fig, axs = plt.subplots(
    1, 3, figsize=(2, 2)
)

vmax = 100

ax = axs[0]
sns.heatmap(appetitive_summed_hist_lvl7.loc[order, 0:4], vmax = vmax, ax = ax, cbar = False)
ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_xlabel('Hops')
ax.set_title('Appetitive')

ax = axs[1]
sns.heatmap(pd.concat([appetitive_clusters.loc[order], 
                                    intersect_clusters.loc[order], 
                                    aversive_clusters.loc[order]], axis = 1), ax = ax, cbar = False, cmap = cmr.lavender)
ax.set_yticks([])

ax = axs[2]
sns.heatmap(aversive_summed_hist_lvl7.loc[order, 0:4], vmax = vmax, ax = ax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set_xlim(5, 0)
ax.set_xlabel('Hops')
ax.set_title('Aversive')

plt.savefig('cascades/aversive_appetitive/plots/aversive_appetitive_cascades.pdf', format='pdf', bbox_inches='tight')

# %%
# known cell types in each valence type

# which cell types are in a set of skids?

annot_list_types = ['sensory', 'PN', 'LHN', 'MBIN', 'KC', 'MBON', 'FBN', 'CN', 'dVNC', 'dSEZ', 'RGN']
annot_list = [list(pymaid.get_annotated('mw brain inputs').name), 
            list(pymaid.get_annotated('mw brain inputs 2nd_order PN').name),
            ['mw LHN'], ['mw MBIN'], ['mw KC'], ['mw MBON'],
            ['mw FBN', 'mw FB2N', 'mw FAN'],
            ['mw CN']
            ]

inputs_skids = pymaid.get_annotated('mw brain inputs', include_sub_annotations = True)
inputs_skids = inputs_skids[inputs_skids.type == 'neuron'].skeleton_ids
inputs_skids = [val for sublist in list(inputs_skids) for val in sublist]

PN_skids = pymaid.get_annotated('mw brain inputs 2nd_order PN', include_sub_annotations = True)
PN_skids = PN_skids[PN_skids.type == 'neuron'].skeleton_ids
PN_skids = [val for sublist in list(PN_skids) for val in sublist]

LHN_skids = pymaid.get_skids_by_annotation('mw LHN')
MBIN_skids = pymaid.get_skids_by_annotation('mw MBIN')
KC_skids = pymaid.get_skids_by_annotation('mw KC')
MBON_skids = pymaid.get_skids_by_annotation('mw MBON')
FBN_skids = pymaid.get_skids_by_annotation(['mw FBN', 'mw FB2N', 'mw FAN'], allow_partial=True)
CN_skids = pymaid.get_skids_by_annotation('mw CN')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ_skids = pymaid.get_skids_by_annotation('mw dSEZ')
RGN_skids = pymaid.get_skids_by_annotation('mw RG')

skid_list = [inputs_skids, PN_skids, LHN_skids, MBIN_skids, KC_skids, MBON_skids, 
            FBN_skids, CN_skids, dVNC_skids, dSEZ_skids, RGN_skids]

def member_types(data, skid_list, celltype_names, col_name):

    fraction_type = []
    for skids in skid_list:
        fraction = len(np.intersect1d(data, skids))/len(data)
        fraction_type.append(fraction)

    fraction_type = pd.DataFrame(fraction_type, index = celltype_names, columns = [col_name])
    return(fraction_type)

def index_to_skid(index, mg):
    return(mg.meta.iloc[index, :].name)

# delist skids from cluster structure, convert to skids
app_skids = [index_to_skid(val, mg) for sublist in appetitive_skids for val in sublist] 
inter_skids = [index_to_skid(val, mg) for sublist in aversive_appetitive_skids for val in sublist]
av_skids = [index_to_skid(val, mg) for sublist in aversive_skids for val in sublist]


appetitive_type = member_types(app_skids, skid_list, annot_list_types, 'Appetitive\n(%d)' %len(app_skids))
intersection_type = member_types(inter_skids, skid_list, annot_list_types, 'Both\n(%d)' %len(inter_skids))
aversive_type = member_types(av_skids, skid_list, annot_list_types, 'Aversive\n(%d)' %len(av_skids))

import cmasher as cmr

width = 1.25
height = 1.5
vmax = 0.3
cmap = cmr.lavender
cbar = False
fontsize = 5

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
sns.heatmap(pd.concat([appetitive_type, intersection_type, aversive_type], axis = 1), annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, ax = axs, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/aversive_appetitive/plots/celltypes.pdf', bbox_inches='tight')


# %%
