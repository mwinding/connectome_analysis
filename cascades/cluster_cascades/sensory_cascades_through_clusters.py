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

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)
order = pd.read_csv('cascades/data/signal_flow_order_lvl7.csv').values

# make array from list of lists
order_delisted = []
for sublist in order:
    order_delisted.append(sublist[0])

order = np.array(order_delisted)

#%%
# pull sensory annotations and then pull associated skids
input_names = pymaid.get_annotated('mw brain inputs').name
input_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

# order names and skids in desired way for the rest of analysis
sensory_order = [0, 3, 4, 1, 2, 6, 5]
input_names_format = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c']
input_names_format_reordered = [input_names_format[i] for i in sensory_order]
input_skids_list_reordered = [input_skids_list[i] for i in sensory_order]

#%%
# cascades from each sensory modality
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert skids to indices
input_indices_list = []
for input_skids in input_skids_list_reordered:
    indices = np.where([x in input_skids for x in mg.meta.index])[0]
    input_indices_list.append(indices)

output_indices_list = []
for input_skids in output_skids_list:
    indices = np.where([x in input_skids for x in mg.meta.index])[0]
    output_indices_list.append(indices)

output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = output_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

input_hit_hist_list = []
for input_indices in input_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = input_indices)
    input_hit_hist_list.append(hit_hist)


#%%
# grouping cascade indices by cluster type

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

# cluster order and number of neurons per cluster
cluster_lvl7 = []
for key in lvl7.groups.keys():
    cluster_lvl7.append([key, len(lvl7.groups[key])])

cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'num_cluster'])

# breaking signal cascades into cluster groups
input_hit_hist_lvl7 = []
for hit_hist in input_hit_hist_list:
    sensory_clustered_hist = []

    for key in lvl7.groups.keys():
        skids = lvl7.groups[key]
        indices = np.where([x in skids for x in mg.meta.index])[0]
        cluster_hist = hit_hist[indices]

        sensory_clustered_hist.append(cluster_hist)
    
    input_hit_hist_lvl7.append(sensory_clustered_hist)

# summed signal cascades per cluster group (hops remain intact)
summed_hist_lvl7 = []
for input_hit_hist in input_hit_hist_lvl7:
    sensory_sum_hist = []
    for i, cluster in enumerate(input_hit_hist):
        sum_cluster = cluster.sum(axis = 0)/(cluster_lvl7.iloc[i].num_cluster) # normalize by number of neurons in cluster
        sensory_sum_hist.append(sum_cluster)

    sensory_sum_hist = pd.DataFrame(sensory_sum_hist) # column names will be hop number
    sensory_sum_hist.index = cluster_lvl7.key # uses cluster name for index of each summed cluster row
    summed_hist_lvl7.append(sensory_sum_hist)

# %%
# plotting signal flow through clusters

fig, axs = plt.subplots(
    4, 2, figsize=(10, 10)
)

fig.tight_layout(pad=2.0)

threshold = 25
vmax = n_init*.8

for i in range(0, len(input_names_format_reordered)):
    if(i<4):
        ax = axs[i, 0]
    if(i>=4):
        ax = axs[i-4,1]
    
    sns.heatmap(summed_hist_lvl7[i].loc[order], ax = ax, rasterized=True, vmax = vmax)
    ax.set_ylabel('Individual Clusters')
    ax.set_yticks([])

    ax.set_xlabel('Hops from %s signal' %input_names_format_reordered[i])

    #sns.heatmap(summed_hist_lvl7[1].loc[sort], ax = ax, rasterized=True)

ax = axs[3, 1]
ax.axis("off")
caption = f"Figure: Hop histogram of individual level 7 clusters\nCascades starting at each sensory modality\nending at brain output neurons"
ax.text(0, 1, caption, va="top")

plt.savefig('cascades/cluster_plots/sensory_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

'''
# sort based on visit numbers per hop
sort_template = summed_hist_lvl7[0]
sort_template = sort_template[sort_template>threshold]
sort_template.fillna(0)
sort = sort_template.sort_values(by = list(sort_template.columns), ascending = False).index
'''

# %%
# mutlisensory nature of each cluster

collapsed_hops_lvl7_list = []
for hist in summed_hist_lvl7:
    collapsed_hops_lvl7_list.append(hist.sum(axis = 1))

collapsed_hops_lvl7 = pd.DataFrame(collapsed_hops_lvl7_list, index = input_names_format_reordered).T

fg = sns.clustermap(collapsed_hops_lvl7.loc[order], col_cluster = False, yticklabels=False, rasterized = True)
ax = fg.ax_heatmap
ax.set_ylabel('Individual Clusters')
fg.savefig('cascades/cluster_plots/multimodal_nature_of_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

# checking where inputs, outputs, etc are located in these reordered clusters
fg.dendrogram_row.reordered_ind
dendrogram_order = order[fg.dendrogram_row.reordered_ind]
lvl7 = list(clusters.groupby('lvl7_labels'))

cluster_membership = []
for cluster_key in dendrogram_order:
    cluster_temp = clusters[clusters.lvl7_labels==cluster_key]
    cluster_dSEZ = sum(cluster_temp.dSEZ == True)
    cluster_dVNC = sum(cluster_temp.dVNC == True)
    cluster_RG = sum(cluster_temp.RG == True)
    cluster_input = sum(cluster_temp.input == True)
    cluster_output = sum(cluster_temp.output == True)
    cluster_brain = sum(cluster_temp.brain_neurons == True)
    cluster_all = len(cluster_temp.index)

    cluster_membership.append(dict({'cluster_key': cluster_key,
                                'total_neurons': cluster_all, 'brain_neurons': cluster_brain/cluster_all, 
                                'outputs': cluster_output/cluster_all, 'inputs': cluster_input/cluster_all,
                                'dVNC': cluster_dVNC/cluster_all, 'dSEZ': cluster_dSEZ/cluster_all, 
                                'RG': cluster_RG/cluster_all}))

cluster_membership = pd.DataFrame(cluster_membership)
sns.heatmap(cluster_membership.iloc[:, 3:8])

# %%
# identifying locations of intersecting signals in individual neurons
# which clusters and which neurons?


