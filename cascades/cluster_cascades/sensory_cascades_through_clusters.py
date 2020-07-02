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

#%%
# pull sensory annotations and then pull associated skids
input_names = pymaid.get_annotated('mw brain inputs').name
input_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

#%%
# cascades from each sensory modality
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert skids to indices
input_indices_list = []
for input_skids in input_skids_list:
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
    1, 1, figsize=(10, 10)
)

fig.tight_layout(pad=2.0)

threshold = 25

sort_template = summed_hist_lvl7[0]
sort_template = sort_template[sort_template>threshold]
sort_template.fillna(0)
sort = sort_template.sort_values(by = list(sort_template.columns), ascending = False).index

ax = axs
sns.heatmap(summed_hist_lvl7[0].loc[sort], ax = ax, rasterized=True)
#sns.heatmap(summed_hist_lvl7[1].loc[sort], ax = ax, rasterized=True)

plt.savefig('cascades/cluster_plots/ORN_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

# %%
