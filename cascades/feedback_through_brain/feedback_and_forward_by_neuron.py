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

'''
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
'''
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

# order skids within groups and convert to indices
cluster_lvl7_indices_list = []
sorted_skids = []
for skids in cluster_lvl7:
    skids_median_visit = meta_with_order.loc[skids, 'median_node_visits']
    skids_sorted = skids_median_visit.sort_values().index

    indices = []
    for skid in skids_sorted:
        index = skid_to_index(skid, mg)
        indices.append(index)
    cluster_lvl7_indices_list.append(indices)
    sorted_skids.append(skids_sorted)

# delist
sorted_skids = [val for sublist in sorted_skids for val in sublist]

sorted_indices = []
for skid in sorted_skids:
    sorted_indices.append(skid_to_index(skid, mg))

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

# run cascades parallel from each individual node
from joblib import Parallel, delayed
from tqdm import tqdm

#neuron_indices_list = [val for sublist in cluster_lvl7_indices_list for val in sublist]
def run_cascades_from_node(i, cdispatch):
    return(cdispatch.multistart(start_nodes = i))
    
cluster_hit_hist_list = []
for indices in tqdm(cluster_lvl7_indices_list):
    hit_hist_list = Parallel(n_jobs=-1)(delayed(run_cascades_from_node)(i, cdispatch) for i in indices)
    cluster_hit_hist_list.append(hit_hist_list)

# testing parallelization of cascades
'''
# run cascades parallel from each individual node
from joblib import Parallel, delayed

neuron_indices_list = [val for sublist in cluster_lvl7_indices_list for val in sublist]
def run_cascades_from_node(i, cdispatch):
    return(cdispatch.multistart(start_nodes = index))

hit_hit_list = Parallel(n_jobs=-1)(delayed(run_cascades_from_node)(i, cdispatch) for i in neuron_indices_list[0:20])
hit_hit_list = [run_cascades_from_node(i, cdispatch) for i in neuron_indices_list[0:20]]

# timing difference
import time

t0 = time.time()
hit_hit_list = [run_cascades_from_node(i, cdispatch) for i in neuron_indices_list[0:20]]
t1 = time.time()

total_original = t1-t0

t0 = time.time()
hit_hit_list = Parallel(n_jobs=-1)(delayed(run_cascades_from_node)(i, cdispatch) for i in neuron_indices_list[0:20])
t1 = time.time()

total_parallel = t1-t0
print(total_original)
print(total_parallel)
print(total_original/total_parallel)
'''

'''
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
'''

# %%
# plot in feedback/feedforward matrix

# delist
neuron_hit_hist_list = [val for sublist in cluster_hit_hist_list for val in sublist]

# sort matrices correctly and sum
neuron_hit_hist_hop_summed = []
for hit_hist in neuron_hit_hist_list:
    hop_summed = hit_hist[sorted_indices, 0:4].sum(axis = 1)
    neuron_hit_hist_hop_summed.append(hop_summed)

neuron_hit_hist_hop_summed = pd.DataFrame(neuron_hit_hist_hop_summed).T

import cmasher as cmr

plt.imshow(neuron_hit_hist_hop_summed, cmap=cmr.ember, interpolation='none')
plt.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_neurons_4hops_ad.pdf', bbox_inches='tight')

# %%
# compare rw to cascades
from src.traverse import to_markov_matrix, RandomWalk

def run_cascades_from_node(i, cdispatch):
    return(cdispatch.multistart(start_nodes = index))

# cascade
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

test = cdispatch.multistart(start_nodes = [skid_to_index(3827211, mg)])

# cascade, non simultaneous
p = 0.05
max_hops = 10
n_init = 100
simultaneous = False
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

test_sim_false = cdispatch.multistart(start_nodes = [skid_to_index(3827211, mg)])

# randomwalk
max_hops = 10
n_init = 100
simultaneous = False
transition_probs = to_markov_matrix(adj)

cdispatch = TraverseDispatcher(
    RandomWalk,
    transition_probs,
    stop_nodes = [],
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

test_rw = cdispatch.multistart(start_nodes = [skid_to_index(3827211, mg)])

fig, axs = plt.subplots(
    2, 1, figsize=(5, 5)
)
vmax = 100

fig.tight_layout(pad=2.0)

ax = axs[0]
sns.heatmap(test, ax = ax, rasterized = True, vmax = vmax)
ax.set_title('Cascade')

ax = axs[1]
sns.heatmap(test_rw, ax = ax, rasterized = True, vmax = 20)
ax.set_title('Cascade, Simultaneous=False')

#plt.savefig('cascades/feedback_through_brain/plots/test_cascade_vs_rw_84_1.pdf', bbox_inches='tight')

# %%
