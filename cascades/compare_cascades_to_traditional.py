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

rm = pymaid.CatmaidInstance(url, token, name, password)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

#%%
# pull sensory annotations and then pull associated skids
input_names = pymaid.get_annotated('mw brain inputs').name
input_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, output_names))
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
n_init = 1000
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


# %%
# collect skids and names of downstream neurons of sensories
order2_names = pymaid.get_annotated('mw brain inputs 2nd_order').name
order3_names = pymaid.get_annotated('mw brain inputs 3rd_order').name
order4_names = pymaid.get_annotated('mw brain inputs 4th_order').name

order2_neurons = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs 2nd_order').name))
order3_neurons = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs 3rd_order').name))
order4_neurons = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs 4th_order').name))

# %%
# order groups properly
order2_neurons = [order2_neurons[i] for i in order2_names.sort_values().index]
order3_neurons = [order3_neurons[i] for i in order3_names.sort_values().index]
order4_neurons = [order4_neurons[i] for i in order4_names.sort_values().index]

order2_neurons = [order2_neurons[i] for i in order2_names.sort_values().index]
order3_neurons = [order3_neurons[i] for i in order3_names.sort_values().index]
order4_neurons = [order4_neurons[i] for i in order4_names.sort_values().index]

input_hit_hist_list = [input_hit_hist_list[i] for i in input_names.sort_values().index]

# intersection over union for neuron layers
for i, input_hit_hist in enumerate(input_hit_hist_list):
    input_hit_hist = input_hit_hist[1, :]


'''
ORN2_indices = np.where(input_hit_hist_list[0][:,1]>n_init/2)[0]
ORN2_skids = mg.meta.index[ORN2_indices]

ORN3_indices = np.where(input_hit_hist_list[0][:,2]>n_init/2)[0]
ORN3_skids = mg.meta.index[ORN3_indices]

ORN4_indices = np.where(input_hit_hist_list[0][:,3]>n_init/2)[0]
ORN4_skids = mg.meta.index[ORN4_indices]

iou_ORN2 = len(np.intersect1d(ORN2_skids, order2_neurons[2]))/len(np.union1d(ORN2_skids, order2_neurons[2]))
iou_ORN3 = len(np.intersect1d(ORN3_skids, order3_neurons[3]))/len(np.union1d(ORN3_skids, order3_neurons[3]))
iou_ORN4 = len(np.intersect1d(ORN4_skids, order4_neurons[3]))/len(np.union1d(ORN4_skids, order4_neurons[3]))

ORNall_skids_cascade = np.concatenate([ORN2_skids, ORN3_skids, ORN4_skids])
ORNall_skids_ds = np.concatenate([order2_neurons[2], order3_neurons[3], order4_neurons[3]])

iou_allORN = len(np.intersect1d(ORNall_skids_cascade, ORNall_skids_ds))/len(np.union1d(ORNall_skids_cascade, ORNall_skids_ds))
'''
# %%
