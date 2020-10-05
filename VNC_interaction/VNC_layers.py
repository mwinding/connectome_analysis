#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rm = pymaid.CatmaidInstance(url, name, password, token)
adj = pd.read_csv('VNC_interaction/data/axon-dendrite.csv', header = 0, index_col = 0)
inputs = pd.read_csv('VNC_interaction/data/input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-09-22.csv', header = 0) # import pairs

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

VNC_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')
#test.adj_inter.loc[(slice(None), slice(None), KC), (slice(None), slice(None), MBON)]

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')
A1_proprio = pymaid.get_skids_by_annotation('mw A1 proprio')
A1_somato = pymaid.get_skids_by_annotation('mw A1 somato')

ds_dVNC = VNC_adj.downstream(dVNC, 0.05, exclude = dVNC, by_group=True)
pd.DataFrame(ds_dVNC).to_csv(f'VNC_interaction/data/ds_dVNC_{str(date.today())}.csv', index = False)

source_dVNC, ds_dVNC2 = VNC_adj.downstream(dVNC, 0.05, exclude = dVNC, by_group=False)

# %%
# upstream of MNs

threshold = 0.025

us_A1_MN = VNC_adj.upstream_multihop(A1_MN, threshold)
ds_proprio = VNC_adj.downstream_multihop(A1_proprio, threshold)
ds_somato = VNC_adj.downstream_multihop(A1_somato, threshold)

# connectivity from dVNCs to VNC layers
# at which layer do dVNCs output?

VNC_layers = [us_A1_MN, ds_proprio, ds_somato]

mat = np.zeros(shape = (len(VNC_layers), 5))
for i in range(0,len(VNC_layers)):
    for j in range(0,5):
        summed = VNC_adj.adj.loc[dVNC, VNC_layers[i][j]].sum().sum()
        mat[i,j] = summed

sns.heatmap(mat)

source, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
mat_neurons = np.zeros(shape = (len(VNC_layers), 5))
for i in range(0,len(VNC_layers)):
    for j in range(0,5):
        neurons = len(np.intersect1d(VNC_layers[i][j], ds_dVNC))
        mat_neurons[i, j] = neurons

sns.heatmap(mat_neurons)

 # which dVNCs talk to each layer
 # which VNCs neurons are at each layer
 
# %%
# how many connections between dVNCs and A1 neurons?
# out of date


source_ds = dVNC_A1.loc[(slice(None), source_dVNC.leftid), (slice(None), ds_dVNC.leftid)]

source_dVNC_outputs = (source_ds>threshold).sum(axis=1)
ds_dVNC_inputs = (source_ds>threshold).sum(axis=0)


fig, axs = plt.subplots(
    1, 2, figsize=(5, 3)
)

fig.tight_layout(pad = 2.5)
binwidth = 1
x_range = list(range(0, 5))

ax = axs[0]
data = ds_dVNC_inputs.values
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
ax.hist(data, bins=bins, align='mid')
ax.set_ylabel('A1 pairs')
ax.set_xlabel('Upstream dVNC Pairs')
ax.set_xticks(x_range)
ax.set(xlim = (0.5, 4.5))

ax = axs[1]
data = source_dVNC_outputs.values
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
ax.hist(data[data>0], bins=bins, align='mid')
ax.set_ylabel('dVNC pairs')
ax.set_xlabel('Downstream A1 Pairs')
ax.set_xticks(x_range)
ax.set(xlim = (0.5, 4.5))

plt.savefig('VNC_interaction/plots/connections_dVNC_A1.pdf')

# %%
