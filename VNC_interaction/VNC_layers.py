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

#ds_dVNC = VNC_adj.downstream(dVNC, 0.05, exclude = dVNC, by_group=True)
#pd.DataFrame(ds_dVNC).to_csv(f'VNC_interaction/data/ds_dVNC_{str(date.today())}.csv', index = False)

#source_dVNC, ds_dVNC2 = VNC_adj.downstream(dVNC, 0.05, exclude = dVNC, by_group=False)

# %%
from connectome_tools.cascade_analysis import Celltype_Analyzer, Celltype

# VNC layering with respect to sensories or motorneurons
threshold = 0.025

us_A1_MN = VNC_adj.upstream_multihop(A1_MN, threshold)
ds_proprio = VNC_adj.downstream_multihop(A1_proprio, threshold)
ds_somato = VNC_adj.downstream_multihop(A1_somato, threshold)

# how many neurons are included in layering?
VNC_layers = [us_A1_MN, ds_proprio, ds_somato]
all_included = [x for sublist in VNC_layers for subsublist in sublist for x in subsublist]

frac_included = len(np.intersect1d(A1, all_included))/len(A1)
print(f'Fraction VNC cells covered = {frac_included}')

# how similar are layers
celltypes_us_MN = [Celltype(f'us-MN-{i}', layer) for i, layer in enumerate(us_A1_MN)]
celltypes_ds_proprio = [Celltype(f'ds-Proprio-{i}', layer) for i, layer in enumerate(ds_proprio)]
celltypes_ds_somato = [Celltype(f'ds-Somato-{i}', layer) for i, layer in enumerate(ds_somato)]

celltypes = celltypes_us_MN + celltypes_ds_proprio + celltypes_ds_somato

VNC_analyzer = Celltype_Analyzer(celltypes)
sns.heatmap(VNC_analyzer.compare_membership(), square = True)
plt.savefig('VNC_interaction/plots/similarity_between_VNC_layers.pdf', bbox_inches='tight')

# %%
# upset plot of VNC types (MN, Proprio, Somato)

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

VNC_types = [[x for layer in us_A1_MN for x in layer], 
                [x for layer in ds_proprio for x in layer], 
                [x for layer in ds_somato for x in layer]]

data = [x for cell_type in VNC_types for x in cell_type]
data = np.unique(data)

cats = []
for skid in data:
    cat = []
    if(skid in VNC_types[0]):
        cat = cat + ['MN']

    if(skid in VNC_types[1]):
        cat = cat + ['Proprio']

    if(skid in VNC_types[2]):
        cat = cat + ['Somato']

    cats.append(cat)

VNC_types_df = from_memberships(cats, data = data)

counts = []
for celltype in np.unique(cats):
    count = 0
    for cat in cats:
        if(celltype == cat):
            count += 1

    counts.append(count)

upset = from_memberships(np.unique(cats), data = counts)
plot(upset, sort_categories_by = None)
plt.savefig('VNC_interaction/plots/VNC_signal_type.pdf', bbox_inches='tight')

# %%
# upset plot of VNC types including layers (MN-0, -1, -2, Proprio-0, -1, 2, Somat-0, -1, -2, etc.)

VNC_type_layers = us_A1_MN + ds_proprio + ds_somato
VNC_type_layer_names = [f'us_MN-{i+1}' for i,x in enumerate(us_A1_MN)] + [f'ds_Proprio-{i+1}' for i,x in enumerate(ds_proprio)] + [f'ds_Somato-{i+1}' for i,x in enumerate(ds_somato)]
data = [x for cell_type in VNC_type_layers for x in cell_type]
data = np.unique(data)

cats = []
for skid in data:
    cat = []
    for i, layer in enumerate(VNC_type_layers):
        if(skid in layer):
            cat = cat + [VNC_type_layer_names[i]]

    cats.append(cat)

VNC_type_layers_df = from_memberships(cats, data = data)

counts = []
for celltype in np.unique(cats):
    count = 0
    for cat in cats:
        if(celltype == cat):
            count += 1

    counts.append(count)

upset = from_memberships(np.unique(cats), data = counts)
plot(upset, sort_categories_by = None)
plt.savefig('VNC_interaction/plots/VNC_layer_signal_type.pdf', bbox_inches='tight')

# %%
# where are ascendings in layering?

A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
celltype_A1_ascending = Celltype('A1 ascending', A1_ascending)

celltypes_us_MN = [Celltype(f'us-MN-{i}', layer) for i, layer in enumerate([A1_MN] + us_A1_MN)]
celltypes_ds_proprio = [Celltype(f'ds-Proprio-{i}', layer) for i, layer in enumerate([A1_proprio] + ds_proprio)]
celltypes_ds_somato = [Celltype(f'ds-Somato-{i}', layer) for i, layer in enumerate([A1_somato] + ds_somato)]

celltypes = celltypes_us_MN + celltypes_ds_proprio + celltypes_ds_somato

VNC_analyzer = Celltype_Analyzer(celltypes)
VNC_analyzer.set_known_types([celltype_A1_ascending], unknown=False)
ascendings_data = VNC_analyzer.memberships(by_celltype=False).T
ascendings_split_data = pd.DataFrame([[x for sublist in ascendings_data.iloc[0:5].values for x in sublist] + [0.0] + [0.0], #add 0.0 to make square matrix
                                    [x for sublist in ascendings_data.iloc[5:10].values for x in sublist] + [0.0] + [0.0],  #add 0.0 to make square matrix
                                    [x for sublist in ascendings_data.iloc[10:17].values for x in sublist]], 
                                    index = ['MN', 'Proprio', 'Somato'],
                                    columns = [f'Layer {i}' for i in range(0, 7)])

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(ascendings_split_data.T*len(A1_ascending), annot=True, cmap = 'Blues', cbar = False, ax = axs)
ax.set_title('Ascending Neurons')
plt.savefig('VNC_interaction/plots/ascending_neuron_layers.pdf', bbox_inches='tight')

# %%
# which dVNCs talk to each layer
# which ds-dVNCs neurons are at each layer

VNC_layers = [[A1_MN] + us_A1_MN, [A1_proprio] + ds_proprio, [A1_somato] + ds_somato]

# number of neurons downstream of dVNC at each VNC layer
source_dVNC, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
mat_neurons = np.zeros(shape = (len(VNC_layers), 7))
for i in range(0,len(VNC_layers)):
    for j in range(0,len(VNC_layers[i])):
        neurons = len(np.intersect1d(VNC_layers[i][j], ds_dVNC))
        mat_neurons[i, j] = neurons

mat_neurons = pd.DataFrame(mat_neurons, index = ['MN', 'Proprio', 'Somato'], columns = [f'Layer {i}' for i in range(0,7)])

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(mat_neurons.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('Downstream of dVNCs')
plt.savefig('VNC_interaction/plots/dVNC_downstream_targets.pdf', bbox_inches='tight')

# %%
# export dVNCs and VNCs
source_dVNC, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
pd.DataFrame(ds_dVNC).to_csv(f'VNC_interaction/data/ds_dVNC_{str(date.today())}.csv', index = False)
pd.DataFrame(source_dVNC).to_csv(f'VNC_interaction/data/source_dVNC_{str(date.today())}.csv', index = False)

# %%
# how many connections between dVNCs and A1 neurons?
# out of date


source_ds = VNC_adj.adj_pairwise.loc[(slice(None), source_dVNC), (slice(None), ds_dVNC)]

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
