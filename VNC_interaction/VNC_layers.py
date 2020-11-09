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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

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

# %%
from connectome_tools.cascade_analysis import Celltype_Analyzer, Celltype

# VNC layering with respect to sensories or motorneurons
threshold = 0.01

us_A1_MN = VNC_adj.upstream_multihop(A1_MN, threshold, min_members=0, exclude = A1_MN + A1_proprio + A1_somato)
ds_proprio = VNC_adj.downstream_multihop(A1_proprio, threshold, min_members=0, exclude = A1_MN + A1_proprio + A1_somato)
ds_somato = VNC_adj.downstream_multihop(A1_somato, threshold, min_members=0, exclude = A1_MN + A1_proprio + A1_somato)

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
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_similarity_between_VNC_layers.pdf', bbox_inches='tight')

# %%
# upset plot of VNC types (MN, Proprio, Somato)

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

hops_included = 3

VNC_types = [[x for layer in us_A1_MN[0:hops_included] for x in layer], 
                [x for layer in ds_proprio[0:hops_included] for x in layer], 
                [x for layer in ds_somato[0:hops_included] for x in layer]]

data = [x for cell_type in VNC_types for x in cell_type]
data = np.unique(data)

cats_simple = []
for skid in data:
    cat = []
    if(skid in VNC_types[0]):
        cat = cat + ['MN']

    if(skid in VNC_types[1]):
        cat = cat + ['Proprio']

    if(skid in VNC_types[2]):
        cat = cat + ['Somato']

    cats_simple.append(cat)

VNC_types_df = from_memberships(cats_simple, data = data)

counts = []
for celltype in np.unique(cats_simple):
    count = 0
    for cat in cats_simple:
        if(celltype == cat):
            count += 1

    counts.append(count)

coverage = np.unique([x for sublist in us_A1_MN[0:hops_included] for x in sublist] + [x for sublist in ds_proprio[0:hops_included] for x in sublist] + [x for sublist in ds_somato[0:hops_included] for x in sublist])

upset = from_memberships(np.unique(cats_simple), data = counts)
plot(upset, sort_categories_by = None)
plt.title(f'{len(np.intersect1d(A1, coverage))/len(A1)*100:.2f}% of A1 neurons covered')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC-signal-type_hops-{hops_included}.pdf', bbox_inches='tight')

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
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layer_signal_type.pdf', bbox_inches='tight')

# %%
# number of VNC neurons per layer

MN_counts = [len(layer) for layer in ([A1_MN] + us_A1_MN)]
Proprio_counts = [len(layer) for layer in ([A1_proprio] + ds_proprio)]
Somato_counts = [len(layer) for layer in ([A1_somato] + ds_somato)]
max_length = max([len(MN_counts), len(Proprio_counts), len(Somato_counts)])

if(len(MN_counts)<max_length):
    MN_counts = MN_counts + [0]*(max_length-len(MN_counts))

if(len(Proprio_counts)<max_length):
    Proprio_counts = Proprio_counts + [0]*(max_length-len(Proprio_counts))

if(len(Somato_counts)<max_length):
    Somato_counts = Somato_counts + [0]*(max_length-len(Somato_counts))

VNC_layer_counts = pd.DataFrame()

VNC_layer_counts['MN'] = MN_counts
VNC_layer_counts['Proprio'] = Proprio_counts
VNC_layer_counts['Somato'] = Somato_counts

VNC_layer_counts.index = [f'Layer {i}' for i in range(0,max_length)]

# alternative way of identifying layers
VNC_layers = [[A1_MN] + us_A1_MN, [A1_proprio] + ds_proprio, [A1_somato] + ds_somato]
VNC_type_names = ['MN', 'Proprio', 'Somato']

all_layers, all_layers_skids = VNC_adj.layer_id(VNC_layers, VNC_type_names, A1)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(VNC_layer_counts, annot=True, fmt='d', cmap = 'Greens', cbar = False, ax = axs)
ax.set_title(f'A1 Neurons; {frac_included*100:.0f}% included')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layers.pdf', bbox_inches='tight')

# %%
# where are ascendings in layering?

A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')

VNC_layers = [[A1_MN] + us_A1_MN, [A1_proprio] + ds_proprio, [A1_somato] + ds_somato]
VNC_type_names = ['MN', 'Proprio', 'Somato']

ascendings_layers, ascendings_layers_skids = VNC_adj.layer_id(VNC_layers, VNC_type_names, A1_ascending)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(ascendings_layers.T, annot=True, fmt='.0f', cmap = 'Blues', cbar = False, ax = axs)
ax.set_title('Ascending Neurons')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_ascending_neuron_layers.pdf', bbox_inches='tight')

# %%
# number of neurons downstream of dVNC at each VNC layer
source_dVNC, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
edges, ds_dVNC = VNC_adj.edge_threshold(source_dVNC, ds_dVNC, threshold, direction='downstream')

ds_dVNC_layers, ds_dVNC_layers_skids = VNC_adj.layer_id(VNC_layers, ['MN', 'Proprio', 'Somato'], ds_dVNC)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(ds_dVNC_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('Downstream of dVNCs')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_dVNC_downstream_targets.pdf', bbox_inches='tight')

# %%
# plot A1 structure together

plt.rcParams['font.size'] = 6

fig, axs = plt.subplots(
    1, 3, figsize = (2.5, 1.5)
)
ax = axs[0]
sns.heatmap(VNC_layer_counts, cbar_kws={'label': 'Number of Neurons'}, annot = True, fmt='.0f', cmap = 'Greens', cbar = False, ax = ax)
ax.set_title('A1 Neurons')

ax = axs[1]
sns.heatmap(ascendings_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Blues', cbar = False, ax = ax)
ax.set_title('Ascendings')
ax.set_yticks([])

ax = axs[2]
sns.heatmap(ds_dVNC_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('ds-dVNCs')
ax.set_yticks([])

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_A1_structure.pdf', bbox_inches='tight')

# %%
# supplementary plot with exclusive MN, proprio, and Somato types

skids_MN_only = list(VNC_types_df.loc[(True, False, False)]) + A1_MN
skids_Proprio_only = list(VNC_types_df.loc[(False, True, False)]) + A1_proprio
skids_Somato_only = list(VNC_types_df.loc[(False, False, True)]) + A1_somato
skids_MN_Proprio = list(VNC_types_df.loc[(True, True, False)])
skids_MN_Somato = list(VNC_types_df.loc[(True, False, True)])
skids_Proprio_Somato = list(VNC_types_df.loc[(False, True, True)])
skids_MN_Proprio_Somato = list(VNC_types_df.loc[(True, True, True)])

upset_names = ['MN only', 'Proprio only', 'Somato only', 'MN + Proprio', 'MN + Somato', 'Proprio + Somato', 'MN + Proprio + Somato']
upset_skids = [skids_MN_only, skids_Proprio_only, skids_Somato_only, 
                skids_MN_Proprio, skids_MN_Somato, skids_Proprio_Somato, skids_MN_Proprio_Somato]

upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(all_layers_skids.T.values, ['MN', 'Proprio', 'Somato'], skids)
    #count_layers.columns = [f'Layer {x}' for x in range(0, len(count_layers.columns))]
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

MN_df = pd.DataFrame()
MN_df['MN'] = all_layers.T.MN

Proprio_df = pd.DataFrame()
Proprio_df['Proprio'] = all_layers.T.Proprio

Somato_df = pd.DataFrame()
Somato_df['Somato'] = all_layers.T.Somato

for i, types in enumerate(upset_types_layers):
    MN_df[f'{upset_names[i]}'] = types.MN
    Proprio_df[f'{upset_names[i]}'] = types.Proprio
    Somato_df[f'{upset_names[i]}'] = types.Somato

# plot types
data = MN_df.iloc[1:len(MN_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Greens', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_VNC-layer-type_MN.pdf', bbox_inches='tight')

data = Proprio_df.iloc[1:len(Proprio_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Greens', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_VNC-layer-type_Proprio.pdf', bbox_inches='tight')


data = Somato_df.iloc[1:len(Somato_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Greens', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_VNC-layer-type_Somato.pdf', bbox_inches='tight')

# ascending neurons

upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(ascendings_layers_skids.T.values, ['MN', 'Proprio', 'Somato'], skids)
    #count_layers.columns = [f'Layer {x}' for x in range(0, len(count_layers.columns))]
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

MN_df = pd.DataFrame()
MN_df['MN'] = ascendings_layers.T.MN

Proprio_df = pd.DataFrame()
Proprio_df['Proprio'] = ascendings_layers.T.Proprio

Somato_df = pd.DataFrame()
Somato_df['Somato'] = ascendings_layers.T.Somato

for i, types in enumerate(upset_types_layers):
    MN_df[f'{upset_names[i]}'] = types.MN
    Proprio_df[f'{upset_names[i]}'] = types.Proprio
    Somato_df[f'{upset_names[i]}'] = types.Somato

# plot types
data = MN_df.iloc[1:len(MN_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Blues', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_ascending-layer-type_MN.pdf', bbox_inches='tight')

data = Proprio_df.iloc[1:len(Proprio_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Blues', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_ascending-layer-type_Proprio.pdf', bbox_inches='tight')


data = Somato_df.iloc[1:len(Somato_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Blues', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_ascending-layer-type_Somato.pdf', bbox_inches='tight')


####
# ds-dVNC neurons
####

upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(ds_dVNC_layers_skids.T.values, ['MN', 'Proprio', 'Somato'], skids)
    #count_layers.columns = [f'Layer {x}' for x in range(0, len(count_layers.columns))]
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

MN_df = pd.DataFrame()
MN_df['MN'] = ds_dVNC_layers.T.MN

Proprio_df = pd.DataFrame()
Proprio_df['Proprio'] = ds_dVNC_layers.T.Proprio

Somato_df = pd.DataFrame()
Somato_df['Somato'] = ds_dVNC_layers.T.Somato

for i, types in enumerate(upset_types_layers):
    MN_df[f'{upset_names[i]}'] = types.MN
    Proprio_df[f'{upset_names[i]}'] = types.Proprio
    Somato_df[f'{upset_names[i]}'] = types.Somato

# plot types
data = MN_df.iloc[1:len(MN_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Reds', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_ds-dVNC-layer-type_MN.pdf', bbox_inches='tight')

data = Proprio_df.iloc[1:len(Proprio_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Reds', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_ds-dVNC-layer-type_Proprio.pdf', bbox_inches='tight')


data = Somato_df.iloc[1:len(Somato_df), :]
mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
mask[:, 0] = [False]*len(data.index)

fig, axs = plt.subplots(
    1, 1, figsize=(2, 2)
)
annotations = data.astype(int).astype(str)
sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = 'Reds', cbar = False)
sns.heatmap(data, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)

plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_ds-dVNC-layer-type_Somato.pdf', bbox_inches='tight')
# layer numbers are off my one (should be layer 1, 2, etc.; not layer 2, 3, etc.)
# this only affects the axis, not the content

# %%
# identities of ascending neurons
from itertools import compress

ascending_pairs = Promat.extract_pairs_from_list(A1_ascending, pairs)[0]
ascending_types = [VNC_types_df.index[VNC_types_df==x] for x in ascending_pairs.leftid]

col = []
for types in ascending_types:
    if(len(types)==0):
        col.append('Unknown')
    if(len(types)>0):
        bool_types = [x for sublist in types for x in sublist]
        col.append(list(compress(['MN', 'Proprio', 'Somato'], bool_types)))

ascending_pairs['type'] = col

# %%
# pathways downstream of each dVNC pair
# with detailed VNC layering types
from tqdm import tqdm

source_dVNC, ds_dVNC = VNC_adj.downstream(dVNC, threshold, exclude=dVNC)
edges, ds_dVNC_cleaned = VNC_adj.edge_threshold(source_dVNC, ds_dVNC, threshold, direction='downstream')
edges[edges.overthres==True]

source_dVNC_cleaned = np.unique(edges[edges.overthres==True].upstream_pair_id)
source_dVNC_pairs = VNC_adj.adj_inter.loc[(slice(None), source_dVNC_cleaned), :].index
source_dVNC_pairs = [x[2] for x in source_dVNC_pairs]
source_dVNC_pairs = Promat.extract_pairs_from_list(source_dVNC_pairs, pairs)[0]

pair_paths = []
for index in tqdm(range(0, len(source_dVNC_pairs))):
    ds_dVNC = VNC_adj.downstream_multihop(list(source_dVNC_pairs.loc[index]), threshold, min_members = 0, hops=5)
    pair_paths.append(ds_dVNC)

# determine which neurons are only in one pathway
VNC_types_df = pd.DataFrame([x for x in VNC_types_df.index], index = VNC_types_df.values, columns = VNC_types_df.index.names)
sensory_type = list(VNC_types_df[(VNC_types_df.MN == False) & ((VNC_types_df.Proprio == True) | (VNC_types_df.Somato == True))].index)
motor_sens_MN = list(np.intersect1d(sensory_type, A1_MN))
motor_MN = list(np.setdiff1d(A1_MN, motor_sens_MN))
sensory_type = np.setdiff1d(sensory_type, A1_MN)

mixed_type = motor_sens_MN + list(VNC_types_df[(VNC_types_df.MN == True) & ((VNC_types_df.Proprio == True) | (VNC_types_df.Somato == True))].index)
motor_type = motor_MN + list(VNC_types_df[(VNC_types_df.MN == True) & (VNC_types_df.Proprio == False) & (VNC_types_df.Somato == False)].index)

# types of neurons
motor_type_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, motor_type)
motor_type_layers_ascend,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, np.intersect1d(motor_type, A1_ascending))
motor_type_layers_MN,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, np.intersect1d(motor_type, A1_MN))

sensory_type_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, sensory_type)
sensory_type_layers_ascend,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, np.intersect1d(sensory_type, A1_ascending))
sensory_type_layers_MN,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, np.intersect1d(sensory_type, A1_MN))

mixed_type_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, mixed_type)
mixed_type_layers_ascend,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, np.intersect1d(mixed_type, A1_ascending))
mixed_type_layers_MN,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, np.intersect1d(mixed_type, A1_MN))


fig, axs = plt.subplots(
    3, 3, figsize=(5, 7)
)
vmax = 4

ax = axs[0,0]
annotations = motor_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(motor_type_layers, annot = annotations, fmt = 's', cmap = 'Reds', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('Individual dVNCs Paths')
ax.set(title='Motor Pathway Exclusive')

ax = axs[0,1]
annotations = motor_type_layers_MN.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(motor_type_layers_MN, annot = annotations, fmt = 's', cmap = 'Reds', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')
ax.set(title='Motor Exclusive Motorneurons')

ax = axs[0,2]
annotations = motor_type_layers_ascend.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(motor_type_layers_ascend, annot = annotations, fmt = 's', cmap = 'Reds', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')
ax.set(title='Motor Exclusive Ascending')

vmax = 20

ax = axs[1,0]
annotations = sensory_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(sensory_type_layers, annot = annotations, fmt = 's', cmap = 'Blues', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('Individual dVNCs Paths')
ax.set(title='VNC Sensory Pathway Exclusive')

ax = axs[1,1]
annotations = sensory_type_layers_MN.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(sensory_type_layers_MN, annot = annotations, fmt = 's', cmap = 'Blues', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')
ax.set(title='VNC Sensory Motorneurons')

ax = axs[1,2]
annotations = sensory_type_layers_ascend.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(sensory_type_layers_ascend, annot = annotations, fmt = 's', cmap = 'Blues', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')
ax.set(title='VNC Sensory Ascending')


vmax = 80

ax = axs[2,0]
annotations = mixed_type_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(mixed_type_layers, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, vmax = vmax, cbar = False)
ax.set_yticks([])
ax.set_ylabel('Individual dVNCs Paths')
ax.set(title='Mixed Pathway Neurons')

ax = axs[2,1]
annotations = mixed_type_layers_MN.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(mixed_type_layers_MN, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, vmax = 60, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Mixed Pathway Motorneurons')

ax = axs[2,2]
annotations = mixed_type_layers_ascend.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(mixed_type_layers_ascend, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, vmax = 20, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Mixed Pathway Ascending')

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_dVNC_paths.pdf', bbox_inches='tight')

# %%
# pathways downstream of each dVNC pair
# simple VNC layering

all_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, VNC_adj.adj.index) # include all neurons to get total number of neurons per layer
motor_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, A1_MN)
ascending_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, A1_ascending)

VNC_sens_type = list(np.setdiff1d(VNC_types_df[(VNC_types_df.MN == False) & ((VNC_types_df.Proprio == True) | (VNC_types_df.Somato == True))].index, A1_MN))
VNC_sens_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, VNC_sens_type)

order = [16, 0, 2, 11, 1, 5, 7, 12, 13, 8, 3, 9, 10, 15, 4, 6, 14]
all_simple_layers = all_simple_layers.iloc[order, :]
motor_simple_layers = motor_simple_layers.iloc[order, :]
ascending_simple_layers = ascending_simple_layers.iloc[order, :]
VNC_sens_layers = VNC_sens_layers.iloc[order, :]

fig, axs = plt.subplots(
    1, 4, figsize=(4, 2.25)
)

ax = axs[0]
annotations = all_simple_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(all_simple_layers, annot = annotations, fmt = 's', cmap = 'Greens', ax = ax, vmax = 80, cbar = False)
ax.set_yticks([])
ax.set_ylabel('Individual dVNCs Paths')
ax.set(title='Pathway Overview')

ax = axs[1]
annotations = motor_simple_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(motor_simple_layers, annot = annotations, fmt = 's', cmap = 'Reds', ax = ax, vmax = 80, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Motorneurons')

ax = axs[2]
annotations = ascending_simple_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(ascending_simple_layers, annot = annotations, fmt = 's', cmap = 'Blues', ax = ax, vmax = 20, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Ascendings')

ax = axs[3]
annotations = VNC_sens_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(VNC_sens_layers, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, vmax = 20, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='A1 Sensory Interneurons')

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_dVNC_paths_simple.pdf', bbox_inches='tight')

# %%
# plot by individual dVNC

# split plot types by dVNC pair
dVNC_pairs = all_simple_layers.index
layer_types = [all_simple_layers, motor_simple_layers, ascending_simple_layers, VNC_sens_layers]
col = ['Greens', 'Reds', 'Blues', 'Purples']

dVNC_list = []
for pair in dVNC_pairs:
    mat = np.zeros(shape=(len(layer_types), len(all_simple_layers.columns)))
    for i, layer_type in enumerate(layer_types):
        mat[i, :] = layer_type.loc[pair]

    dVNC_list.append(mat)

# loop through pairs to plot
for i, dVNC in enumerate(dVNC_list):

    data = pd.DataFrame(dVNC, index = ['All', 'Motor', 'Ascend', 'Sens IN'])
    mask_list = []
    for i_iter in range(0, len(data.index)):
        mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
        mask[i_iter, :] = [False]*len(data.columns)
        mask_list.append(mask)

    fig, axs = plt.subplots(
        1, 1, figsize=(.9, .5)
    )
    for j, mask in enumerate(mask_list):
        if((j == 0) | (j == 1)):
            vmax = 60
        if((j == 2) | (j == 3)):
            vmax = 20
        ax = axs
        annotations = data.astype(int).astype(str)
        annotations[annotations=='0']=''
        sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

    plt.savefig(f'VNC_interaction/plots/individual_dVNC_paths/{i}_dVNC-{dVNC_pairs[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')


# %%
# export different neuron types at each VNC layer

def readable_df(skids_list):
    max_length = max([len(x) for x in skids_list])

    df = pd.DataFrame()
    
    for i, layer in enumerate(skids_list):
    
        skids = list(layer)

        if(len(layer)==0):
            skids = ['']
        if(len(skids) != max_length):
            skids = skids + ['']*(max_length-len(skids))

        df[f'Layer {i}'] = skids

    return(df)

readable_df(ds_dVNC_layers_skids.MN).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ds_dVNC_MN_layers_{str(date.today())}.csv', index = False)
readable_df(ds_dVNC_layers_skids.Proprio).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ds_dVNC_Proprio_layers_{str(date.today())}.csv', index = False)
readable_df(ds_dVNC_layers_skids.Somato).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ds_dVNC_Somato_layers_{str(date.today())}.csv', index = False)

readable_df(ascendings_layers_skids.MN).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ascendings_MN_layers_{str(date.today())}.csv', index = False)
readable_df(ascendings_layers_skids.Proprio).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ascendings_Proprio_layers_{str(date.today())}.csv', index = False)
readable_df(ascendings_layers_skids.Somato).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ascendings_Somato_layers_{str(date.today())}.csv', index = False)

# %%
# export ds-dVNCs and dVNCs
pd.DataFrame(ds_dVNC).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_ds_dVNC_{str(date.today())}.csv', index = False)
pd.DataFrame(source_dVNC).to_csv(f'VNC_interaction/data/csvs/Threshold-{threshold}_source_dVNC_{str(date.today())}.csv', index = False)

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

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_connections_dVNC_A1.pdf')

# %%
# deprecated
# pathways identification
# which MNs have which dVNCs upstream?
from tqdm import tqdm

# identify dVNCs associated with MN pathways
ds_dVNC_MN = [x for sublist in list(ds_dVNC_layers_skids.MN) for x in sublist]
edges_MN_layers, ds_dVNC_MN = VNC_adj.edge_threshold(source_dVNC, ds_dVNC_MN, threshold, direction='downstream')

source_dVNC_MN = np.unique(edges_MN_layers[edges_MN_layers.overthres==True].upstream_pair_id)
source_dVNC_MN = VNC_adj.adj_inter.loc[(slice(None), source_dVNC_MN), :].index
source_dVNC_MN = [x[2] for x in source_dVNC_MN]
source_dVNC_MN = Promat.extract_pairs_from_list(source_dVNC_MN, pairs)[0]

motor_paths = []
for index in tqdm(range(0, len(source_dVNC_MN))):
    ds_dVNC = VNC_adj.downstream_multihop(list(source_dVNC_MN.loc[index]), threshold, min_members = 0, hops=4)
    motor_paths.append(ds_dVNC)

all_layers_motor = []
for layers in motor_paths:
    layers_motor = []
    for layer in layers:
        layer_motor = np.intersect1d(layer, A1_MN)
        layers_motor.append(layer_motor)

    all_layers_motor.append(layers_motor)
