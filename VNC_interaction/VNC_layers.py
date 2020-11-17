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
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

VNC_adj = Adjacency_matrix(adj.values, adj.index, pairs, inputs,'axo-dendritic')
#test.adj_inter.loc[(slice(None), slice(None), KC), (slice(None), slice(None), MBON)]

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_acess = pymaid.get_skids_by_annotation('mw A1 accessory neurons')

A1 = A1 + A1_acess

A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')
A1_proprio = pymaid.get_skids_by_annotation('mw A1 proprio')
A1_chordotonal = pymaid.get_skids_by_annotation('mw A1 chordotonals')
A1_noci = pymaid.get_skids_by_annotation('mw A1 noci')
A1_somato = pymaid.get_skids_by_annotation('mw A1 somato')

# %%
from connectome_tools.cascade_analysis import Celltype_Analyzer, Celltype

# VNC layering with respect to sensories or motorneurons
threshold = 0.01

###### 
# Modify this section if new layering groups need to be added
######

# manual add neuron groups of interest here and names
names = ['us-MN', 'ds-Proprio', 'ds-Chord', 'ds-Noci', 'ds-Somato']
general_names = ['pre-MN', 'Proprio', 'Chord', 'Noci', 'Somato']
all_source = A1_MN + A1_proprio + A1_chordotonal + A1_noci + A1_somato
min_members = 4

# manually determine upstream or downstream relation
us_A1_MN = VNC_adj.upstream_multihop(A1_MN, threshold, min_members=min_members, exclude = all_source)
ds_proprio = VNC_adj.downstream_multihop(A1_proprio, threshold, min_members=min_members, exclude = all_source)
ds_chord = VNC_adj.downstream_multihop(A1_chordotonal, threshold, min_members=min_members, exclude = all_source)
ds_noci = VNC_adj.downstream_multihop(A1_noci, threshold, min_members=min_members, exclude = all_source)
ds_somato = VNC_adj.downstream_multihop(A1_somato, threshold, min_members=min_members, exclude = all_source)

VNC_layers = [us_A1_MN, ds_proprio, ds_chord, ds_noci, ds_somato]

########
########

# how many neurons are included in layering?
all_included = [x for sublist in VNC_layers for subsublist in sublist for x in subsublist]

frac_included = len(np.intersect1d(A1, all_included))/len(A1)
print(f'Fraction VNC cells covered = {frac_included}')

# how similar are layers
celltypes = []
for ct_i, celltype in enumerate(VNC_layers):
    ct = [Celltype(f'{names[ct_i]}-{i}', layer) for i, layer in enumerate(celltype)]
    celltypes = celltypes + ct

VNC_analyzer = Celltype_Analyzer(celltypes)
fig, axs = plt.subplots(1, 1, figsize = (10, 10))
sns.heatmap(VNC_analyzer.compare_membership(), square = True, ax = axs)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_similarity_between_VNC_layers.pdf', bbox_inches='tight')

# %%
# number of VNC neurons per layer
VNC_layers = [[A1_MN] + us_A1_MN, [A1_proprio] + ds_proprio, [A1_chordotonal] + ds_chord, [A1_noci] + ds_noci, [A1_somato] + ds_somato]
all_layers, all_layers_skids = VNC_adj.layer_id(VNC_layers, general_names, A1)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(all_layers.T, annot=True, fmt='.0f', cmap = 'Greens', cbar = False, ax = axs)
ax.set_title(f'A1 Neurons; {frac_included*100:.0f}% included')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layers.pdf', bbox_inches='tight')
# %%
# where are ascendings in layering?

A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A00c = pymaid.get_skids_by_annotation('mw A00c')

A1_ascending = A1_ascending + A00c #include A00c's as ascending (they are not in A1, but in A4/5/6 and so have different annotations)

ascendings_layers, ascendings_layers_skids = VNC_adj.layer_id(VNC_layers, general_names, A1_ascending)

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

ds_dVNC_layers, ds_dVNC_layers_skids = VNC_adj.layer_id(VNC_layers, general_names, ds_dVNC)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(ds_dVNC_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('Downstream of dVNCs')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_dVNC_downstream_targets.pdf', bbox_inches='tight')
# %%
# location of special-case neurons in VNC layering
# gorogoro, basins, A00cs
gorogoro = pymaid.get_skids_by_annotation('gorogoro')
basins = pymaid.get_skids_by_annotation('a1basins')
A00c_layers, A00c_skids = VNC_adj.layer_id(VNC_layers, general_names, A00c)
gorogoro_layers, gorogoro_skids = VNC_adj.layer_id(VNC_layers, general_names, gorogoro)
basins_layers, basins_skids = VNC_adj.layer_id(VNC_layers, general_names, basins)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(A00c_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Purples', cbar = False, ax = ax)
ax.set_title('A00c location')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layers_A00c_location.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs 
sns.heatmap(gorogoro_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Purples', cbar = False, ax = ax)
ax.set_title('gorogoro location')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layers_gorogoro_location.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(basins_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Purples', cbar = False, ax = ax)
ax.set_title('basins location')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layers_basins_location.pdf', bbox_inches='tight')

# same neurons but checking if they are directly downstream of dVNCs
dsdVNC_A00c_layers, dsdVNC_A00c_skids = VNC_adj.layer_id(ds_dVNC_layers_skids.T.values, general_names, A00c)
dsdVNC_gorogoro_layers, dsdVNC_gorogoro_skids = VNC_adj.layer_id(ds_dVNC_layers_skids.T.values, general_names, A00c)
dsdVNC_basins_layers, dsdVNC_basins_skids = VNC_adj.layer_id(ds_dVNC_layers_skids.T.values, general_names, A00c)

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(dsdVNC_A00c_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('A00c location - ds-dVNCs')

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(dsdVNC_gorogoro_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('gorogoro location - ds-dVNCs')

fig, axs = plt.subplots(
    1, 1, figsize = (2.5, 3)
)
ax = axs
sns.heatmap(dsdVNC_basins_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('basins location - ds-dVNCs')

# conclusion - basins/gorogoro/A00c's don't receive direct dVNC input
# %%
# plot A1 structure together
plt.rcParams['font.size'] = 5

fig, axs = plt.subplots(
    1, 3, figsize = (3.25, 1.5)
)
ax = axs[0]
sns.heatmap(all_layers.T, cbar_kws={'label': 'Number of Neurons'}, annot = True, fmt='.0f', cmap = 'Greens', cbar = False, ax = ax)
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
plt.rcParams['font.size'] = 6

# %%
# upset plot of VNC types (MN, Proprio, Somato) with certain number of hops (hops_included)

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

hops_included = 3

VNC_types=[]
for celltype in VNC_layers:
    VNC_type = [x for layer in celltype[0:hops_included] for x in layer]
    VNC_types.append(VNC_type)

data = [x for cell_type in VNC_types for x in cell_type]
data = np.unique(data)

cats_simple = []
for skid in data:
    cat = []

    for i in range(0, len(general_names)):
        if(skid in VNC_types[i]):
            cat = cat + [f'{general_names[i]}']

    cats_simple.append(cat)

VNC_types_df = from_memberships(cats_simple, data = data)

counts = []
for celltype in np.unique(cats_simple):
    count = 0
    for cat in cats_simple:
        if(celltype == cat):
            count += 1

    counts.append(count)

# how many neurons belong to a category with X hops_included
coverage = []
for celltype in VNC_layers:
    celltype_list = [x for sublist in celltype[0:hops_included] for x in sublist]
    coverage = coverage + celltype_list
coverage = np.unique(coverage)

upset = from_memberships(np.unique(cats_simple), data = counts)
plot(upset, sort_categories_by = 'cardinality')
plt.title(f'{len(np.intersect1d(A1, coverage))/len(A1)*100:.2f}% of A1 neurons covered')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC-signal-type_hops-{hops_included}.pdf', bbox_inches='tight')

# %%
# upset plot of VNC types including layers (MN-0, -1, -2, Proprio-0, -1, 2, Somat-0, -1, -2, etc.)

VNC_type_layers = [x for sublist in VNC_layers for x in sublist]
VNC_type_layer_names = [x.name for x in celltypes]
data = [x for cell_type in VNC_type_layers for x in cell_type]
data = np.unique(data)

cats_complex = []
for skid in data:
    cat = []
    for i, layer in enumerate(VNC_type_layers):
        if(skid in layer):
            cat = cat + [VNC_type_layer_names[i]]

    cats_complex.append(cat)

VNC_type_layers_df = from_memberships(cats_complex, data = data)

counts = []
for celltype in np.unique(cats_complex):
    count = 0
    for cat in cats_complex:
        if(celltype == cat):
            count += 1

    counts.append(count)

upset_complex = from_memberships(np.unique(cats_complex), data = counts)
plot(upset_complex, sort_categories_by = None)
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC_layer_signal_type.pdf', bbox_inches='tight')

# %%
# supplementary plot with exclusive MN, proprio, and Somato types
from upsetplot import UpSet

def upset_subset(upset_types_layers, upset_names, column_layer, layer_structure_name, column_name, height, width_col, color, cat_order):
    # column_layer example = all_layers.T.{column_name}
        
    df = pd.DataFrame()
    df[f'{column_name}'] = column_layer # add first column of layering information for baseline, has nothing to do with UpSet data 

    for i, types in enumerate(upset_types_layers):
        df[f'{upset_names[i]}'] = types.loc[:, f'{column_name}']

    # plot types
    data = df.iloc[1:len(df), :]
    nonzero_cols = data.sum(axis=0)!=0
    data_cleaned = data.loc[:, nonzero_cols]

    data_cleaned_summed = data_cleaned.sum(axis=1)
    signal = []
    for i in reversed(range(0, len(data_cleaned_summed))):
        if(data_cleaned_summed.iloc[i]>0):
            signal=i+1
            break

    data_cleaned = data_cleaned.iloc[0:signal, :]

    mask = np.full((len(data_cleaned.index),len(data_cleaned.columns)), True, dtype=bool)
    mask[:, 0] = [False]*len(data_cleaned.index)

    fig, axs = plt.subplots(
        1, 1, figsize=(width_col*len(data_cleaned.columns), height*len(data_cleaned.index))
    )

    annotations = data_cleaned.astype(int).astype(str)
    annotations[annotations=='0']=''
    sns.heatmap(data_cleaned, annot = annotations, fmt = 's', mask = mask, ax=axs, cmap = color, cbar = False)
    sns.heatmap(data_cleaned, annot = annotations, fmt = 's', mask = np.invert(mask), ax=axs, cbar = False)
    plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_{layer_structure_name}_{column_name}.pdf', bbox_inches='tight')

    #Upset plot
    nonzero_permut = [upset_names[i] for i, boolean in enumerate(nonzero_cols[1:]) if boolean==True]
    nonzero_counts = data_cleaned.sum(axis=0)[1:]

    permut_types_df = from_memberships(nonzero_permut, nonzero_counts)
    permut_types_df.index = permut_types_df.index.reorder_levels(cat_order)
    plot(permut_types_df, sort_categories_by = None)
    plt.savefig(f'VNC_interaction/plots/supplemental/Supplemental_{layer_structure_name}_{column_name}_Upset.pdf', bbox_inches='tight')

# all  permutations when considering hops_included UpSet plot
permut = UpSet(upset).intersections.index

# names of these permutations
upset_names = []
for sublist in permut:
    upset_names.append([permut.names[i] for i, boolean in enumerate(sublist) if boolean==True ])

# reorder multiindex of VNC_types_df according to permut
VNC_types_df = VNC_types_df.reorder_levels(permut.names)

# skids for each permutation
upset_skids = [VNC_types_df.loc[x] for x in permut]

cat_order = ['pre-MN', 'Proprio', 'Somato', 'Chord', 'Noci']

upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(all_layers_skids.T.values, general_names, skids.values) 
    upset_types_layers.append(count_layers.T) 
    upset_types_skids.append(layer_skids)

for layer_type in all_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, all_layers.T.loc[:, layer_type], 'VNC_layers', layer_type, 0.2, 0.2, 'Greens', cat_order)

upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(ascendings_layers_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in ascendings_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, ascendings_layers.T.loc[:, layer_type], 'Ascendings_layers', layer_type, 0.2, 0.2, 'Blues', cat_order)

upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(ds_dVNC_layers_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in ds_dVNC_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, ds_dVNC_layers.T.loc[:, layer_type], 'ds-dVNCs_layers', layer_type, 0.2, 0.2, 'Reds', cat_order)

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
        col.append(list(compress(general_names, bool_types)))

ascending_pairs['type'] = col

# %%
# pathways downstream of each dVNC pair
# with detailed VNC layering types
# ** rework this section for final figure (not including Chord, Noci, etc. right now)
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
