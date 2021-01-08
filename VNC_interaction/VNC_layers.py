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
A1_classII_III = pymaid.get_skids_by_annotation('mw A1 somato')
A1_external = pymaid.get_skids_by_annotation('mw A1 external sensories')
# A1 vtd doesn't make 

# %%
from connectome_tools.cascade_analysis import Celltype_Analyzer, Celltype

# VNC layering with respect to sensories or motorneurons
threshold = 0.01

###### 
# Modify this section if new layering groups need to be added
######

# manual add neuron groups of interest here and names
names = ['us-MN', 'ds-Proprio', 'ds-Noci', 'ds-Chord', 'ds-ClassII_III', 'ds-ES']
general_names = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES']
all_source = A1_MN + A1_proprio + A1_chordotonal + A1_noci + A1_classII_III + A1_external
min_members = 4

# manually determine upstream or downstream relation
us_A1_MN = VNC_adj.upstream_multihop(A1_MN, threshold, min_members=min_members, exclude = all_source)
ds_proprio = VNC_adj.downstream_multihop(A1_proprio, threshold, min_members=min_members, exclude = all_source)
ds_chord = VNC_adj.downstream_multihop(A1_chordotonal, threshold, min_members=min_members, exclude = all_source)
ds_noci = VNC_adj.downstream_multihop(A1_noci, threshold, min_members=min_members, exclude = all_source)
ds_classII_III = VNC_adj.downstream_multihop(A1_classII_III, threshold, min_members=min_members, exclude = all_source)
ds_external = VNC_adj.downstream_multihop(A1_external, threshold, min_members=min_members, exclude = all_source)

VNC_layers = [us_A1_MN, ds_proprio, ds_noci, ds_chord, ds_classII_III, ds_external]
cat_order = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES']

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
VNC_layers = [[A1_MN] + us_A1_MN, [A1_proprio] + ds_proprio, [A1_noci] + ds_noci, [A1_chordotonal] + ds_chord, [A1_classII_III] + ds_classII_III, [A1_external] + ds_external]
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
source_dVNC, ds_dVNC = VNC_adj.downstream(source=dVNC, threshold=threshold, exclude=dVNC)
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
sns.heatmap(all_layers.T.loc[:, cat_order], cbar_kws={'label': 'Number of Neurons'}, annot = True, fmt='.0f', cmap = 'Greens', cbar = False, ax = ax)
ax.set_title('A1 Neurons')

ax = axs[1]
sns.heatmap(ascendings_layers.T.loc[:, cat_order], cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Blues', cbar = False, ax = ax)
ax.set_title('Ascendings')
ax.set_yticks([])

ax = axs[2]
sns.heatmap(ds_dVNC_layers.T.loc[:, cat_order], cbar_kws={'label': 'Number of Neurons'}, annot = True, cmap = 'Reds', cbar = False, ax = ax)
ax.set_title('ds-dVNCs')
ax.set_yticks([])

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_A1_structure.pdf', bbox_inches='tight')
plt.rcParams['font.size'] = 6

# %%
# upset plot of VNC types (MN, Proprio, etc.) with certain number of hops (hops_included)

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

hops_included = 2

VNC_types=[]
for celltype in VNC_layers:
    VNC_type = [x for layer in celltype[0:hops_included+1] for x in layer]
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
    celltype_list = [x for sublist in celltype[0:hops_included+1] for x in sublist]
    coverage = coverage + celltype_list
coverage = np.unique(coverage)

# threhold small categories (<=4 neurons) to simplify the plot
upset_thres = [x>4 for x in counts]
cats_simple_cut = [x for i, x in enumerate(np.unique(cats_simple)) if upset_thres[i]]
counts_cut = [x for i, x in enumerate(counts) if upset_thres[i]]

# set up the data variable
upset = from_memberships(np.unique(cats_simple_cut), data = counts_cut)
upset.index = upset.index.reorder_levels(cat_order) # order categories

plot(upset, sort_categories_by = None)
plt.title(f'{len(np.intersect1d(A1, coverage))/len(A1)*100:.2f}% of A1 neurons covered')
plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_VNC-signal-type_hops-{hops_included}.pdf', bbox_inches='tight')

# %%
# upset plot of VNC types including layers (MN-0, -1, -2, Proprio-0, -1, 2, Somat-0, -1, -2, etc.)

VNC_layers_nostart = [us_A1_MN, ds_proprio, ds_chord, ds_noci, ds_classII_III, ds_external]

VNC_type_layers = [x for sublist in VNC_layers_nostart for x in sublist]
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

    # added for some edge cases
    if(sum(column_layer)==0):
        return()

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
    cat_order = [cat_order[i] for i, x in enumerate([x in permut_types_df.index.names for x in cat_order]) if x==True] # remove any cat_order names if missing; added for edge cases
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

# all VNC layers
upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(all_layers_skids.T.values, general_names, skids.values) 
    upset_types_layers.append(count_layers.T) 
    upset_types_skids.append(layer_skids)

for layer_type in all_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, all_layers.T.loc[:, layer_type], 'VNC_layers', layer_type, 0.2, 0.2, 'Greens', cat_order)

# ascending locations
upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(ascendings_layers_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in ascendings_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, ascendings_layers.T.loc[:, layer_type], 'Ascendings_layers', layer_type, 0.2, 0.2, 'Blues', cat_order)

# ds-dVNC_layers
upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(ds_dVNC_layers_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in ds_dVNC_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, ds_dVNC_layers.T.loc[:, layer_type], 'ds-dVNCs_layers', layer_type, 0.2, 0.2, 'Reds', cat_order)

# %%
# locations of basins/goro/A00c
# not working in this version of the script for some reason

# basin locations
upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(basins_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in basins_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, all_layers.T.loc[:, layer_type], 'basin_layers', layer_type, 0.2, 0.2, 'Greens', cat_order)

# gorogoro locations
upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(gorogoro_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in gorogoro_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, all_layers.T.loc[:, layer_type], 'goro_layers', layer_type, 0.2, 0.2, 'Greens', cat_order)
'''
# A00c locations
upset_types_layers = []
upset_types_skids = []
for skids in upset_skids:
    count_layers, layer_skids = VNC_adj.layer_id(A00c_skids.T.values, general_names, skids.values)
    upset_types_layers.append(count_layers.T)
    upset_types_skids.append(layer_skids)

for layer_type in A00c_layers.T.columns:
    upset_subset(upset_types_layers, upset_names, ascendings_layers.T.loc[:, layer_type], 'A00c_layers', layer_type, 0.2, 0.2, 'Blues', cat_order)
'''
# %%
# identities of ascending neurons
# further develop this to identify next hit on "unknown" ascendings

from itertools import compress
from tqdm import tqdm

# ascending identities using 2 hops from sensory/motorneurons
# no difference between 1st-order and 2nd-order
ascending_pairs = Promat.extract_pairs_from_list(A1_ascending, pairs)[0]
VNC_types_df = VNC_types_df.reorder_levels(general_names)
ascending_types = [VNC_types_df.index[VNC_types_df==x] for x in ascending_pairs.leftid]

col = []
for types in ascending_types:
    if(len(types)==0):
        col.append('Unknown')
    if(len(types)>0):
        bool_types = [x for sublist in types for x in sublist]
        col.append(list(compress(general_names, bool_types)))

ascending_pairs['type'] = col

# multiple-hop matrix of A1 sensories to A1 ascendings
MN_pairs = Promat.extract_pairs_from_list(A1_MN, pairs)[0]
proprio_pairs = Promat.extract_pairs_from_list(A1_proprio, pairs)[0]
chord_pairs = Promat.extract_pairs_from_list(A1_chordotonal, pairs)[0]
noci_pairs = Promat.extract_pairs_from_list(A1_noci, pairs)[0]
classII_III_pairs = Promat.extract_pairs_from_list(A1_classII_III, pairs)[0]
external_pairs = Promat.extract_pairs_from_list(A1_external, pairs)[0]

sens_pairs = pd.concat([proprio_pairs, noci_pairs, chord_pairs, classII_III_pairs, external_pairs])
sens_pairs.index = range(0, len(sens_pairs))

# determining hops from each sensory modality for each ascending neuron (using all hops)

# sensory modalities generally
sens_paths = VNC_layers
ascending_layers,ascending_skids = VNC_adj.layer_id(sens_paths, general_names, A1_ascending)
sens_asc_mat, sens_asc_mat_plotting = VNC_adj.hop_matrix(ascending_skids.T, general_names, ascending_pairs.leftid, include_start=True)

# hops from each modality
sens_asc_mat.T

# hops from each modality, threshold = 2
hops = 2

sens_asc_mat_thresh = sens_asc_mat.T.copy()
sens_asc_mat_thresh[sens_asc_mat_thresh>hops]=0
sens_asc_mat_thresh

# sorting ascendings by type
proprio_order1 = list(sens_asc_mat_thresh.index[sens_asc_mat_thresh.Proprio==1])
chord_order1_2 = list(sens_asc_mat_thresh.index[sens_asc_mat_thresh.Chord==1]) + list(sens_asc_mat_thresh.index[(sens_asc_mat_thresh.Chord==2) & (sens_asc_mat_thresh.Noci==0)])
classII_III_order1 = list(sens_asc_mat_thresh.index[sens_asc_mat_thresh.ClassII_III==1])
noci_order2 = list(sens_asc_mat_thresh.index[sens_asc_mat_thresh.Noci==2])
unknown = list(sens_asc_mat_thresh.index[(sens_asc_mat_thresh!=0).sum(axis=1)==0])

# manual reordering based on secondary sensory partners
proprio_order1 = [proprio_order1[i] for i in [1,2,0]]
#chord_order1_2 = [chord_order1_2[i] for i in [1,2,0,5,3,4]] #lost 11455472
noci_order2 = [noci_order2[i] for i in [3,1,2,0]]
unknown = [unknown[i] for i in [2, 3, 0, 1, 4, 5, 6]]

sens_asc_order = proprio_order1 + chord_order1_2 + classII_III_order1 + noci_order2 + unknown
annotations = sens_asc_mat.loc[:, sens_asc_order].astype(int).astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1,figsize=(1.75,1))
sens_asc_mat_plotting_2 = sens_asc_mat_plotting.copy()
sens_asc_mat_plotting_2 = sens_asc_mat_plotting_2.loc[:, sens_asc_order]
sens_asc_mat_plotting_2[sens_asc_mat_plotting_2<7] = 0
sens_asc_mat_plotting_2[sens_asc_mat_plotting_2.loc[:, proprio_order1]<8] = 0
sns.heatmap(sens_asc_mat_plotting_2, annot=annotations, fmt = 's', cmap = 'Blues', ax=ax, cbar = False)
plt.xticks(range(len(sens_asc_mat_plotting_2.columns)), sens_asc_mat_plotting_2.columns, ha='left')
plt.setp(ax.get_xticklabels(), Fontsize=4);
ax.tick_params(left=False, bottom=False, length=0)
plt.savefig(f'VNC_interaction/plots/individual_asc_paths/Supplemental_ascending_identity_matrix.pdf', bbox_inches='tight')

# export raw data using ascending type sorting
sens_asc_mat_thresh.T.loc[:, sens_asc_order].to_csv(f'VNC_interaction/plots/individual_asc_paths/ascending_identity_{hops}-hops.csv')
sens_asc_mat.loc[:, sens_asc_order].to_csv(f'VNC_interaction/plots/individual_asc_paths/ascending_identity_all-hops.csv')

# %%
#UpSet based on first two hops from each sensory modality
#doesn't do it based on a pairwise measure?
#**** probably needs more work****

hops_included = 2

celltypes_2o = []
for ct_i, celltype in enumerate(VNC_layers_nostart):
    ct = [Celltype(f'{names[ct_i]}-{i+1}', layer) for i, layer in enumerate(celltype) if i<2]
    celltypes_2o = celltypes_2o + ct

celltypes_2o_layers = [x.get_skids() for x in celltypes_2o]
celltypes_2o_names = [x.name for x in celltypes_2o]
data = [x for cell_type in celltypes_2o_layers for x in cell_type]
data = np.unique(data)

cats_2o = []
for skid in data:
    cat = []
    for i, layer in enumerate(celltypes_2o_layers):
        if(skid in layer):
            cat = cat + [celltypes_2o_names[i]]

    cats_2o.append(cat)

celltypes_2o_df = from_memberships(cats_2o, data = data)

counts = []
for celltype in np.unique(cats_2o):
    count = 0
    for cat in cats_2o:
        if(celltype == cat):
            count += 1

    counts.append(count)

upset_2o = from_memberships(np.unique(cats_2o), data = counts)
plot(upset_2o, sort_categories_by = None)
#plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_ascendings_signal_type.pdf', bbox_inches='tight')

# UpSet based on first two hops, ascendings only
data = A1_ascending

cats_2o_asc = []
for skid in data:
    cat = []
    for i, layer in enumerate(celltypes_2o_layers):
        if(skid in layer):
            cat = cat + [celltypes_2o_names[i]]

    cats_2o_asc.append(cat)

celltypes_2o_asc_df = from_memberships(cats_2o_asc, data = data)

counts = []
for celltype in np.unique(cats_2o_asc):
    count = 0
    for cat in cats_2o:
        if(celltype == cat):
            count += 1

    counts.append(count)

upset_2o_asc = from_memberships(np.unique(cats_2o_asc), data = counts)
plot(upset_2o_asc, sort_categories_by = None)

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
VNC_types_index = pd.DataFrame([x for x in VNC_types_df.index], index = VNC_types_df.values, columns = VNC_types_df.index.names)
sensory_type = list(VNC_types_index[(VNC_types_index['pre-MN'] == False)].index)
motor_sens_MN = list(np.intersect1d(sensory_type, A1_MN))
motor_MN = list(np.setdiff1d(A1_MN, motor_sens_MN))
sensory_type = np.setdiff1d(sensory_type, A1_MN)

mixed_type = motor_sens_MN + list(VNC_types_index[(VNC_types_index['pre-MN'] == True) & (sum(VNC_types_index.iloc[0, VNC_types_index.columns != 'pre-MN'])>0)].index)
motor_type = motor_MN + list(VNC_types_index[(VNC_types_index['pre-MN'] == True) & (sum(VNC_types_index.iloc[0, VNC_types_index.columns != 'pre-MN'])==0)].index)

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

# set up sensory categories to test against dVNC pathways
# excluded motorneurons from these categories
proprio_1o = list(np.setdiff1d(ds_proprio[0], A1_MN))
proprio_2o = list(np.setdiff1d(ds_proprio[1], A1_MN))
somato_1o = list(np.setdiff1d(ds_classII_III[0] + ds_chord[0] + ds_noci[0] + ds_external[0], A1_MN))
somato_2o = list(np.setdiff1d(ds_classII_III[1] + ds_chord[1] + ds_noci[1] + ds_external[1], A1_MN))
sens_12o = np.unique(proprio_1o + proprio_2o + somato_1o + somato_2o)
sens_1o = np.unique(proprio_1o + somato_1o)
sens_2o = np.unique(proprio_2o + somato_2o)

# check overlap between these sensory categories (just for curiosity)
A1_ct = Celltype('A1_all', A1)
proprio_1o_ct = Celltype('Proprio 1o', proprio_1o)
proprio_2o_ct = Celltype('Proprio 2o', proprio_2o)
somato_1o_ct = Celltype('Somato 1o', somato_1o)
somato_2o_ct = Celltype('Somato 2o', somato_2o)
sens_12o_ct = Celltype('All Sens 1o/2o', sens_12o)
sens_1o_ct = Celltype('All Sens 1o', sens_1o)
sens_2o_ct = Celltype('All Sens 2o', sens_2o)

cta = Celltype_Analyzer([A1_ct, sens_12o_ct, sens_1o_ct, sens_2o_ct, proprio_1o_ct, proprio_2o_ct, somato_1o_ct, somato_2o_ct])
sns.heatmap(cta.compare_membership(), annot=True)

#VNC_sens_type = list(np.setdiff1d(VNC_types_index[(VNC_types_index['pre-MN'] == False)].index, A1_MN))
#VNC_sens_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, VNC_sens_type)

# identifying different cell types in dVNC pathways
all_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, VNC_adj.adj.index) # include all neurons to get total number of neurons per layer
motor_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, A1_MN)
ascending_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, A1_ascending) 
goro_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, gorogoro)
basins_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, basins)
A00c_simple_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, A00c) # no contact
proprio_1o_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, proprio_1o)
proprio_2o_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, proprio_2o)
somato_1o_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, somato_1o)
somato_2o_layers,_ = VNC_adj.layer_id(pair_paths, source_dVNC_pairs.leftid, somato_2o)

# order manually identified for figure
dVNC_order = [15639294, 18305433, 10553248, 10728328, # type-1: only to MN
                19361427, 11013583, 19298644, 5690425, 6446394, # type-2: immediately to ascending
                17777031, 10609443,                             # type-3: ascending on the way to MN
                20556072, 10728333, 19298625, 10018150,          # type-4: ascending after MN
                3979181,                                        # other: gorogoro first order
                7227010, 16851496, 17053270]                    # supplemental: terminates in A1

all_simple_layers = all_simple_layers.loc[dVNC_order, :]
motor_simple_layers = motor_simple_layers.loc[dVNC_order, :]
ascending_simple_layers = ascending_simple_layers.loc[dVNC_order, :]
proprio_1o_layers = proprio_1o_layers.loc[dVNC_order, :]
proprio_2o_layers = proprio_2o_layers.loc[dVNC_order, :]
somato_1o_layers = somato_1o_layers.loc[dVNC_order, :]
somato_2o_layers = somato_2o_layers.loc[dVNC_order, :]

fig, axs = plt.subplots(
    1, 7, figsize=(7, 2.25)
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
annotations = proprio_1o_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(proprio_1o_layers, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, vmax = 50, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Proprio 1o')

ax = axs[4]
annotations = proprio_2o_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(proprio_2o_layers, annot = annotations, fmt = 's', cmap = 'Purples', ax = ax, vmax = 50, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Proprio 2o')

ax = axs[5]
annotations = somato_1o_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(somato_1o_layers, annot = annotations, fmt = 's', cmap = 'GnBu', ax = ax, vmax = 50, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Somato 1o')

ax = axs[6]
annotations = somato_2o_layers.astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(somato_2o_layers, annot = annotations, fmt = 's', cmap = 'GnBu', ax = ax, vmax = 50, cbar = False)
ax.set_yticks([])
ax.set_ylabel('')
ax.set(title='Somato 2o')

plt.savefig(f'VNC_interaction/plots/Threshold-{threshold}_individual_dVNC_paths_simple.pdf', bbox_inches='tight')

# %%
# plot by individual dVNC

# split plot types by dVNC pair
dVNC_pairs = all_simple_layers.index
layer_types = [all_simple_layers, motor_simple_layers, ascending_simple_layers, proprio_1o_layers, proprio_2o_layers, 
                somato_1o_layers, somato_2o_layers, goro_simple_layers, basins_simple_layers, A00c_simple_layers]
col = ['Greens', 'Reds', 'Blues', 'Purples', 'Purples', 'GnBu', 'GnBu', 'Reds', 'Purples', 'Blues']

dVNC_list = []
for pair in dVNC_pairs:
    mat = np.zeros(shape=(len(layer_types), len(all_simple_layers.columns)))
    for i, layer_type in enumerate(layer_types):
        mat[i, :] = layer_type.loc[pair]

    dVNC_list.append(mat)

# loop through pairs to plot
for i, dVNC in enumerate(dVNC_list):

    data = pd.DataFrame(dVNC, index = ['All', 'Motor', 'Ascend', 'Proprio-1', 'Proprio-2', 'Somato-1', 'Somato-2', 'Gorogoro', 'Basins', 'A00c'])
    mask_list = []
    for i_iter in range(0, len(data.index)):
        mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
        mask[i_iter, :] = [False]*len(data.columns)
        mask_list.append(mask)

    fig, axs = plt.subplots(
        1, 1, figsize=(.8, 1.25)
    )
    for j, mask in enumerate(mask_list):
        if((j in [0,1])):
            vmax = 60
        if((j in [2])):
            vmax = 20
        if((j in [3, 4, 5, 6])):
            vmax = 40
        if((j in [7, 8, 9])):
            vmax = 10
        ax = axs
        annotations = data.astype(int).astype(str)
        annotations[annotations=='0']=''
        sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

    plt.savefig(f'VNC_interaction/plots/individual_dVNC_paths/{i}_dVNC-{dVNC_pairs[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')

# %%
# main figure summary of individual dVNC paths

MN_exclusive= [15639294, 18305433, 10553248, 10728328]
ascending1 = [19361427, 11013583, 19298644, 5690425, 6446394]
ascending2 = [17777031, 10609443]
ascending_postMN = [20556072, 10728333, 19298625, 10018150]
dVNC_types_name = ['MN-exclusive', 'Ascending-1o', 'Ascending-2o', 'Ascending-postMN']

dVNC_types = [MN_exclusive, ascending1, ascending2, ascending_postMN]
dVNC_types = [[Promat.get_paired_skids(x, pairs) for x in sublist] for sublist in dVNC_types] # convert leftid's to both skids from each pair
dVNC_types = [sum(x, []) for x in dVNC_types] # unlist nested lists

# multihop downstream
type_paths = []
for index in tqdm(range(0, len(dVNC_types))):
    ds_dVNC = VNC_adj.downstream_multihop(list(dVNC_types[index]), threshold, min_members = 0, hops=5)
    type_paths.append(ds_dVNC)

# identifying different cell types in dVNC pathways
all_simple_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, VNC_adj.adj.index) # include all neurons to get total number of neurons per layer
motor_simple_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, A1_MN)
ascending_simple_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, A1_ascending) 
goro_simple_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, gorogoro)
basins_simple_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, basins)
A00c_simple_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, A00c) # no contact
proprio_1o_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, proprio_1o)
proprio_2o_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, proprio_2o)
somato_1o_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, somato_1o)
somato_2o_layers_type,_ = VNC_adj.layer_id(type_paths, dVNC_types_name, somato_2o)

# split plot types by dVNC pair
layer_types = [all_simple_layers_type, motor_simple_layers_type, ascending_simple_layers_type, proprio_1o_layers_type, proprio_2o_layers_type, 
                somato_1o_layers_type, somato_2o_layers_type, goro_simple_layers_type, basins_simple_layers_type, A00c_simple_layers_type]
col = ['Greens', 'Reds', 'Blues', 'Purples', 'Purples', 'GnBu', 'GnBu', 'Reds', 'Purples', 'Blues']

dVNC_type_list = []
for name in dVNC_types_name:
    mat = np.zeros(shape=(len(layer_types), len(all_simple_layers.columns)))
    for i, layer_type in enumerate(layer_types):
        mat[i, :] = layer_type.loc[name]

    dVNC_type_list.append(mat)

# loop through pairs to plot
for i, dVNC in enumerate(dVNC_type_list):

    data = pd.DataFrame(dVNC, index = ['All', 'Motor', 'Ascend', 'Proprio-1', 'Proprio-2', 'Somato-1', 'Somato-2', 'Gorogoro', 'Basins', 'A00c'])
    mask_list = []
    for i_iter in range(0, len(data.index)):
        mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
        mask[i_iter, :] = [False]*len(data.columns)
        mask_list.append(mask)

    fig, axs = plt.subplots(
        1, 1, figsize=(.8, 1.25)
    )
    for j, mask in enumerate(mask_list):
        if((j in [0,1])):
            vmax = 60
        if((j in [2])):
            vmax = 20
        if((j in [3, 4, 5, 6])):
            vmax = 40
        if((j in [7, 8, 9])):
            vmax = 10
        ax = axs
        annotations = data.astype(int).astype(str)
        annotations[annotations=='0']=''
        sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

    plt.savefig(f'VNC_interaction/plots/individual_dVNC_paths/Type_{i}_dVNC-{dVNC_types_name[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')

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
