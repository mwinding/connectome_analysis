#%%
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.cluster_analysis as clust

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj = pm.Promat.pull_adj('ad', subgraph='brain and accessory')
edges = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

# ignore all outputs to A1 neurons; this will make sure dVNC feedback only goes to brain
# while also allowing ascending -> dVNC connections
VNC_neurons = pymaid.get_skids_by_annotation('mw A1 neurons paired') + pymaid.get_skids_by_annotation('mw A00c')
edges = edges[[x not in VNC_neurons for x in edges.downstream_skid]]

pairs = pm.Promat.get_pairs()
dVNCs = pymaid.get_skids_by_annotation('mw dVNC')

dVNC_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids_bothsides', skids=dVNCs, use_skids=True)

# %%
# dVNC projectome data prep

import cmasher as cmr

projectome = pd.read_csv('data/projectome/projectome_adjacency.csv', index_col = 0, header = 0)
projectome.index = [str(x) for x in projectome.index]

# identify meshes
meshes = ['Brain Hemisphere left', 'Brain Hemisphere right', 'SEZ_left', 'SEZ_right', 'T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right', 'A1_left', 'A1_right', 'A2_left', 'A2_right', 'A3_left', 'A3_right', 'A4_left', 'A4_right', 'A5_left', 'A5_right', 'A6_left', 'A6_right', 'A7_left', 'A7_right', 'A8_left', 'A8_right']

pairOrder_dVNC = [x for sublist in zip(dVNC_pairs.leftid, dVNC_pairs.rightid) for x in sublist]

input_projectome = projectome.loc[meshes, [str(x) for x in pairOrder_dVNC]]
output_projectome = projectome.loc[[str(x) for x in pairOrder_dVNC], meshes]

dVNC_projectome_pairs_summed_output = []
indices = []
for i in np.arange(0, len(output_projectome.index), 2):
    combined_pairs = (output_projectome.iloc[i, :] + output_projectome.iloc[i+1, :])

    combined_hemisegs = []
    for j in np.arange(0, len(combined_pairs), 2):
        combined_hemisegs.append((combined_pairs[j] + combined_pairs[j+1]))
    
    dVNC_projectome_pairs_summed_output.append(combined_hemisegs)
    indices.append(output_projectome.index[i])

dVNC_projectome_pairs_summed_output = pd.DataFrame(dVNC_projectome_pairs_summed_output, index = indices, columns = ['brain','SEZ', 'T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'])
#dVNC_projectome_pairs_summed_output = dVNC_projectome_pairs_summed_output.iloc[:, 1:len(dVNC_projectome_pairs_summed_output)]

#normalize # of presynaptic sites
dVNC_projectome_pairs_summed_output_norm = dVNC_projectome_pairs_summed_output.copy()
for i in range(len(dVNC_projectome_pairs_summed_output)):
    sum_row = sum(dVNC_projectome_pairs_summed_output_norm.iloc[i, :])
    for j in range(len(dVNC_projectome_pairs_summed_output.columns)):
        dVNC_projectome_pairs_summed_output_norm.iloc[i, j] = dVNC_projectome_pairs_summed_output_norm.iloc[i, j]/sum_row

# remove brain from columns
dVNC_projectome_pairs_summed_output_norm_no_brain = dVNC_projectome_pairs_summed_output_norm.iloc[:, 1:len(dVNC_projectome_pairs_summed_output)]
dVNC_projectome_pairs_summed_output_no_brain = dVNC_projectome_pairs_summed_output.iloc[:, 1:len(dVNC_projectome_pairs_summed_output)]

# %%
# ordering and plotting

# sorting with normalized data
sort_threshold = 0
dVNC_projectome_pairs_summed_output_sort_norm = dVNC_projectome_pairs_summed_output_norm_no_brain.copy()
dVNC_projectome_pairs_summed_output_sort_norm[dVNC_projectome_pairs_summed_output_sort_norm<sort_threshold]=0
order = ['SEZ', 'T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
order.reverse()
dVNC_projectome_pairs_summed_output_sort_norm.sort_values(by=order, ascending=False, inplace=True)
sort = dVNC_projectome_pairs_summed_output_sort_norm.index

cmap = plt.cm.get_cmap('Blues') # modify 'Blues' cmap to have a white background
blue_cmap = cmap(np.linspace(0, 1, 20))
blue_cmap[0] = np.array([1, 1, 1, 1])
blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Blues', colors=blue_cmap)

cmap = blue_cmap
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.heatmap(dVNC_projectome_pairs_summed_output_norm.loc[sort, :], ax=ax, cmap=cmap)
plt.savefig(f'VNC_interaction/plots/projectome/A8-T1_sort_projectome_normalized_sortThres{sort_threshold}.pdf', bbox_inches='tight')

# sorting with raw data
sort_threshold = 0
dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output_no_brain.copy()
dVNC_projectome_pairs_summed_output_sort[dVNC_projectome_pairs_summed_output_sort<sort_threshold]=0
order = ['SEZ', 'T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
order.reverse()
dVNC_projectome_pairs_summed_output_sort.sort_values(by=order, ascending=False, inplace=True)
sort = dVNC_projectome_pairs_summed_output_sort.index

vmax = 70
cmap = blue_cmap
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.heatmap(dVNC_projectome_pairs_summed_output.loc[sort, :], ax=ax, cmap=cmap, vmax=vmax)
plt.savefig(f'VNC_interaction/plots/projectome/A8-T1_sort_projectome_sortThres{sort_threshold}.pdf', bbox_inches='tight')

# %%
# paths 2-hop upstream of each dVNC
from tqdm import tqdm

# sort dVNC pairs
sort = [int(x) for x in sort]
dVNC_pairs.set_index('leftid', drop=False, inplace=True)
dVNC_pairs = dVNC_pairs.loc[sort, :]
dVNC_pairs.reset_index(inplace=True, drop=True)

hops = 2
threshold = 0.01

dVNC_pair_paths_us = [pm.Promat.upstream_multihop(edges=edges, sources=dVNC_pairs.loc[i].to_list(), hops=hops) for i in tqdm(range(0, len(dVNC_pairs)))]
dVNC_pair_paths_ds = [pm.Promat.downstream_multihop(edges=edges, sources=dVNC_pairs.loc[i].to_list(), hops=hops) for i in tqdm(range(0, len(dVNC_pairs)))]

# %%
# plotting individual dVNC paths

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

skids_list = [list(adj.index)] + [x.get_skids() for x in celltypes]

# UPSTREAM
all_layers_us = [ct.Celltype_Analyzer.layer_id(dVNC_pair_paths_us, dVNC_pairs.leftid, skids_type)[0] for skids_type in skids_list]
layer_names = ['Total'] + [x.get_name() for x in celltypes]

threshold = 0.01

layer_colors = ['Greens', 'Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Blues', 'Reds', 'Purples', 'Reds', 'Purples', 'Reds', 'Purples', 'Reds']
layer_vmax = [200, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 50, 50, 50, 50, 50, 50]
save_path = 'VNC_interaction/plots/dVNC_partners/Upstream_'
ct.Celltype_Analyzer.plot_layer_types(layer_types=all_layers_us, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs.leftid, figsize=(.5*hops/3, 1.5), save_path=save_path, threshold=threshold, hops=hops)

# DOWNSTREAM
all_layers_ds = [ct.Celltype_Analyzer.layer_id(dVNC_pair_paths_ds, dVNC_pairs.leftid, skids_type)[0] for skids_type in skids_list]
layer_names = ['Total'] + [x.get_name() for x in celltypes]
threshold = 0.01

layer_colors = ['Greens', 'Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Blues', 'Reds', 'Purples', 'Reds', 'Purples', 'Reds', 'Purples', 'Reds']
layer_vmax = [200, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 50, 50, 50, 50, 50, 50]
save_path = 'VNC_interaction/plots/dVNC_partners/Downstream-in-brain_'
ct.Celltype_Analyzer.plot_layer_types(layer_types=all_layers_ds, layer_names=layer_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=dVNC_pairs.leftid, figsize=(.5*hops/3, 1.5), save_path=save_path, threshold=threshold, hops=hops)

# %%
# make bar plots for 1-hop and 2-hop

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

figsize = (2,0.5)
# UPSTREAM
us_1order = ct.Celltype_Analyzer([ct.Celltype(str(dVNC_pairs.loc[i].leftid) + '_us_1o', x[0]) for i, x in enumerate(dVNC_pair_paths_us)])
us_2order = ct.Celltype_Analyzer([ct.Celltype(str(dVNC_pairs.loc[i].leftid) + '_us_2o', x[1]) for i, x in enumerate(dVNC_pair_paths_us)])
us_1order.set_known_types(celltypes)
us_2order.set_known_types(celltypes)

path = 'VNC_interaction/plots/dVNC_partners/summary_plot_1st_order_upstream.pdf'
us_1order.plot_memberships(path = path, figsize=figsize)

path = 'VNC_interaction/plots/dVNC_partners/summary_plot_2nd_order_upstream.pdf'
us_2order.plot_memberships(path = path, figsize=figsize)

# DOWNSTREAM
ds_1order = ct.Celltype_Analyzer([ct.Celltype(str(dVNC_pairs.loc[i].leftid) + '_ds_1o', x[0]) for i, x in enumerate(dVNC_pair_paths_ds)])
ds_2order = ct.Celltype_Analyzer([ct.Celltype(str(dVNC_pairs.loc[i].leftid) + '_ds_2o', x[1]) for i, x in enumerate(dVNC_pair_paths_ds)])
ds_1order.set_known_types(celltypes)
ds_2order.set_known_types(celltypes)

path = 'VNC_interaction/plots/dVNC_partners/summary_plot_1st_order_downstream.pdf'
ds_1order.plot_memberships(path = path, figsize=figsize)

path = 'VNC_interaction/plots/dVNC_partners/summary_plot_2nd_order_downstream.pdf'
ds_2order.plot_memberships(path = path, figsize=figsize)


# %%
# combine all data types for dVNCs: us1o, us2o, ds1o, ds2o, projectome

fraction_cell_types_1o_us = pd.DataFrame([x.iloc[:, 0] for x in fraction_types], index = fraction_types_names).T
fraction_cell_types_1o_us.columns = [f'1o_us_{x}' for x in fraction_cell_types_1o_us.columns]
unk_col = 1-fraction_cell_types_1o_us.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_1o_us['1o_us_unk']=unk_col

fraction_cell_types_2o_us = pd.DataFrame([x.iloc[:, 1] for x in fraction_types], index = fraction_types_names).T
fraction_cell_types_2o_us.columns = [f'2o_us_{x}' for x in fraction_cell_types_2o_us.columns]
unk_col = 1-fraction_cell_types_2o_us.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_2o_us['2o_us_unk']=unk_col

fraction_cell_types_1o_ds = pd.DataFrame([x.iloc[:, 0] for x in fraction_types_ds], index = fraction_types_names).T
fraction_cell_types_1o_ds.columns = [f'1o_ds_{x}' for x in fraction_cell_types_1o_ds.columns]
unk_col = 1-fraction_cell_types_1o_ds.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_1o_ds['1o_ds_unk']=unk_col
fraction_cell_types_1o_ds[fraction_cell_types_1o_ds==-1]=0

fraction_cell_types_2o_ds = pd.DataFrame([x.iloc[:, 1] for x in fraction_types_ds], index = fraction_types_names).T
fraction_cell_types_2o_ds.columns = [f'2o_ds_{x}' for x in fraction_cell_types_2o_ds.columns]
unk_col = 1-fraction_cell_types_2o_ds.sum(axis=1)
unk_col[unk_col==11]=0
fraction_cell_types_2o_ds['2o_ds_unk']=unk_col
fraction_cell_types_2o_ds[fraction_cell_types_2o_ds==-1]=0

all_data = dVNC_projectome_pairs_summed_output_norm.copy()
all_data.index = [int(x) for x in all_data.index]

all_data = pd.concat([fraction_cell_types_1o_us, fraction_cell_types_2o_us, all_data, fraction_cell_types_1o_ds, fraction_cell_types_2o_ds], axis=1)
all_data.fillna(0, inplace=True)

# clustered version of all_data combined
cluster = sns.clustermap(all_data, col_cluster = False, figsize=(30,30), rasterized=True)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_all_data.pdf', bbox_inches='tight')
order = cluster.dendrogram_row.reordered_ind
fig,ax=plt.subplots(1,1,figsize=(6,4))
sns.heatmap(all_data.iloc[order, :].drop(list(fraction_cell_types_1o_us.columns) + list(fraction_cell_types_2o_us.columns) + list(fraction_cell_types_1o_ds.columns) + list(fraction_cell_types_2o_ds.columns), axis=1), ax=ax, rasterized=True)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_all_data_same_size.pdf', bbox_inches='tight')

cluster = sns.clustermap(all_data.drop(['1o_us_pre-dVNC', '2o_us_pre-dVNC'], axis=1), col_cluster = False, figsize=(20,15), rasterized=True)
plt.savefig(f'VNC_interaction/plots/projectome/clustered_projectome_all_data_removed_us-pre-dVNCs.pdf', bbox_inches='tight')

# decreasing sort of all_data but with feedback and non-feedback dVNC clustered
for i in range(1, 50):
    dVNCs_with_FB = all_data.loc[:, list(fraction_cell_types_1o_ds.columns) + list(fraction_cell_types_2o_ds.columns)].sum(axis=1)
    dVNCs_FB_true_skids = dVNCs_with_FB[dVNCs_with_FB>0].index
    dVNCs_FB_false_skids = dVNCs_with_FB[dVNCs_with_FB==0].index

    dVNC_projectome_pairs_summed_output_sort = all_data.copy()
    dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output_sort.loc[:, ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']]
    dVNC_projectome_pairs_summed_output_sort = dVNC_projectome_pairs_summed_output_sort.loc[dVNCs_FB_true_skids]
    dVNC_projectome_pairs_summed_output_sort[dVNC_projectome_pairs_summed_output_sort<(i/100)]=0
    dVNC_projectome_pairs_summed_output_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order_FB_true = dVNC_projectome_pairs_summed_output_sort.index

    second_sort = all_data.copy()
    second_sort = second_sort.loc[:, ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']]
    second_sort = second_sort.loc[dVNCs_FB_false_skids]
    second_sort[second_sort<(i/100)]=0
    second_sort.sort_values(by=['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'], ascending=False, inplace=True)
    row_order_FB_false = second_sort.index
    row_order = list(row_order_FB_true) + list(row_order_FB_false)
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(all_data.loc[row_order, :], ax=ax, rasterized=True)
    plt.savefig(f'VNC_interaction/plots/projectome/splitFB_projectome_0.{i}-sort-threshold.pdf', bbox_inches='tight')
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(all_data.loc[row_order, ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax=ax, rasterized=True)
    plt.savefig(f'VNC_interaction/plots/projectome/splitFB_same-size_projectome_0.{i}-sort-threshold.pdf', bbox_inches='tight')
# %%
# what fraction of us and ds neurons are from different cell types per hop?

fraction_cell_types_1o_us = pd.DataFrame([x.iloc[:, 0] for x in fraction_types], index = fraction_types_names)
fraction_cell_types_1o_us = fraction_cell_types_1o_us.fillna(0) # one dVNC with no inputs

fraction_cell_types_2o_us = pd.DataFrame([x.iloc[:, 1] for x in fraction_types], index = fraction_types_names)
fraction_cell_types_2o_us = fraction_cell_types_2o_us.fillna(0) # one dVNC with no inputs

fraction_cell_types_1o_us_scatter = []
for j in range(1, len(fraction_cell_types_1o_us.columns)):
    for i in range(0, len(fraction_cell_types_1o_us.index)):
        fraction_cell_types_1o_us_scatter.append([fraction_cell_types_1o_us.iloc[i, j], fraction_cell_types_1o_us.index[i]]) 

fraction_cell_types_1o_us_scatter = pd.DataFrame(fraction_cell_types_1o_us_scatter, columns = ['fraction', 'cell_type'])

fraction_cell_types_2o_us_scatter = []
for j in range(1, len(fraction_cell_types_2o_us.columns)):
    for i in range(0, len(fraction_cell_types_2o_us.index)):
        fraction_cell_types_2o_us_scatter.append([fraction_cell_types_2o_us.iloc[i, j], fraction_cell_types_2o_us.index[i]]) 

fraction_cell_types_2o_us_scatter = pd.DataFrame(fraction_cell_types_2o_us_scatter, columns = ['fraction', 'cell_type'])

fig, ax = plt.subplots(1, 1, figsize=(1.25,1))
sns.stripplot(x='cell_type', y='fraction', data=fraction_cell_types_1o_us_scatter, ax=ax, size=.5, jitter=0.2)
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-0.05,1))
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-1o.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(1.25,1))
sns.stripplot(x='cell_type', y='fraction', data=fraction_cell_types_2o_us_scatter, ax=ax, size=.5, jitter=0.2)
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-0.05,1))
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-2o.pdf', format='pdf', bbox_inches='tight')

# %%
# number of us and ds neurons are from different cell types per hop?

layer_types_ds_counts = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]

cell_types_1o_us = pd.DataFrame([x.iloc[:, 0] for x in layer_types_ds_counts], index = fraction_types_names)
cell_types_1o_us = cell_types_1o_us.fillna(0) # one dVNC with no inputs

cell_types_2o_us = pd.DataFrame([x.iloc[:, 1] for x in layer_types_ds_counts], index = fraction_types_names)
cell_types_2o_us = cell_types_2o_us.fillna(0) # one dVNC with no inputs

cell_types_1o_us_scatter = []
for j in range(1, len(cell_types_1o_us.columns)):
    for i in range(0, len(cell_types_1o_us.index)):
        cell_types_1o_us_scatter.append([cell_types_1o_us.iloc[i, j], cell_types_1o_us.index[i]]) 

cell_types_1o_us_scatter = pd.DataFrame(cell_types_1o_us_scatter, columns = ['counts', 'cell_type'])

cell_types_2o_us_scatter = []
for j in range(1, len(cell_types_2o_us.columns)):
    for i in range(0, len(cell_types_2o_us.index)):
        cell_types_2o_us_scatter.append([cell_types_2o_us.iloc[i, j], cell_types_2o_us.index[i]]) 

cell_types_2o_us_scatter = pd.DataFrame(cell_types_2o_us_scatter, columns = ['counts', 'cell_type'])

fig, ax = plt.subplots(1, 1, figsize=(1.25,1.25))
sns.stripplot(x='cell_type', y='counts', hue='cell_type', data=cell_types_1o_us_scatter, palette = colors, ax=ax, size=.5, jitter=0.25, alpha=0.9, edgecolor="none")
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-2, 80), xlabel='')
ax.get_legend().remove()

median_width = 0.6

for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"

    # calculate the median value for all replicates of either X or Y
    median_val = cell_types_1o_us_scatter[cell_types_1o_us_scatter['cell_type']==sample_name].counts.median()

    # plot horizontal lines across the column, centered on the tick
    ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val], lw=0.25, color='k', zorder=100)

plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-1o_counts.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(1.25,1.25))
sns.stripplot(x='cell_type', y='counts', hue='cell_type', data=cell_types_2o_us_scatter, palette = colors, ax=ax, size=.5, jitter=0.25, alpha=0.9, edgecolor="none")
plt.xticks(rotation=45, ha='right')
ax.set(ylabel='', ylim=(-2, 80), xlabel='')
ax.get_legend().remove()

median_width = 0.6

for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"

    # calculate the median value for all replicates of either X or Y
    median_val = cell_types_2o_us_scatter[cell_types_2o_us_scatter['cell_type']==sample_name].counts.median()

    # plot horizontal lines across the column, centered on the tick
    ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val], lw=0.25, color='k', zorder=100)

plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_fraction-cell-types_us-2o_counts.pdf', format='pdf', bbox_inches='tight')

# %%
# how many dVNCs have different cell types per hop?

cell_types = [PN_type_layers, LHN_type_layers, MBIN_type_layers, MBON_type_layers, 
                FBN_type_layers, CN_type_layers, asc_type_layers, dSEZ_type_layers, predVNC_type_layers, dVNC_type_layers]
cell_types_names = ['PN', 'LHN', 'MBIN', 'MBON', 'MB-FBN', 'CN', 'A1-asc', 'dSEZ', 'pre-dVNC', 'dVNC']

counts = []
for i in range(0, len(cell_types)):
    counts.append([f'{cell_types_names[i]}' , sum(cell_types[i].iloc[:, 0]>0), sum(cell_types[i].iloc[:, 1]>0)])

counts = pd.DataFrame(counts, columns = ['cell_type', 'number_1o', 'number_2o']).set_index('cell_type')

plt.rcParams['font.size'] = 5
x = np.arange(len(counts))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(2,1.5))
order1 = ax.bar(x - width/2, counts.number_1o, width, label='Directly Upstream')
order2 = ax.bar(x + width/2, counts.number_2o, width, label='2-Hop Upstream')

ax.set_ylabel('Number of dVNCs')
ax.set_xticks(x)
ax.set_xticklabels(counts.index)
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.savefig('VNC_interaction/plots/dVNC_upstream/summary_plot_1o_2o_counts.pdf', format='pdf', bbox_inches='tight')

# %%
# histogram of number of cell types per dVNC at 1o and 2o upstream and downstream
# work in progress
from matplotlib import pyplot as py
'''
fig, ax = plt.subplots(figsize=(3,3))
bins_2o = np.arange(max(cell_types_2o_us.T.PN)/2) - 0.5
sns.distplot(cell_types_2o_us.T.PN/2, bins=bins_2o, kde=False, ax=ax, color='blue')
sns.distplot(cell_types_1o_us.T.PN/2, bins=bins_2o, kde=False, ax=ax, color='blue')
ax.set(xlim=(-1, max(cell_types_2o_us.T.PN)/2), xticks=(range(0, 20, 1)))
'''

fig, axs = plt.subplots(2, 2, figsize=(4,4))

data_1o = cell_types_1o_us.T.PN/2
data_2o = cell_types_2o_us.T.PN/2
max_value = int(max(list(data_2o) + list(data_1o)))
bins_2o = np.arange(max_value+2) - 0.5

ax = axs[0,0] 
ax.hist(data_1o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))

ax = axs[0,1] 
ax.hist(data_2o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))


data_1o = cell_types_1o_us.T.LHN/2
data_2o = cell_types_2o_us.T.LHN/2
max_value = int(max(list(data_2o) + list(data_1o)))
bins_2o = np.arange(max_value+2) - 0.5

ax = axs[1,0] 
ax.hist(data_1o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))

ax = axs[1,1] 
ax.hist(data_2o, bins=bins_2o)
ax.set(xlim=(-1, max_value+1), xticks=(range(0, max_value, 1)))

# %%
# 3D Plot

ct_1o_us = cell_types_1o_us.T/2
ct_2o_us = cell_types_2o_us.T/2

data_1o_3d_list = []
for i in range(0, len(ct_1o_us.columns)):
    data_1o = ct_1o_us.iloc[:, i]
    data_1o_3d = [[i, sum(data_1o==i)] for i in range(int(max(data_1o+1)))]
    data_1o_3d = pd.DataFrame(data_1o_3d, columns=['number', 'height'])
    data_1o_3d_list.append(data_1o_3d)

data_2o_3d_list = []
for i in range(0, len(ct_2o_us.columns)):
    data_2o = ct_2o_us.iloc[:, i]
    data_2o_3d = [[i, sum(data_2o==i)] for i in range(int(max(data_2o+1)))]
    data_2o_3d = pd.DataFrame(data_2o_3d, columns=['number', 'height'])
    data_2o_3d_list.append(data_2o_3d)

fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
for zs_values, i in zip([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], range(len(data_1o_3d_list))):
    ax.bar(data_1o_3d_list[i].number, data_1o_3d_list[i].height, zs = zs_values, zdir='y')

fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
for zs_values, i in zip([0,1,3,4,6,7,9,10,12,13], range(len(data_2o_3d_list))):
    ax.bar(data_2o_3d_list[i].number, data_2o_3d_list[i].height, zs = zs_values, zdir='y')

# %%
# violinplot
cell_types_1o_us_scatter['order']=['1st-order']*len(cell_types_1o_us_scatter)
cell_types_2o_us_scatter['order']=['2nd-order']*len(cell_types_2o_us_scatter)

celltypes_1o_2o_us = cell_types_1o_us_scatter.append(cell_types_2o_us_scatter)

fig, axs = plt.subplots(figsize=(10,5))
sns.boxenplot(y='counts', x='cell_type', hue='order', data =celltypes_1o_2o_us, ax=axs, outlier_prop=0)
#sns.boxenplot(y='counts', x='cell_type', data=celltypes_1o_2o_us, ax=axs)
plt.savefig('VNC_interaction/plots/dVNC_upstream/boxenplot.pdf', bbox_inches='tight', transparent = True)

# %%
# Ridgeline plot
import joypy
#fig, axes = joypy.joyplot(cell_types_2o_us_scatter, by="cell_type", overlap=4) #, hist=True, bins=int(max(cell_types_2o_us_scatter.counts)))
joypy.joyplot(fraction_cell_types_1o_us_scatter, by="cell_type", overlap=4)
#fig, axes = joypy.joyplot(cell_types_1o_us_scatter, by="cell_type")

# %%
# multi-hop matrix of all cell types to dVNCs
# incomplete

threshold = 0.01
all_type_layers,all_type_layers_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, br)
dVNC_type_layers,dVNC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, dVNC)
predVNC_type_layers,predVNC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, pre_dVNC)
dSEZ_type_layers,dSEZ_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, dSEZ)
LHN_type_layers,LHN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, LHN)
CN_type_layers,CN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, CN)
MBON_type_layers,MBON_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, MBON)
MBIN_type_layers,MBIN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, MBIN)
FBN_type_layers,FBN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, FBN_all)
KC_type_layers,KC_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, KC)
PN_type_layers,PN_type_skids = br_adj.layer_id(dVNC_pair_paths, dVNC_pairs.leftid, PN)

#sns.barplot(x = [1, 2], y = [sum(MBON_type_layers.iloc[:, 0]>0)/len(MBON_type_layers.iloc[:, 0]), sum(MBON_type_layers.iloc[:, 1]>0)/len(MBON_type_layers.iloc[:, 1])])
MBON_dVNC, MBON_dVNC_plotting = br_adj.hop_matrix(MBON_type_skids.T, dVNC_pairs.leftid, Promat.extract_pairs_from_list(MBON, pairs)[0].leftid)

# %%
# specific interactions between dVNCs with phenotypes and a couple selected dVNCs

from connectome_tools.cascade_analysis import Celltype, Celltype_Analyzer

dVNC_important = [[17353986, np.where(dVNC_pairs.leftid==17353986)[0][0], 'backup', 'unpublished', 'dVNC'],
                    [10728328, np.where(dVNC_pairs.leftid==10728328)[0][0], 'backup', 'published', 'dVNC'],
                    [10728333, np.where(dVNC_pairs.leftid==10728333)[0][0], 'backup', 'published', 'dVNC'],
                    [6446394, np.where(dVNC_pairs.leftid==6446394)[0][0], 'stop', 'published', 'dVNC'],
                    [10382686, np.where(dVNC_pairs.leftid==10382686)[0][0], 'stop', 'unpublished', 'dSEZ'],
                    [16851496, np.where(dVNC_pairs.leftid==16851496)[0][0], 'cast', 'unpublished', 'dVNC'],
                    [10553248, np.where(dVNC_pairs.leftid==10553248)[0][0], 'cast', 'unpublished', 'dVNC'],
                    [3044500, np.where(dVNC_pairs.leftid==3044500)[0][0], 'cast_onset_offset', 'unpublished', 'dSEZ'],
                    [3946166, np.where(dVNC_pairs.leftid==3946166)[0][0], 'cast_onset_offset', 'unpublished', 'dVNC']]

dVNC_important = pd.DataFrame(dVNC_important, columns=['leftid', 'index', 'behavior', 'status', 'celltype'])
#dVNC_exclusive = dVNC_important.loc[dVNC_important.celltype=='dVNC']
#dVNC_exclusive.reset_index(inplace=True, drop=True)
#dVNC_important_us = all_type_layers_skids.loc[:, dVNC_exclusive.leftid]

#dVNC_important_us.iloc[0, :]

# check overlap between us networks
us_cts = []
for i in range(len(dVNC_important_us.index)):
    for j in range(len(dVNC_important_us.columns)):
        cts = Celltype(f'{dVNC_exclusive.behavior[j]} {dVNC_important_us.columns[j]} {i+1}-order', dVNC_important_us.iloc[i, j])
        us_cts.append(cts)

cta = Celltype_Analyzer(us_cts)
sns.heatmap(cta.compare_membership(), annot=True, fmt='.0%')

# number of neurons in us networks
us_1order = [x for sublist in dVNC_important_us.loc[0] for x in sublist]
us_2order = [x for sublist in dVNC_important_us.loc[1] for x in sublist]

us_1order_unique = np.unique(us_1order)
us_2order_unique = np.unique(us_2order)

pymaid.add_annotations(dVNC_important.leftid.values, 'mw dVNC important')
pymaid.add_annotations([pairs[pairs.leftid==x].rightid.values[0] for x in dVNC_important.leftid.values], 'mw dVNC important')
pymaid.add_annotations(us_1order_unique, 'mw dVNC important 1st-order upstream')

for i, us in enumerate(dVNC_important_us.loc[0]):
    pymaid.add_annotations(us, f'mw dVNC upstream-1o {dVNC_important_us.columns[i]}')
    pymaid.add_meta_annotations(f'mw dVNC upstream-1o {dVNC_important_us.columns[i]}', 'mw dVNC upstream-1o')
# %%
