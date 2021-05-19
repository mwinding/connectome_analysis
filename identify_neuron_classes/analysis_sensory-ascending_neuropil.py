#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load neuron types based on annotations

# warning: hardcoded names to get order correct
# could be problem if more are added
order = ['olfactory', 'gustatory-external', 'gustatory-internal', 'enteric', 'thermo', 'visual', 'noci', 'mechano', 'proprio', 'touch', 'intero']
order2_names = [f'mw brain 2nd_order {celltype}' for celltype in order]
order3_names = [f'mw brain 3rd_order {celltype}' for celltype in order]

order2 = [pymaid.get_skids_by_annotation(annot) for annot in order2_names]
order3 = [pymaid.get_skids_by_annotation(annot) for annot in order3_names]

order2_ct = ct.Celltype_Analyzer([ct.Celltype(order2_names[i].replace('mw brain ', ''), skids) for i, skids in enumerate(order2)])
order3_ct = ct.Celltype_Analyzer([ct.Celltype(order3_names[i].replace('mw brain ', ''), skids) for i, skids in enumerate(order3)])

# upset plots of 2nd/3rd order centers
#   returned values are Celltypes of upset plot partitions 
order2_cats = order2_ct.upset_members(path='identify_neuron_classes/plots/2nd-order_upset-plot', plot_upset=True, threshold=10, exclude_singletons_from_threshold=True)
order3_cats = order3_ct.upset_members(path='identify_neuron_classes/plots/3rd-order_upset-plot', plot_upset=True, threshold=10, exclude_singletons_from_threshold=True)

# %%
# manually generate Sankey-like plots
# use numbers extracted here for sizes of each bar
order = ['olfactory', 'gustatory-external', 'gustatory-internal', 'enteric', 'thermo', 'visual', 'noci', 'mechano', 'proprio', 'touch', 'intero']
sens = [ct.Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
order2 = [ct.Celltype(f'2nd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 2nd_order {name}')) for name in order]
order3 = [ct.Celltype(f'3rd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 3rd_order {name}')) for name in order]

LNs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
RGN = pymaid.get_skids_by_annotation('mw RGN')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')

columns = []
for i in range(0, len(sens)):
    sens_len = len(sens[i].get_skids())

    order2_LN = len(np.intersect1d(order2[i].get_skids(), LNs))
    order2_RGN = len(np.intersect1d(order2[i].get_skids(), RGN))
    order2_dSEZ = len(np.intersect1d(order2[i].get_skids(), dSEZ))
    order2_dVNC = len(np.intersect1d(order2[i].get_skids(), dVNC))
    order2_other = len(np.setdiff1d(order2[i].get_skids(), LNs + RGN + dSEZ + dVNC))

    order3_LN = len(np.intersect1d(order3[i].get_skids(), LNs))
    order3_RGN = len(np.intersect1d(order3[i].get_skids(), RGN))
    order3_dSEZ = len(np.intersect1d(order3[i].get_skids(), dSEZ))
    order3_dVNC = len(np.intersect1d(order3[i].get_skids(), dVNC))
    order3_other = len(np.setdiff1d(order3[i].get_skids(), LNs + RGN + dSEZ + dVNC))

    print(f'{order[i]}:')
    print(f'sensory count: {sens_len}')
    print(f'2nd_order count: LNs - {order2_LN}, other - {order2_other}, RGN - {order2_RGN}, dSEZ - {order2_dSEZ}, dVNC - {order2_dVNC}')
    print(f'3rd_order count: LNs - {order3_LN}, other - {order3_other}, RGN - {order3_RGN}, dSEZ - {order3_dSEZ}, dVNC - {order3_dVNC}')
    print('')

    columns.append([sens_len, 0, 0, 0, 0, 0])
    columns.append([0, order2_LN, order2_other, order2_RGN, order2_dSEZ, order2_dVNC])
    columns.append([0, order3_LN, order3_other, order3_RGN, order3_dSEZ, order3_dVNC])

df = pd.DataFrame(columns, columns = ['Sens', 'LN', 'Other', 'RGN', 'dSEZ', 'dVNC'])

spacer = 10
width = 0.4
fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['Sens'], width = width)
ax.bar(x = df.index, height = df['LN'], width = width)
ax.bar(x = df.index, height = df['Other'], bottom = df['LN']+ spacer, width = width)
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN'] + df['Other']+ spacer*2, width = width)
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN'] + df['Other'] + df['RGN']+ spacer*3, width = width)
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['dSEZ']+ spacer*4, width = width)
ax.axis('off')

plt.savefig('identify_neuron_classes/plots/source_for_sankey_plot.pdf', bbox_inches='tight', format = 'pdf')

# plot fraction of each type
row_sum = df.sum(axis=1).values

for i in range(0, len(df)):
    df.iloc[i, :] = df.iloc[i, :]/row_sum[i]

df=df.fillna(0) # fill NaN from divide by 0 with 0

fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['Sens'])
ax.bar(x = df.index, height = df['LN'])
ax.bar(x = df.index, height = df['Other'], bottom = df['LN'])
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN'] + df['Other'])
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN'] + df['Other'] + df['RGN'])
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['dSEZ'])
ax.axis('off')

plt.savefig('identify_neuron_classes/plots/fraction-LNs-and-outputs_per_neuropil.pdf', bbox_inches='tight', format = 'pdf')

# %%
# known cell types per 2nd/3rd order
order = ['olfactory', 'gustatory-external', 'gustatory-internal', 'enteric', 'thermo', 'visual', 'noci', 'mechano', 'proprio', 'touch', 'intero']
sens = [ct.Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
order2 = [ct.Celltype(f'2nd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 2nd_order {name}')) for name in order]
order3 = [ct.Celltype(f'3rd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 3rd_order {name}')) for name in order]

sens_cta = ct.Celltype_Analyzer(sens)
order2_cta = ct.Celltype_Analyzer(order2)
order3_cta = ct.Celltype_Analyzer(order3)

celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes() # will have to add a way of removing particular groups from this list in the future

# intercalated 2nd/3rd order identities
intercalated_order2_3 = [x for sublist in list(zip(order2, order3)) for x in sublist]
intercalated_cta = ct.Celltype_Analyzer(intercalated_order2_3)
intercalated_cta.set_known_types(celltypes)
memberships = intercalated_cta.memberships(raw_num=True).drop('sensories').T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

sns.heatmap(memberships, annot=True, fmt='d', cmap='Greens', cbar=False)

# plot 2nd order identities
order2_cta.set_known_types(celltypes)
memberships = order2_cta.memberships(raw_num=True).drop(['sensories', 'ascendings']).T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(2.35,1.25))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identities_2nd_order.pdf', bbox_inches='tight', format = 'pdf')

# plot 3rd order identities
order3_cta.set_known_types(celltypes)
memberships = order3_cta.memberships(raw_num=True).drop(['sensories', 'ascendings']).T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(2.35,1.25))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identites_3rd-order.pdf', bbox_inches='tight', format = 'pdf')

# %%
# adjacency matrix of all types

brain_inputs = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities')
brain = pymaid.get_skids_by_annotation('mw brain neurons') + brain_inputs

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad, adj_aa, adj_dd, adj_da = [pd.read_csv(f'data/adj/all-neurons_{name}.csv', index_col = 0).rename(columns=int) for name in adj_names]
adj_ad = adj_ad.loc[np.intersect1d(adj_ad.index, brain), np.intersect1d(adj_ad.index, brain)]
adj_aa = adj_aa.loc[np.intersect1d(adj_aa.index, brain), np.intersect1d(adj_aa.index, brain)]
adj_dd = adj_dd.loc[np.intersect1d(adj_dd.index, brain), np.intersect1d(adj_dd.index, brain)]
adj_da = adj_da.loc[np.intersect1d(adj_da.index, brain), np.intersect1d(adj_da.index, brain)]
adjs = [adj_ad, adj_aa, adj_dd, adj_da]

vmaxs = [75, 30, 30, 10]
for i, adj in enumerate(adjs):
    neuron_types_cta = ct.Celltype_Analyzer(order2 + order3)
    summed_mat = neuron_types_cta.connectivity(adj=adj, normalize_post_num=True)
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.heatmap(summed_mat, square=True, vmax=vmaxs[i])
    plt.savefig(f'identify_neuron_classes/plots/connectivity-between-neuropils_{adj_names[i]}.pdf', bbox_inches='tight', format = 'pdf')

# %%
# cascades through sensory neuropils