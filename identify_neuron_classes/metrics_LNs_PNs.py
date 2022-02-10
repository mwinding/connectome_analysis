#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import connectome_tools.process_skeletons as skel
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
adj_mat = pm.Adjacency_matrix(adj, inputs, 'ad')
pairs = pm.Promat.get_pairs()

# %%
# load previously generated adjacency matrices
# see 'network_analysis/generate_all_edges.py'

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')
adj_aa = pm.Promat.pull_adj(type_adj='aa', subgraph='brain')
adj_dd = pm.Promat.pull_adj(type_adj='dd', subgraph='brain')
adj_da = pm.Promat.pull_adj(type_adj='da', subgraph='brain')

# load input counts
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
outputs = pd.read_csv('data/graphs/outputs.csv', index_col=0)

# %%
# LN metrics

# old stuff

# identify LNs
# use 0.5 output fraction within group threshold
threshold = 0.5
LNs_2nd = [celltype.identify_LNs(threshold, summed_adj, adj_aa, sens[i], outputs, exclude=exclude)[1] for i, celltype in enumerate(order2_ct)]
LNs_3rd = [celltype.identify_LNs(threshold, summed_adj, adj_aa, order2_ct[i].get_skids(), outputs, exclude=exclude)[1] for i, celltype in enumerate(order3_ct)]
LNs_4th = [celltype.identify_LNs(threshold, summed_adj, adj_aa, order3_ct[i].get_skids(), outputs, exclude=exclude)[1] for i, celltype in enumerate(order4_ct)]


# %%
# LN metrics

summed_adj = pd.read_csv(f'data/adj/all-neurons_all-all.csv', index_col = 0).rename(columns=int)
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
brain_inputs = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities', split=False)
brain_inputs = brain_inputs + pymaid.get_skids_by_annotation('mw A1 ascending unknown')

sens = [ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {celltype}') for celltype in order]
order2_ct = [ct.Celltype(f'2nd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 2nd_order')) for celltype in order]
order3_ct = [ct.Celltype(f'3rd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 3rd_order')) for celltype in order]
order4_ct = [ct.Celltype(f'4th-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 4th_order')) for celltype in order]
exclude = [pymaid.get_skids_by_annotation(x) for x in ['mw MBON', 'mw MBIN', 'mw RGN', 'mw dVNC', 'mw dSEZ', 'mw KC']]
exclude = [x for sublist in exclude for x in sublist]
exclude = exclude + brain_inputs

LNs_2nd = [celltype.identify_LNs(threshold, summed_adj, adj_aa, sens[i], outputs, exclude=exclude)[1] for i, celltype in enumerate(order2_ct)]
LNs_2nd_morpho = [skel.axon_dendrite_centroids_pairwise(celltype) for celltype in order2_ct]
LNs_2nd_metrics = [pd.concat([LNs_2nd[i], LNs_2nd_morpho[i]], axis=1) for i in range(0, len(LNs_2nd))]

LNs_3rd = [celltype.identify_LNs(threshold, summed_adj, adj_aa, sens[i], outputs, exclude=exclude)[1] for i, celltype in enumerate(order3_ct)]
LNs_3rd_morpho = [skel.axon_dendrite_centroids_pairwise(celltype) for celltype in order3_ct]
LNs_3rd_metrics = [pd.concat([LNs_3rd[i], LNs_3rd_morpho[i]], axis=1) for i in range(0, len(LNs_3rd))]

LNs_4th = [celltype.identify_LNs(threshold, summed_adj, adj_aa, sens[i], outputs, exclude=exclude)[1] for i, celltype in enumerate(order4_ct)]
LNs_4th_morpho = [skel.axon_dendrite_centroids_pairwise(celltype) for celltype in order4_ct]
LNs_4th_metrics = [pd.concat([LNs_4th[i], LNs_4th_morpho[i]], axis=1) for i in range(0, len(LNs_4th))]

# %%
# plot metrics

# remove all excluded neuron types
brain_inputs = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities', split=False)
brain_inputs = brain_inputs + pymaid.get_skids_by_annotation('mw A1 ascending unknown')
exclude = [pymaid.get_skids_by_annotation(x) for x in ['mw MBON', 'mw MBIN', 'mw RGN', 'mw dVNC', 'mw dSEZ', 'mw KC']]
exclude = [x for sublist in exclude for x in sublist]
exclude = exclude + brain_inputs

LNs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')

LNs_2nd_metrics_filtered = []
for metrics in LNs_2nd_metrics:
    skids = [x[2] for x in metrics.index]
    df = metrics.loc[(slice(None), slice(None), np.setdiff1d(skids, exclude)), :]
    skids = [x[2] for x in df.index]

    LN_status = []
    for skid in skids:
        if(skid in LNs): LN_status.append('LN')
        if(skid not in LNs): LN_status.append('non-LN')

    df['is_LN'] = LN_status
    LNs_2nd_metrics_filtered.append(df)

LNs_3rd_metrics_filtered = []
for metrics in LNs_3rd_metrics:
    skids = [x[2] for x in metrics.index]
    df = metrics.loc[(slice(None), slice(None), np.setdiff1d(skids, exclude)), :]
    skids = [x[2] for x in df.index]

    LN_status = []
    for skid in skids:
        if(skid in LNs): LN_status.append('LN')
        if(skid not in LNs): LN_status.append('non-LN')

    df['is_LN'] = LN_status
    LNs_3rd_metrics_filtered.append(df)

LNs_4th_metrics_filtered = []
for metrics in LNs_4th_metrics:
    skids = [x[2] for x in metrics.index]
    df = metrics.loc[(slice(None), slice(None), np.setdiff1d(skids, exclude)), :]
    skids = [x[2] for x in df.index]

    LN_status = []
    for skid in skids:
        if(skid in LNs): LN_status.append('LN')
        if(skid not in LNs): LN_status.append('non-LN')

    df['is_LN'] = LN_status
    LNs_4th_metrics_filtered.append(df)

# font settings
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'arial'

# 2nd_order
threshold=0.5
layer = '2nd-order'
n_rows = 4
n_cols = 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows*2,n_cols*3))
fig.tight_layout(pad=6)
for i, LN_metrics in enumerate(LNs_2nd_metrics_filtered):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = axs[inds]
    #sns.scatterplot(data=LN_metrics, x='percent_output_intragroup', y='distance', hue='is_LN', ax=ax, s=10)
    sns.scatterplot(data=LN_metrics, x='percent_output_intragroup', y='distance', ax=ax, s=10)
    ax.set(xlim=(-0.1, 1.1), ylim=(-5, 150), title=f'{order[i]} {layer}')
#plt.savefig(f'identify_neuron_classes/plots/LN-metrics_{layer}_LN-colored.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'identify_neuron_classes/plots/LN-metrics_{layer}.pdf', format='pdf', bbox_inches='tight')

# 3rd_order
threshold=0.5
layer = '3rd-order'
n_rows = 4
n_cols = 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows*2,n_cols*3))
fig.tight_layout(pad=6)
for i, LN_metrics in enumerate(LNs_3rd_metrics_filtered):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = axs[inds]
    #sns.scatterplot(data=LN_metrics, x='percent_output_intragroup', y='distance', hue='is_LN', ax=ax, s=10)
    sns.scatterplot(data=LN_metrics, x='percent_output_intragroup', y='distance', ax=ax, s=10)
    ax.set(xlim=(-0.1, 1.1), ylim=(-5, 150), title=f'{order[i]} {layer}')
#plt.savefig(f'identify_neuron_classes/plots/LN-metrics_{layer}_LN-colored.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'identify_neuron_classes/plots/LN-metrics_{layer}.pdf', format='pdf', bbox_inches='tight')

# 4th_order
threshold=0.5
layer = '4th-order'
n_rows = 4
n_cols = 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows*2,n_cols*3))
fig.tight_layout(pad=6)
for i, LN_metrics in enumerate(LNs_4th_metrics_filtered):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = axs[inds]
    #sns.scatterplot(data=LN_metrics, x='percent_output_intragroup', y='distance', hue='is_LN', ax=ax, s=10)
    sns.scatterplot(data=LN_metrics, x='percent_output_intragroup', y='distance', ax=ax, s=10)
    ax.set(xlim=(-0.1, 1.1), ylim=(-5, 150), title=f'{order[i]} {layer}')
#plt.savefig(f'identify_neuron_classes/plots/LN-metrics_{layer}_LN-colored.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'identify_neuron_classes/plots/LN-metrics_{layer}.pdf', format='pdf', bbox_inches='tight')

# %%
# export data
import pickle 

#pickle.dump(LNs_2nd_metrics, open('identify_neuron_classes/plots/LNs_2nd_metrics.p', 'wb'))
#pickle.dump(LNs_3rd_metrics, open('identify_neuron_classes/plots/LNs_3rd_metrics.p', 'wb'))
#pickle.dump(LNs_4th_metrics, open('identify_neuron_classes/plots/LNs_4th_metrics.p', 'wb'))

LNs_2nd_metrics = pickle.load(open('identify_neuron_classes/plots/LNs_2nd_metrics.p', 'rb'))
LNs_3rd_metrics = pickle.load(open('identify_neuron_classes/plots/LNs_3rd_metrics.p', 'rb'))
LNs_4th_metrics = pickle.load(open('identify_neuron_classes/plots/LNs_4th_metrics.p', 'rb'))

# %%
# violinplot of axon-dendrite distances

LNs = ct.Celltype('LN', ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs'))
PNs = ct.Celltype('PN', ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain PNs'))
PNs_somato = ct.Celltype('PN-somato', ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain PNs-somato'))
brain = pymaid.get_skids_by_annotation('mw brain neurons')
other_interneurons = list(np.setdiff1d(brain, np.unique(LNs.skids+PNs.skids+PNs_somato.skids)))
other_interneurons = ct.Celltype('other_interneuron', other_interneurons)

LNs_morpho = skel.axon_dendrite_centroids_pairwise(LNs)
PNs_morpho = skel.axon_dendrite_centroids_pairwise(PNs)
PNs_somato_morpho = skel.axon_dendrite_centroids_pairwise(PNs_somato)
other_morpho = skel.axon_dendrite_centroids_pairwise(other_interneurons)

df = pd.concat([LNs_morpho, PNs_morpho, PNs_somato_morpho, other_morpho])

fig, ax = plt.subplots(1,1,figsize=(4,3))
sns.violinplot(data=df, x='celltype', y='distance', orient='vertical', ax=ax, scale='width')
ax.set(ylim=(0,80))
plt.savefig(f'identify_neuron_classes/plots/PN-LN_axon-dendrite-distance.pdf', format='pdf', bbox_inches='tight')

# %%
