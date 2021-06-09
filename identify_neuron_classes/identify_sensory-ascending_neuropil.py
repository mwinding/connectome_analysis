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
# load previously generated adjacency matrices
# see 'network_analysis/generate_all_edges.py'

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')
adj_aa = pm.Promat.pull_adj(type_adj='aa', subgraph='brain')
adj_dd = pm.Promat.pull_adj(type_adj='dd', subgraph='brain')
adj_da = pm.Promat.pull_adj(type_adj='da', subgraph='brain')
adjs = [adj_ad, adj_aa, adj_dd, adj_da]

# load input counts
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
outputs = pd.read_csv('data/graphs/outputs.csv', index_col=0)

# process to produce %input pairwise matrix
mat_ad, mat_aa, mat_dd, mat_da = [pm.Adjacency_matrix(adj=adj, input_counts=inputs, mat_type=adj_names[i]) for i, adj in enumerate(adjs)]
# %%
# load appropriate sensory and ascending types
modalities = 'mw brain sensory modalities'
input_types, input_names = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation(modalities, split=True)
input_names = [annot.replace('mw ', '') for annot in input_names]
input_types = pd.DataFrame(zip(input_names, input_types), columns = ['type', 'source'])

# %%
# identify all 2nd/3rd/4th-order neurons

modalities = 'mw brain sensory modalities'
brain_inputs = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation(modalities, split=False)
brain_inputs = brain_inputs + pymaid.get_skids_by_annotation('mw A1 ascending unknown')
brain = pymaid.get_skids_by_annotation('mw brain neurons') + brain_inputs

pdiff = pymaid.get_skids_by_annotation('mw partially differentiated')

threshold = 0.01
order2 = [mat_ad.downstream_multihop(source=skids, threshold=threshold, hops=1, exclude=brain_inputs+pdiff)[0] for skids in input_types.source]
input_types['order2'] = order2

all_order2 = list(np.unique([x for sublist in input_types.order2 for x in sublist]))
order3 = [mat_ad.downstream_multihop(source=skids, threshold=threshold, hops=1, exclude=brain_inputs+pdiff+all_order2)[0] for skids in input_types.order2]
input_types['order3'] = order3

all_order3 = list(np.unique([x for sublist in input_types.order3 for x in sublist]))
order4 = [mat_ad.downstream_multihop(source=skids, threshold=threshold, hops=1, exclude=brain_inputs+pdiff+all_order3+all_order2)[0] for skids in input_types.order3]
input_types['order4'] = order4

all_order4 = list(np.unique([x for sublist in input_types.order4 for x in sublist]))
order5 = [mat_ad.downstream_multihop(source=skids, threshold=threshold, hops=1, exclude=brain_inputs+pdiff+all_order4+all_order3+all_order2)[0] for skids in input_types.order4]
input_types['order5'] = order5

all_order5 = list(np.unique([x for sublist in input_types.order5 for x in sublist]))
order6 = [mat_ad.downstream_multihop(source=skids, threshold=threshold, hops=1, exclude=brain_inputs+pdiff+all_order5+all_order4+all_order3+all_order2)[0] for skids in input_types.order5]
input_types['order6'] = order6

all_order6 = list(np.unique([x for sublist in input_types.order6 for x in sublist]))

# export IDs
'''
[pymaid.add_annotations(input_types.order2.loc[index], f'mw {input_types.type.loc[index]} 2nd_order') for index in input_types.index]
pymaid.add_meta_annotations([f'mw {input_types.type.loc[index]} 2nd_order' for index in input_types.index], 'mw brain inputs 2nd_order')
[pymaid.add_annotations(input_types.order3.loc[index], f'mw {input_types.type.loc[index]} 3rd_order') for index in input_types.index]
pymaid.add_meta_annotations([f'mw {input_types.type.loc[index]} 3rd_order' for index in input_types.index], 'mw brain inputs 3rd_order')
[pymaid.add_annotations(input_types.order4.loc[index], f'mw {input_types.type.loc[index]} 4th_order') for index in input_types.index]
pymaid.add_meta_annotations([f'mw {input_types.type.loc[index]} 4th_order' for index in input_types.index], 'mw brain inputs 4th_order')
[pymaid.add_annotations(input_types.order5.loc[index], f'mw {input_types.type.loc[index]} 5th_order') for index in input_types.index if len(input_types.order5.loc[index])!=0]
pymaid.add_meta_annotations([f'mw {input_types.type.loc[index]} 5th_order' for index in input_types.index if len(input_types.order5.loc[index])!=0], 'mw brain inputs 5th_order')
'''
input_types = input_types.set_index('type') # for future chunks

# %%
# intersection between 2nd/3rd/4th neuropils
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
#order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'mechano-II/III', 'respiratory','thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'proprio']
sim_type = 'cosine'

# look at overlap between order2 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 2nd-order', input_types.order2.loc[i]) for i in order]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type=sim_type), ax=ax, square=True)
fig.savefig(f'identify_neuron_classes/plots/similarity-{sim_type}_sens_2nd-order.pdf', format='pdf', bbox_inches='tight')

# look at overlap between order3 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 3rd-order', input_types.order3.loc[i]) for i in order]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type=sim_type), ax=ax, square=True)
fig.savefig(f'identify_neuron_classes/plots/similarity-{sim_type}_sens_3rd-order.pdf', format='pdf', bbox_inches='tight')

# look at overlap between order4 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 4th-order', input_types.order4.loc[i]) for i in order]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type=sim_type), ax=ax, square=True)
fig.savefig(f'identify_neuron_classes/plots/similarity-{sim_type}_sens_4th-order.pdf', format='pdf', bbox_inches='tight')

# look at overlap between order5 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 5th-order', input_types.order5.loc[i]) for i in order if len(input_types.order5.loc[i])!=0]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type=sim_type), ax=ax, square=True)
fig.savefig(f'identify_neuron_classes/plots/similarity-{sim_type}_sens_5th-order.pdf', format='pdf', bbox_inches='tight')

# %%
# identify LNs in each layer
pymaid.clear_cache()
summed_adj = pd.read_csv(f'data/adj/all-neurons_all-all.csv', index_col = 0).rename(columns=int)
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']

sens = [ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {celltype}') for celltype in order]
order2_ct = [ct.Celltype(f'2nd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 2nd_order')) for celltype in order]
order3_ct = [ct.Celltype(f'3rd-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 3rd_order')) for celltype in order]
order4_ct = [ct.Celltype(f'4th-order {celltype}', pymaid.get_skids_by_annotation(f'mw {celltype} 4th_order')) for celltype in order]
exclude = [pymaid.get_skids_by_annotation(x) for x in ['mw MBON', 'mw MBIN', 'mw RGN', 'mw dVNC', 'mw dSEZ', 'mw KC']]
exclude = [x for sublist in exclude for x in sublist]
exclude = exclude + brain_inputs

# identify LNs
# use 0.5 output fraction within group threshold
threshold = 0.5
LNs_2nd = [celltype.identify_LNs(threshold, summed_adj, adj_aa, sens[i], outputs, exclude=exclude)[0] for i, celltype in enumerate(order2_ct)]
LNs_3rd = [celltype.identify_LNs(threshold, summed_adj, adj_aa, order2_ct[i].get_skids(), outputs, exclude=exclude)[0] for i, celltype in enumerate(order3_ct)]
LNs_4th = [celltype.identify_LNs(threshold, summed_adj, adj_aa, order3_ct[i].get_skids(), outputs, exclude=exclude)[0] for i, celltype in enumerate(order4_ct)]

# export LNs
[pymaid.add_annotations(LNs_2nd[i], f'mw brain 2nd_order LN {name}') for i, name in enumerate(order) if len(LNs_2nd[i])>0]
pymaid.add_meta_annotations([f'mw brain 2nd_order LN {name}' for i, name in enumerate(order) if len(LNs_2nd[i])>0], 'mw brain inputs 2nd_order LN')
[pymaid.add_annotations(LNs_3rd[i], f'mw brain 3rd_order LN {name}') for i, name in enumerate(order) if len(LNs_3rd[i])>0]
pymaid.add_meta_annotations([f'mw brain 3rd_order LN {name}' for i, name in enumerate(order) if len(LNs_3rd[i])>0], 'mw brain inputs 3rd_order LN')
[pymaid.add_annotations(LNs_4th[i], f'mw brain 4th_order LN {name}') for i, name in enumerate(order) if len(LNs_4th[i])>0]
pymaid.add_meta_annotations([f'mw brain 4th_order LN {name}' for i, name in enumerate(order) if len(LNs_4th[i])>0], 'mw brain inputs 4th_order LN')

# add special case for olfactory/gustatory 2nd-order because it's so interconnected
pymaid.clear_cache()
ct_skids = pymaid.get_skids_by_annotation('mw olfactory 2nd_order') + pymaid.get_skids_by_annotation('mw gustatory-external 2nd_order') + pymaid.get_skids_by_annotation('mw gustatory-pharyngeal 2nd_order')
input_skids = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw olfactory') + ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw gustatory-external') + ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw gustatory-pharyngeal')
olf_gust_order2 = ct.Celltype('2nd-order olfactory-gustatory', ct_skids)
olf_gust_LN_order2 = olf_gust_order2.identify_LNs(threshold, summed_adj, adj_aa, input_skids, outputs, exclude=exclude)[0] 
pymaid.add_annotations(olf_gust_LN_order2, 'mw brain 2nd_order LN olfactory-gustatory')
pymaid.clear_cache()
pymaid.add_meta_annotations('mw brain 2nd_order LN olfactory-gustatory', 'mw brain inputs 2nd_order LN')
# APL not picked up because it doesn't receive input from 2nd-order olfactory neurons
# perhaps could update the definition of LN a bit

# add input/output type of LN
# use 0.5 output fraction within group threshold
threshold = 0.5
LNs_io_2nd = [celltype.identify_in_out_LNs(threshold, summed_adj, outputs, inputs, exclude=exclude)[0] for i, celltype in enumerate(order2_ct)]
LNs_io_3rd = [celltype.identify_in_out_LNs(threshold, summed_adj, outputs, inputs, exclude=exclude)[0] for i, celltype in enumerate(order3_ct)]
LNs_io_4th = [celltype.identify_in_out_LNs(threshold, summed_adj, outputs, inputs, exclude=exclude)[0] for i, celltype in enumerate(order4_ct)]

# export LNs
[pymaid.add_annotations(LNs_io_2nd[i], f'mw brain 2nd_order LN_io {name}') for i, name in enumerate(order) if len(LNs_io_2nd[i])>0]
pymaid.add_meta_annotations([f'mw brain 2nd_order LN_io {name}' for i, name in enumerate(order) if len(LNs_io_2nd[i])>0], 'mw brain inputs 2nd_order LN_io')
[pymaid.add_annotations(LNs_io_3rd[i], f'mw brain 3rd_order LN_io {name}') for i, name in enumerate(order) if len(LNs_io_3rd[i])>0]
pymaid.add_meta_annotations([f'mw brain 3rd_order LN_io {name}' for i, name in enumerate(order) if len(LNs_io_3rd[i])>0], 'mw brain inputs 3rd_order LN_io')
[pymaid.add_annotations(LNs_io_4th[i], f'mw brain 4th_order LN_io {name}') for i, name in enumerate(order) if len(LNs_io_4th[i])>0]
pymaid.add_meta_annotations([f'mw brain 4th_order LN_io {name}' for i, name in enumerate(order) if len(LNs_io_4th[i])>0], 'mw brain inputs 4th_order LN_io')

# %%
# plot sensories/ascendings

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-cold', 'thermo-warm', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#DBA728', '#35B3E7', '#ed1c24',
            '#5c62ad', '#f15a29', '#007c70', '#49b292', '#e04f9c', '#662d91']

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in input_types.source.loc[order]])
min_alpha = 0.1
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in input_types.source.loc[order]]

n_rows = 2
n_cols = 6

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, skids in enumerate(input_types.source.loc[order]):
    neurons = pymaid.get_neurons(skids)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=colors[i], alpha=alphas[i], ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_sens_asc.png', format='png', dpi=300, transparent=True)

# %%
# plot 2nd-order types

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-cold', 'thermo-warm', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#DBA728', '#35B3E7', '#ed1c24',
            '#5c62ad', '#f15a29', '#007c70', '#49b292', '#e04f9c', '#662d91']

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in input_types.order2.loc[order]])
min_alpha = 0.1
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in input_types.order2.loc[order]]

n_rows = 2
n_cols = 6

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, skids in enumerate(input_types.order2.loc[order]):
    neurons = pymaid.get_neurons(skids)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=colors[i], alpha=alphas[i], ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_sens_2ndOrder.png', format='png', dpi=300, transparent=True)

# %%
# plot 3rd-order

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-cold', 'thermo-warm', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#DBA728', '#35B3E7', '#ed1c24',
            '#5c62ad', '#f15a29', '#007c70', '#49b292', '#e04f9c', '#662d91']

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in input_types.order3.loc[order]])
min_alpha = 0.025
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in input_types.order3.loc[order]]

n_rows = 2
n_cols = 6

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, skids in enumerate(input_types.order3.loc[order]):
    neurons = pymaid.get_neurons(skids)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=colors[i], alpha=alphas[i], ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_sens_3rdOrder.png', format='png', dpi=300)

# %%
# plot each type sequentially: sens -> 2nd-order -> 3rd-order

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-cold', 'thermo-warm', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#DBA728', '#35B3E7', '#ed1c24',
            '#5c62ad', '#f15a29', '#007c70', '#49b292', '#e04f9c', '#662d91']

neurons = [x for sublist in list(zip(input_types.source.loc[order], input_types.order2.loc[order], input_types.order3.loc[order])) for x in sublist]
colors = list(np.repeat(colors, 3))

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in neurons])
min_alpha = 0.025
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in neurons]

n_rows = len(input_types.index)
n_cols = 3

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, neuron_types in enumerate(neurons):

    # load neurons one at a time to prevent CATMAID time-out bug
    neurons_loaded = pymaid.get_neurons(neuron_types[0])
    for j in range(1, len(neuron_types)):
        loaded = pymaid.get_neurons(neuron_types[j])
        neurons_loaded = neurons_loaded + loaded

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons_loaded, neuropil], connectors_only=False, color=colors[i], alpha=alphas[i], ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_sens_2nd_3rd_order.png', format='png', dpi=300)

# %%
