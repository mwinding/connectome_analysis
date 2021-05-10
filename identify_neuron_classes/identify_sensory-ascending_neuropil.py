#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load previously generated adjacency matrices
# see 'network_analysis/generate_all_edges.py'

brain_inputs = [x for sublist in [pymaid.get_skids_by_annotation(annot) for annot in pymaid.get_annotated('mw brain inputs and ascending').name] for x in sublist]
brain = pymaid.get_skids_by_annotation('mw brain neurons') + brain_inputs

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad, adj_aa, adj_dd, adj_da = [pd.read_csv(f'data/adj/all-neurons_{name}.csv', index_col = 0).rename(columns=int) for name in adj_names]
adj_ad = adj_ad.loc[np.intersect1d(adj_ad.index, brain), np.intersect1d(adj_ad.index, brain)]
adj_aa = adj_aa.loc[np.intersect1d(adj_aa.index, brain), np.intersect1d(adj_aa.index, brain)]
adj_dd = adj_dd.loc[np.intersect1d(adj_dd.index, brain), np.intersect1d(adj_dd.index, brain)]
adj_da = adj_da.loc[np.intersect1d(adj_da.index, brain), np.intersect1d(adj_da.index, brain)]
adjs = [adj_ad, adj_aa, adj_dd, adj_da]

# load input counts
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
outputs = pd.read_csv('data/graphs/outputs.csv', index_col=0)

# process to produce %input pairwise matrix
mat_ad, mat_aa, mat_dd, mat_da = [pm.Adjacency_matrix(adj=adj, input_counts=inputs, mat_type=adj_names[i]) for i, adj in enumerate(adjs)]
# %%
# load appropriate sensory and ascending types
input_types = [[annot.replace('mw ', ''), pymaid.get_skids_by_annotation(annot)] for annot in pymaid.get_annotated('mw brain inputs and ascending').name]
input_types = pd.DataFrame(input_types, columns = ['type', 'source'])

all_ascending = [pymaid.get_skids_by_annotation(annot) for annot in pymaid.get_annotated('mw brain ascendings').name]
all_ascending = [x for sublist in all_ascending for x in sublist]

input_types = input_types.append(pd.DataFrame([['A1 ascendings all', all_ascending]], columns = ['type', 'source']))
input_types = input_types.reset_index(drop=True)

# %%
# identify all 2nd-order neurons

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

# export IDs
#[pymaid.add_annotations(input_types.order2.iloc[i], f'mw brain 2nd_order {input_types.type.iloc[i]}') for i in range(0, 10)]
#pymaid.add_meta_annotations([f'mw brain 2nd_order {input_types.type.iloc[i]}' for i in range(0, 10)], 'mw brain inputs 2nd_order')
#[pymaid.add_annotations(input_types.order3.iloc[i], f'mw brain 3rd_order {input_types.type.iloc[i]}') for i in range(0, 10)]
#pymaid.add_meta_annotations([f'mw brain 3rd_order {input_types.type.iloc[i]}' for i in range(0, 10)], 'mw brain inputs 3rd_order')

order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
input_types = input_types.set_index('type')

# look at overlap between order2 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 2nd-order', input_types.order2.loc[i]) for i in order]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type='iou'), ax=ax)
fig.savefig(f'identify_neuron_classes/plots/similarity_sens_2nd-order.pdf', format='pdf')

# look at overlap between order3 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 3rd-order', input_types.order3.loc[i]) for i in order]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type='iou'), ax=ax)
fig.savefig(f'identify_neuron_classes/plots/similarity_sens_3rd-order.pdf', format='pdf')

# look at overlap between order4 neurons
fig, ax = plt.subplots(1,1, figsize=(3,2))
cts = [ct.Celltype(i + ' 4th-order', input_types.order4.loc[i]) for i in order]
cts_analyze = ct.Celltype_Analyzer(cts)
sns.heatmap(cts_analyze.compare_membership(sim_type='iou'), ax=ax)
fig.savefig(f'identify_neuron_classes/plots/similarity_sens_4th-order.pdf', format='pdf')

# %%
# visualize neurons in adjacency
summed_adj = pd.read_csv(f'data/adj/all-neurons_all-all.csv', index_col = 0).rename(columns=int)

# identify all PNs / LNs
def identify_LNs(summed_adj, adj_aa, skids, input_skids, outputs, pairs = pm.Promat.get_pairs(), sort = True):
    mat = summed_adj.loc[np.intersect1d(summed_adj.index, skids), np.intersect1d(summed_adj.index, skids)]
    mat = mat.sum(axis=1)

    mat_axon = adj_aa.loc[np.intersect1d(adj_aa.index, skids), np.intersect1d(adj_aa.index, input_skids)]
    mat_axon = mat_axon.sum(axis=1)

    # convert to % outputs
    skid_percent_output = []
    for skid in skids:
        skid_output = 0
        output = sum(outputs.loc[skid, :])
        if(output != 0):
            if(skid in mat.index):
                skid_output = skid_output + mat.loc[skid]/output
            if(skid in mat_axon.index):
                skid_output = skid_output + mat_axon.loc[skid]/output

        skid_percent_output.append([skid, skid_output])

    skid_percent_output = pm.Promat.convert_df_to_pairwise(pd.DataFrame(skid_percent_output, columns=['skid', 'percent_output_intragroup']).set_index('skid'))

    LNs = skid_percent_output.groupby('pair_id').sum()      
    LNs = LNs[np.array([x for sublist in (LNs>=1).values for x in sublist])]
    MBONs = pymaid.get_skids_by_annotation('mw MBON')
    MBINs = pymaid.get_skids_by_annotation('mw MBIN')
    RGNs = pymaid.get_skids_by_annotation('mw RGN')
    exclude = RGNs + MBINs + MBONs
    LNs = np.setdiff1d(list(LNs.index), exclude) # don't count MBONs/MBINs/RGNs as LNs
    return(list(LNs), skid_percent_output)

# identify LNs of different sens order
LNs_2nd = [identify_LNs(summed_adj, adj_aa, input_types.order2.loc[i], input_types.source.loc[i], outputs)[0] for i in order]
LNs_2nd_data = [identify_LNs(summed_adj, adj_aa, input_types.order2.loc[i], input_types.source.loc[i], outputs)[1] for i in order]

LNs_3rd = [identify_LNs(summed_adj, adj_aa, input_types.order3.loc[i], input_types.order2.loc[i], outputs)[0] for i in order]
LNs_3rd_data = [identify_LNs(summed_adj, adj_aa, input_types.order3.loc[i], input_types.order2.loc[i], outputs)[1] for i in order]

LNs_4th = [identify_LNs(summed_adj, adj_aa, input_types.order4.loc[i], input_types.order3.loc[i], outputs)[0] for i in order]
LNs_4th_data = [identify_LNs(summed_adj, adj_aa, input_types.order4.loc[i], input_types.order3.loc[i], outputs)[1] for i in order]
# nothing useful came from 4th level LNs

# export LNs
[pymaid.add_annotations(LNs_2nd[i], f'mw brain 2nd_order LN {name}') for i, name in enumerate(order) if len(LNs_2nd[i])>0]
pymaid.add_meta_annotations([f'mw brain 2nd_order LN {name}' for i, name in enumerate(order) if len(LNs_2nd[i])>0], 'mw brain inputs 2nd_order LN')
[pymaid.add_annotations(LNs_3rd[i], f'mw brain 3rd_order LN {name}') for i, name in enumerate(order) if len(LNs_3rd[i])>0]
pymaid.add_meta_annotations([f'mw brain 3rd_order LN {name}' for i, name in enumerate(order) if len(LNs_3rd[i])>0], 'mw brain inputs 3rd_order LN')

# %%
# identify 2nd-order PNs from ANs/MNs

# %%
# plot 2nd-order types

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
colors = ['#00753F', '#1D79B7', '#5D8C90', '#D88052', '#FF8734', '#E55560', '#F9EB4D', '#8C7700', '#9467BD','#D88052', '#A52A2A']
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#35B3E7', '#ED1C24',
            '#662D91', '#F15A29', '#00A79D', '#F93DB6', '#754C29']

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in input_types.order2.loc[order]])
min_alpha = 0.1
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in input_types.order2.loc[order]]

n_rows = 2
n_cols = 5

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
'''
# plot synapses on the same plot to see neuropil areas
fig = plt.figure(figsize=(2, 2))
gs = plt.GridSpec(1, 1, figure=fig, wspace=0, hspace=0)
axs = np.empty((1, 1), dtype=object)
ax = fig.add_subplot(gs[0], projection="3d")

for i, skids in enumerate(input_types.order2.iloc[0]):
    neurons = pymaid.get_neurons(skids)

    navis.plot2d(x=[neurons, neuropil], connectors_only=True, ax=ax)

    ax.azim = -90
    ax.elev = -90
    ax.dist = 5.75
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_sens_2ndOrder_synapses.pdf', format='pdf')
'''
# %%
# plot 3rd-order

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#35B3E7', '#ED1C24',
            '#662D91', '#F15A29', '#00A79D', '#F93DB6', '#754C29']

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in input_types.order3.loc[order]])
min_alpha = 0.025
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in input_types.order3.loc[order]]

n_rows = 2
n_cols = 5

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
    ax.dist = 5.75
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_sens_3rdOrder.png', format='png', dpi=300)




# %%
