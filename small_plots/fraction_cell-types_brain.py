#%%

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

brain = pymaid.get_skids_by_annotation('mw brain neurons')

# inputs
input_names = pymaid.get_annotated('mw brain inputs and ascending').name
input_names_formatted = ['ORN', 'thermo', 'photo', 'AN-sens', 'MN-sens', 'vtd', 'proprio', 'mechano', 'class II_III', 'noci', 'unknown']
inputs = [pymaid.get_skids_by_annotation(x) for x in input_names]
inputs = inputs + [pymaid.get_skids_by_annotation('mw A1 ascending unknown')]

outputs = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain outputs').name]
outputs_names_formatted = ['dSEZs', 'dVNCs', 'RGNs']

outputs_names_formatted = [outputs_names_formatted[i] for i in [2,0,1]]
outputs = [outputs[i] for i in [2,0,1]]

outputs_all = [x for sublist in outputs for x in sublist]

brain = list(np.setdiff1d(brain, outputs_all))

celltypes_df, celltypes = ct.Celltype_Analyzer.default_celltypes()

all_celltypes = [celltypes[1]] + celltypes[3:len(celltypes)]
all_interneurons = [celltypes[1]] + celltypes[3:len(celltypes)-3]

all_neurons = [x for sublist in [x.skids for x in celltypes] for x in sublist]
unknown_brain = list(np.setdiff1d(brain, all_neurons))
unknown_ct = ct.Celltype('Other', unknown_brain, color='tab:grey')
all_celltypes = all_celltypes + [unknown_ct]
celltype_names = [x.get_name() for x in all_celltypes]
# %%
# plot number brain inputs, interneurons, outputs

col_width = 0.125
plot_height = 1
ylim = (0, 500)

# interneurons (note that these counts are for mutually exclusive types)
colors = [x.color for x in all_interneurons]
fig, ax = plt.subplots(1,1,figsize=(col_width*len(all_interneurons), plot_height))
graph = sns.barplot(x=[x.get_name() for x in all_interneurons], y=[len(x.skids) for x in all_interneurons], ax=ax, palette = colors)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5, [len(x.skids) for x in all_interneurons][i], ha="center", color=colors[i], fontdict = {'fontsize': 4})
    i += 1
ax.set(ylim=ylim)
plt.savefig('small_plots/plots/general-celltype-counts.pdf', format='pdf', bbox_inches='tight')

# inputs
colors = ['#004C26', '#00753F', '#008743', '#00a854', '#00bf5f', '#00e271', '#0089a0', '#00a5d1', '#00b7f7', '#77cdfc', '#c0e7f9']
fig, ax = plt.subplots(1,1,figsize=(col_width*len(inputs),plot_height))
graph = sns.barplot(x=input_names_formatted, y=[len(skids) for skids in inputs], ax=ax, palette=colors)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5, [len(skids) for skids in inputs][i], ha="center", color=colors[i], fontdict = {'fontsize': 4})
    i += 1
ax.set(ylim=ylim)
plt.savefig('small_plots/plots/input-counts.pdf', format='pdf', bbox_inches='tight')

# outputs
colors = ['#9467BD','#D88052', '#A52A2A']
fig, ax = plt.subplots(1,1,figsize=(col_width*len(outputs),plot_height))
graph = sns.barplot(x=outputs_names_formatted, y=[len(skids) for skids in outputs], ax=ax, palette=colors)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5, [len(skids) for skids in outputs][i],ha="center", color=colors[i], fontdict = {'fontsize': 4})
    i += 1
ax.set(ylim=ylim)
plt.savefig('small_plots/plots/output-counts.pdf', format='pdf', bbox_inches='tight')

# %%
# plot interneurons with overlap allowed

overlap_celltype_names = celltype_names[0:-1] # remove the 'Other' category and recalculate later
annots = ['mw brain ' + name for name in overlap_celltype_names]

celltype_skids = [list(np.unique(ct.Celltype_Analyzer.get_skids_from_meta_annotation(annot))) for annot in annots]
celltype_skids = celltype_skids + [unknown_ct.skids]

ylim = (0,1000)
colors = [x.color for x in all_celltypes]
fig, ax = plt.subplots(1,1,figsize=(col_width*len(celltype_skids), plot_height))
graph = sns.barplot(x=celltype_names, y=[len(skids) for skids in celltype_skids], ax=ax, palette = colors)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5, [len(skids) for skids in celltype_skids][i], ha="center", color=colors[i], fontdict = {'fontsize': 4})
    i += 1
ax.set(ylim=ylim)
plt.savefig('small_plots/plots/general-celltype-counts_overlaps-allowed.pdf', format='pdf', bbox_inches='tight')

# upset plot between all cell types
#overlap_celltype_df = pd.DataFrame(zip(celltype_names, celltype_skids), columns=['celltype', 'skids'])
overlap_celltype_cts = [ct.Celltype(x[0], x[1], x[2]) for x in zip(celltype_names, celltype_skids, colors)]
overlap_celltype_cts = ct.Celltype_Analyzer(overlap_celltype_cts)
overlap_celltype_cts.upset_members(
        threshold = 6,
        path = 'small_plots/plots/general-celltype-counts_overlaps-allowed_UPSET', 
        plot_upset=True,
        exclude_singletons_from_threshold = True,
        threshold_dual_cats=6
    )

# %%
# plot morphology of each cell type: exclusive and promiscuous cells displays separately

exclusive_skids = overlap_celltype_cts.upset_members(threshold=10000, exclude_singletons_from_threshold=True)[1]
exclusive_skids.reverse()

celltype_data_df = pd.DataFrame(zip(celltype_names, [x.skids for x in exclusive_skids], celltype_skids), columns=['celltype', 'exclusive_skids', 'all_skids'])
celltype_data_df['promiscuous_skids'] = [list(np.setdiff1d(celltype_data_df['all_skids'][i], celltype_data_df['exclusive_skids'][i])) for i in range(len(celltype_data_df))]

# plot morphology
neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)

neurons = [x for sublist in list(zip(celltype_data_df.all_skids[0:12], celltype_data_df.exclusive_skids[0:12], celltype_data_df.promiscuous_skids[0:12])) for x in sublist]
colors = [x.color for x in overlap_celltype_cts.Celltypes]
colors = list(np.repeat(colors, 3))

# alpha determined by number of neurons being plotted
max_members = max([len(x) for x in neurons])
min_alpha = 0.025
max_alpha = 0.2 # max is really min+max
alphas = [min_alpha+(max_alpha-len(x)/max_members*max_alpha) for x in neurons]

n_rows = len(celltype_data_df.index)
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

fig.savefig(f'small_plots/plots/morpho_brain-cell-types.png', format='png', dpi=300)



# plot only unkonwn neurons
# load neurons one at a time to prevent CATMAID time-out bug
max_members = max([len(x) for x in neurons])
min_alpha = 0.025
max_alpha = 0.2 # max is really min+max
alpha = min_alpha+(max_alpha-len(neurons_loaded)/max_members*max_alpha)

neurons_other = [celltype_data_df.all_skids[12], celltype_data_df.exclusive_skids[12]]

n_rows = 1
n_cols = 2

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, neuron_types in enumerate(neurons_other):

    # load neurons one at a time to prevent CATMAID time-out bug
    neurons_loaded = pymaid.get_neurons(neuron_types[0])
    for j in range(1, len(neuron_types)):
        loaded = pymaid.get_neurons(neuron_types[j])
        neurons_loaded = neurons_loaded + loaded

    ax = fig.add_subplot(gs[0, i], projection="3d")
    axs[0,i] = ax
    navis.plot2d(x=[neurons_loaded, neuropil], connectors_only=False, color='tab:grey', alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'small_plots/plots/morpho_brain-cell-types_other.png', format='png', dpi=300)


# plot pre-output neurons with different alpha
alpha = 0.13
neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)

neurons_preoutput = [celltype_data_df.all_skids[10], celltype_data_df.exclusive_skids[10], celltype_data_df.promiscuous_skids[10],
                        celltype_data_df.all_skids[11], celltype_data_df.exclusive_skids[11], celltype_data_df.promiscuous_skids[11]]

n_rows = 2
n_cols = 3

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)
colors = ['#e2b39f', '#ba8583']
colors = list(np.repeat(colors, 3))

for i, neuron_types in enumerate(neurons_preoutput):

    # load neurons one at a time to prevent CATMAID time-out bug
    neurons_loaded = pymaid.get_neurons(neuron_types[0])
    for j in range(1, len(neuron_types)):
        loaded = pymaid.get_neurons(neuron_types[j])
        neurons_loaded = neurons_loaded + loaded

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons_loaded, neuropil], connectors_only=False, color=colors[i], alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'small_plots/plots/morpho_brain-cell-types_preoutputs.png', format='png', dpi=300)

# %%
