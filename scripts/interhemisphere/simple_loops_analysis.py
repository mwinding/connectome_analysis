#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Prograph, Analyze_Nx_G
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'


adj = Promat.pull_adj('ad', data_date=data_date)
ad_edges = Promat.pull_edges('ad', threshold=0.01, data_date=data_date, pairs_combined=True)
ad_edges_split = Promat.pull_edges('ad', threshold=0.01, data_date=data_date, pairs_combined=False)

graph = Analyze_Nx_G(ad_edges)
graph_split = Analyze_Nx_G(ad_edges_split, split_pairs=True)

pairs = Promat.get_pairs(pairs_path=pairs_path)
# %%
# load neuron types

# majority types
ipsi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
ipsi = ipsi + list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))
bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
contralateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))

# minority types
ipsi_bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
bilateral_bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
contra_bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))

all_contralateral_axons = bilateral + contralateral + contra_bilateral

# %%
# find loops of various types

# prepping cell types
# major
contra_pairs = Promat.get_pairs_from_list(np.intersect1d(contralateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
bilateral_pairs = Promat.get_pairs_from_list(np.intersect1d(bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
ipsi_pairs = Promat.get_pairs_from_list(np.intersect1d(ipsi, graph_split.G.nodes), pairList=pairs, return_type='pairs')

# minor
# did some pilot experiments and didn't find anything interesting; didn't keep the code
ipsi_bilateral_pairs = Promat.get_pairs_from_list(np.intersect1d(ipsi_bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
bilateral_bilateral_pairs = Promat.get_pairs_from_list(np.intersect1d(bilateral_bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
contra_bilateral_pairs = Promat.get_pairs_from_list(np.intersect1d(contra_bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')

# identifying loops
types = ['ipsi', 'bilateral', 'contra']
path_lengths = [2,3,4,5] # not including source node
loop_df = []
loop_df_plotting = []
for i, pair_type in enumerate([ipsi_pairs, bilateral_pairs, contra_pairs]):
    for length in path_lengths:
        loop_data = graph_split.partner_loop_probability(pairs = pair_type, length=length)

        loop_df.append([types[i], length, loop_data[0], loop_data[1], loop_data[2]])
        loop_df_plotting.append([types[i], length, loop_data[1], f'nonpartner_loops'])
        loop_df_plotting.append([types[i], length, loop_data[0], f'partner_loops'])

loop_df = pd.DataFrame(loop_df, columns = ['celltype', 'path_length', 'connection_probability', 'fraction_nonpartner_loops', 'raw_paths'])
loop_df_plotting = pd.DataFrame(loop_df_plotting, columns = ['celltype', 'path_length', 'connection_probability', 'loop_type'])
# %%
# plot data

length = 2
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(.8,1.5))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.25, alpha=0.5)
ax.set(ylim=(-0.01, 0.15), title = f'Edges in Loop:{length}', yticks=[0,0.05, .1, .15])
plt.savefig(f'plots/interhemisphere_partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(1.5,0.8))
sns.barplot(data = data, x='celltype', y='connection_probability', hue='loop_type', ax=ax)
ax.set(ylim=(-0.01, 0.15), title = f'Edges in Loop:{length}', yticks=[0,0.05, .1, .15])
plt.savefig(f'plots/interhemisphere_partner-loops_length-{length}_barplot.pdf', format='pdf', bbox_inches='tight')

length = 3
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(1.5,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')
plt.savefig(f'plots/interhemisphere_partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')

length = 4
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(1.5,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.50), title = f'Edges in Loop:{length}')
plt.savefig(f'plots/interhemisphere_partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')

length = 5
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')
plt.savefig(f'plots/interhemisphere_partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')


# plot all together
n_rows = 1
n_cols = 4

fig, axs = plt.subplots(n_rows,n_cols, figsize=(2.5, 1.5), sharey=True)
fig.tight_layout(pad=0)
for i, length in enumerate(path_lengths):
    #inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = axs[i]

    data = loop_df_plotting[loop_df_plotting.path_length == length]
    sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.25, alpha=0.5)
    ax.set(ylim=(-0.01, 0.2), title = f'Edges in Loop:{length}', ylabel='', yticks=[0, 0.05, .1, .15, 0.2])
    if(i>0):
        ax.legend([],[], frameon=False)

plt.savefig(f'plots/interhemisphere_partner-loops_all.pdf', format='pdf', bbox_inches='tight')

# rearrange as matrix
# use this for main figure
data = loop_df_plotting.pivot(index=['path_length' , 'loop_type'], columns=['celltype']).loc[:, list(zip(['connection_probability']*3, ['ipsi', 'bilateral', 'contra']))]
fig, axs = plt.subplots(1,1, figsize=(0.75, 1))
sns.heatmap(data, cmap = 'Blues', vmax=0.2)
plt.savefig(f'plots/interhemisphere_partner-loops-all_heatmap.pdf', format='pdf', bbox_inches='tight')

# %%
# direct reciprocal loops between contra neurons

contra_test = ad_edges.set_index('upstream_pair_id', drop=False).copy()
contra_test = contra_test.loc[np.intersect1d(all_contralateral_axons, contra_test.index)]
contra_test = contra_test.set_index('downstream_pair_id', drop=False)
contra_test = contra_test.loc[np.intersect1d(all_contralateral_axons, contra_test.index)]

pair_loops = []
nonpair_loops = []
for num, i in enumerate(np.unique(contra_test.upstream_pair_id)):
    for j in np.unique(contra_test.downstream_pair_id):
        if(sum((contra_test.upstream_pair_id==i) & (contra_test.downstream_pair_id==j) & (contra_test.type=='contralateral'))>0):
            if(sum((contra_test.upstream_pair_id==j) & (contra_test.downstream_pair_id==i) & (contra_test.type=='contralateral'))>0):

                if(i==j): 
                    pair_loops.append([i,j])
                    print(f'Pair loop exists between {i} and {j}')

                if(i!=j): 
                    nonpair_loops.append([i,j])
                    print(f'Non-pair loop exists between {i} and {j}')

    print(num)


connection_prob = [len(nonpair_loops)/(len(np.intersect1d(all_contralateral_axons, pairs.leftid))*(len(np.intersect1d(all_contralateral_axons, pairs.leftid))-1)),
                    len(pair_loops)/len(np.intersect1d(all_contralateral_axons, pairs.leftid))]

fig, ax = plt.subplots(1,1,figsize=(1,1))
sns.barplot(y=connection_prob, x=['non-pair', 'pair'], ax=ax)
ax.set(yticks=(0, 0.025, 0.05))
ax.bar_label(ax.containers[0])
plt.savefig(f'plots/interhemisphere_partner-loops-2hop_barplot.pdf', format='pdf', bbox_inches='tight')

# %%
# celltypes in pair loops and non-pair loops

pair_loops_ct = Celltype('pair-loops', [x[0] for x in pair_loops])
nonpair_loops_ct = Celltype('nonpair-loops', [x for sublist in nonpair_loops for x in sublist])

loops_cta = Celltype_Analyzer([pair_loops_ct, nonpair_loops_ct])

_, celltypes = Celltype_Analyzer.default_celltypes()
loops_cta.set_known_types(celltypes)
num_loops_df = loops_cta.memberships(raw_num=True)
loops_cta.plot_memberships('plots/interhemisphere_pair-nonpair_loops_celltypes.pdf', figsize=(1,1))


# pull official celltype colors
official_order = ['sensories', 'ascendings', 'PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
colors = list(pymaid.get_annotated('mw brain simple colors').name)
colors_names = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
color_sort = [np.where(x.replace('mw brain ', '')==np.array(official_order))[0][0] for x in colors_names]
colors = [element for _, element in sorted(zip(color_sort, colors))]
colors = colors + ['tab:gray']

# donut plot of cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(num_loops_df.loc[official_order, 'pair-loops (24)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/pair-loop_types.pdf', format='pdf', bbox_inches='tight')

# %%
# connection probability between ipsi/bilateral/contra
contra_pairs = pm.Promat.get_pairs_from_list(contralateral, pairList=pairs, return_type='pairs')
bilateral_pairs = pm.Promat.get_pairs_from_list(bilateral, pairList=pairs, return_type='pairs')
ipsi_pairs = pm.Promat.get_pairs_from_list(ipsi, pairList=pairs, return_type='pairs')

data_adj = ad_edges_split.set_index(['upstream_pair_id', 'downstream_pair_id'])
celltypes = [list(ipsi_pairs.leftid), list(bilateral_pairs.leftid), list(contra_pairs.leftid), 
                list(contra_pairs.rightid), list(bilateral_pairs.rightid), list(ipsi_pairs.rightid)]

mat = np.zeros(shape=(len(celltypes), len(celltypes)))
for i, pair_type1 in enumerate(celltypes):
    for j, pair_type2 in enumerate(celltypes):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph_split.G.edges): connection.append(1)
                if((skid1, skid2) not in graph_split.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = ['ipsi-left', 'bilateral-left', 'contra-left', 'contra-right', 'bilateral-right', 'ipsi-right'],
                        index = ['ipsi-left', 'bilateral-left', 'contra-left', 'contra-right', 'bilateral-right', 'ipsi-right'])

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.heatmap(df, square=True, cmap='Blues', vmax=0.02)
plt.savefig(f'interhemisphere/plots/connection-probability_all-interhemisphere-types.pdf', format='pdf', bbox_inches='tight')

# %%
