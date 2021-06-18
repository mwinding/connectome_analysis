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

ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
ad_edges_split = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

graph = pg.Analyze_Nx_G(ad_edges)
graph_split = pg.Analyze_Nx_G(ad_edges_split, split_pairs=True)

pairs = pm.Promat.get_pairs()
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
contra_pairs = pm.Promat.get_pairs_from_list(np.intersect1d(contralateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
bilateral_pairs = pm.Promat.get_pairs_from_list(np.intersect1d(bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
ipsi_pairs = pm.Promat.get_pairs_from_list(np.intersect1d(ipsi, graph_split.G.nodes), pairList=pairs, return_type='pairs')

# minor
# did some pilot experiments and didn't find anything interesting; didn't keep the code
ipsi_bilateral_pairs = pm.Promat.get_pairs_from_list(np.intersect1d(ipsi_bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
bilateral_bilateral_pairs = pm.Promat.get_pairs_from_list(np.intersect1d(bilateral_bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')
contra_bilateral_pairs = pm.Promat.get_pairs_from_list(np.intersect1d(contra_bilateral, graph_split.G.nodes), pairList=pairs, return_type='pairs')

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
fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')
plt.savefig(f'interhemisphere/plots/partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')

length = 3
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')
plt.savefig(f'interhemisphere/plots/partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')

length = 4
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')
plt.savefig(f'interhemisphere/plots/partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')

length = 5
data = loop_df_plotting[loop_df_plotting.path_length == length]
fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')
plt.savefig(f'interhemisphere/plots/partner-loops_length-{length}.pdf', format='pdf', bbox_inches='tight')


# plot all together
n_rows = 1
n_cols = 4

fig, axs = plt.subplots(n_rows,n_cols, figsize=(9, 3))
fig.tight_layout(pad=3.0)
for i, length in enumerate(path_lengths):
    #inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = axs[i]

    data = loop_df_plotting[loop_df_plotting.path_length == length]
    sns.pointplot(data = data, x='loop_type', y='connection_probability', hue='celltype', ax=ax, scale=0.5, alpha=0.5)
    ax.set(ylim=(-0.01, 0.20), title = f'Edges in Loop:{length}')

plt.savefig(f'interhemisphere/plots/partner-loops_all.pdf', format='pdf', bbox_inches='tight')

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
