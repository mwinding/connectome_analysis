#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load clusters and plot counts

cluster_lvl = 6 #this is cluster level 7
clusters, cluster_names = ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_lvl}', split=True)

counts = [len(cluster) for cluster in clusters]


fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.stripplot(y=counts, orient='v', ax=ax, s=1.25, color='k', alpha=0.5)
sns.boxplot(y=counts, whis=np.inf, ax=ax, linewidth=0.5)
plt.savefig(f'small_plots/plots/counts-per-cluster-level-{cluster_lvl+1}.pdf', format='pdf', bbox_inches='tight')

# %%
# distribution of intracluster similarity

morpho_sim = [1, 1, 1, 0.95, 0.89, 0.97, 1, .92, .96, .94, .97, .99,
                .89, .99, .95, 1, 1, .94, .86, .93, .45, .81, .91, .58,
                .75, 1, .71, .91, .91, .87, .94, .99, .99,
                .99, .83, .99, .69, .90, .37, 1, .97, .85, .64, .83, .62,
                .75, 1, .88, .53, .84, .65, .72, .82, .68, .64, .43, .75,
                .73, .66, .68, .67, .81, .42, .67, .9, .49, .68, .69,
                .72, .84, .79, .76, .82, .65, .88, .82, .87, .66, .95,
                .76, .68, .90, .87, .94, .77]


fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.histplot(morpho_sim, binwidth=.05, ax=ax)
ax.set(xlim=(0,1.05), ylim=(0, 20))
plt.savefig(f'small_plots/plots/morpho-similarity_cluster-level-{cluster_lvl+1}.pdf', format='pdf', bbox_inches='tight')

# %%
# left/right neurons per cluster

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

left_counts = [len(np.intersect1d(cluster, left)) for cluster in clusters]
right_counts = [len(np.intersect1d(cluster, right)) for cluster in clusters]

left_fraction = np.array(left_counts)/(np.array(left_counts)+np.array(right_counts))
right_fraction = np.array(right_counts)/(np.array(left_counts)+np.array(right_counts))
ind = np.arange(0, len(left_fraction))

# standard barplot
fig, ax = plt.subplots(1,1, figsize=(4,2))
plt.bar(ind, left_fraction, color = 'tab:blue')
plt.bar(ind, right_fraction, bottom = left_fraction, color = 'tab:purple')
plt.savefig('small_plots/plots/left-right_per-cluster.pdf', format='pdf', bbox_inches='tight')


# custom dot-line plot
fig, ax = plt.subplots(1,1, figsize=(1,2))
# plotting the points
plt.scatter(np.zeros(len(left_fraction)), left_fraction)
plt.scatter(np.ones(len(right_fraction)), right_fraction)

# plotting the lines
for i in range(len(left_fraction)):
    plt.plot( [0,1], [left_fraction[i], right_fraction[i]], c='k')

plt.xticks([0,1], ['left', 'right'])
plt.show()
ax.set(ylim=(0,1), xlim=(-0.25, 1.25))
plt.savefig('small_plots/plots/left-right_per-cluster_dot-line-plot.pdf', format='pdf', bbox_inches='tight')

# %%
# fraction brain inputs, periphery, memory, deep brain, pre-output, and outputs per cluster

celltype_df,celltypes = ct.Celltype_Analyzer.default_celltypes()
celltype_df.index = celltype_df.name

inputs = ct.Celltype('inputs', celltype_df.loc['sensories'].skids + celltype_df.loc['ascendings'].skids, color='#00441b')
innate = ct.Celltype('innate', celltype_df.loc['PNs'].skids + celltype_df.loc['PNs-somato'].skids + celltype_df.loc['LHNs'].skids + celltype_df.loc['LNs'].skids + celltype_df.loc['FFNs'].skids, color='#00a551')
memory = ct.Celltype('learning/memory', celltype_df.loc['MBINs'].skids + celltype_df.loc['KCs'].skids + celltype_df.loc['MBONs'].skids, color='#ec1c24')
deep = ct.Celltype('deep brain', celltype_df.loc['CNs'].skids + celltype_df.loc['MB-FBNs'].skids, color='#00adee')
preout = ct.Celltype('pre-output', celltype_df.loc['pre-dSEZs'].skids + celltype_df.loc['pre-dVNCs'].skids, color='#a87c4f')
out = ct.Celltype('output', celltype_df.loc['dSEZs'].skids + celltype_df.loc['dVNCs'].skids + celltype_df.loc['RGNs'].skids, color='#603813')

cts = [inputs, innate, memory, deep, preout, out]
cluster_cts = [ct.Celltype(f'cluster-{i}', skids) for i, skids in enumerate(clusters)]
cluster_cts = ct.Celltype_Analyzer(cluster_cts)

cluster_cts.set_known_types(cts)
cluster_cts.plot_memberships(path='small_plots/plots/general-brain-regions_per-cluster.pdf', figsize=(2, 1))



# %%
