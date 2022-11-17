#%%
from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from contools import Promat, Celltype, Celltype_Analyzer, Analyze_Cluster

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)
brain = pymaid.get_skids_by_annotation('mw brain neurons')

lineages = pymaid.get_annotated('Volker').name
lineages = [('\\' + x) if x[0]=='*' in x else x for x in lineages]
remove_these = [ 'DPLal',
                'BLP1/2_l akira',
                'BLAd_l akira',
                'BLD5/6_l akira',
                'DPMl_l',
                '\\*DPLc_ant_med_r akira',
                '\\*DPLc_ant_med_r akira',
                '\\*DPLm_r akira',
                '\\*DPMl12_post_r akira',
                '\\*DPMpl3_r akira',
                'unknown lineage']
lineages = [x for x in lineages if x not in remove_these ]
lineages.sort()
lineage_skids = [list(np.intersect1d(pymaid.get_skids_by_annotation(x), brain)) for x in lineages]

lineage_skids_summed = [lineage_skids[i] + lineage_skids[i+1] for i in np.arange(0, len(lineages), 2)]
lineages_oneside = [lineages[i] for i in np.arange(0, len(lineages), 2)]

# load cluster data
all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

lvl=7
clusters_lvl7 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

# %%
# known cell types in clusters

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl7.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

# %%
# lineages in clusters

all_lineage_skids = [Celltype(lineages_oneside[i].replace('\\*', '').replace(' akira', '').replace('_l', '').replace(' left', ''), lineage_skids_summed[i]) for i in range(0, len(lineage_skids_summed))]
cluster_analyze = clusters_lvl7.cluster_cta
for i in range(len(cluster_analyze.Celltypes)):
    cluster_analyze.Celltypes[i].name = f'cluster_{i}'

cluster_analyze.set_known_types(all_lineage_skids)
lineage_memberships = cluster_analyze.memberships()

memberships_sort = lineage_memberships.iloc[0:len(lineage_memberships.index)-1, :]
memberships_sort.sort_values(by=list(memberships_sort.columns), ascending=False, inplace=True)

plt.rcParams['font.size'] = 3
ind = np.arange(0, len(cluster_analyze.Celltypes))

g = sns.jointplot(data=memberships_sort, kind='hist', legend=False)
g.ax_marg_y.cla()
g.ax_marg_x.cla()
sns.heatmap(memberships_sort, ax=g.ax_joint, cbar=False, cmap='Blues', vmax=1)

g.ax_marg_x.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
bottom = memberships.iloc[0, :]
for i in range(1, len(memberships.index)):
    g.ax_marg_x.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
    bottom = bottom + memberships.iloc[i, :]
g.ax_marg_x.set(xlim = (-1, len(ind)), xticks=([]), yticks=([]))
g.ax_marg_y.axis('off')
g.ax_marg_x.axis('off')

plt.savefig('plots/lineage-clusters.pdf', format='pdf', bbox_inches='tight')

plt.rcParams['font.size'] = 5
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.boxplot(y=(memberships_sort>0).sum(axis=0), orient='v', ax=ax)
ax.set(ylabel='Number of lineages per cluster', xticks=([]))
plt.savefig('plots/lineages-per-clusters_boxplot.pdf', format='pdf', bbox_inches='tight')

plt.rcParams['font.size'] = 5
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.stripplot(y=(memberships_sort>0).sum(axis=0), orient='v', ax=ax, s= 3)
ax.set(ylabel='Number of lineages per cluster', xticks=([]))
plt.savefig('plots/lineages-per-clusters.pdf', format='pdf', bbox_inches='tight')

thresholds = [0.025, 0.05, 0.1, 0.2, 0.5]
for threshold in thresholds:
    fig, ax = plt.subplots(1,1,figsize=(2,2))
    sns.boxplot(y=(memberships_sort>threshold).sum(axis=0), orient='v', ax=ax)
    ax.set(ylabel='Number of lineages per cluster', xticks=([]))
    plt.savefig(f'plots/lineages-per-clusters_threshold-{threshold}_boxplot.pdf', format='pdf', bbox_inches='tight')

thresholds = [0.025, 0.05, 0.1, 0.2, 0.5]
for threshold in thresholds:
    fig, ax = plt.subplots(1,1,figsize=(2,2))
    sns.violinplot(y=(memberships_sort>threshold).sum(axis=0), orient='v', ax=ax)
    ax.set(ylabel='Number of lineages per cluster', xticks=([]))
    plt.savefig(f'plots/lineages-per-clusters_threshold-{threshold}_stripplot.pdf', format='pdf', bbox_inches='tight')
# %%
