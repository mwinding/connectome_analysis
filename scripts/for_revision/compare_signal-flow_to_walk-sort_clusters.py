# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Analyze_Cluster

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load clusters and sort using walk-sort or signal-flow

skids = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
skids = list(np.setdiff1d(skids, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

sort = 'walk_sort'
cluster_lvls = [0,1,2,3,4,5,6,7,8]
clusters_walksort = [Analyze_Cluster(x, meta_data_path='data/graphs/meta_data.csv', skids=skids, sort=sort).cluster_df for x in cluster_lvls]
annot_names = [list(map(lambda x: (f'{x[0]}_level-{cluster_lvls[i]}_clusterID-{x[1]:.0f}_{sort}-{x[2]:.3f}'), zip(clust.index, clust.cluster, clust.loc[:, f'sum_{sort}']))) for i, clust in enumerate(clusters_walksort)]

for i in range(len(clusters_walksort)):
    clusters_walksort[i]['annot'] = annot_names[i]

sort = 'signal_flow'
cluster_lvls = [0,1,2,3,4,5,6,7,8]
clusters_signalflow = [Analyze_Cluster(x, meta_data_path='data/graphs/meta_data.csv', skids=skids, sort=sort).cluster_df for x in cluster_lvls]
annot_names = [list(map(lambda x: (f'{x[0]}_level-{cluster_lvls[i]}_clusterID-{x[1]:.0f}_{sort}-{x[2]:.3f}'), zip(clust.index, clust.cluster, clust.loc[:, f'sum_{sort}']))) for i, clust in enumerate(clusters_signalflow)]

for i in range(len(clusters_signalflow)):
    clusters_signalflow[i]['annot'] = annot_names[i]

sort = 'rank_signal_flow'
cluster_lvls = [0,1,2,3,4,5,6,7,8]
clusters_rank_signalflow = [Analyze_Cluster(x, meta_data_path='data/graphs/meta_data.csv', skids=skids, sort=sort).cluster_df for x in cluster_lvls]
annot_names = [list(map(lambda x: (f'{x[0]}_level-{cluster_lvls[i]}_clusterID-{x[1]:.0f}_{sort}-{x[2]:.3f}'), zip(clust.index, clust.cluster, clust.loc[:, f'sum_{sort}']))) for i, clust in enumerate(clusters_rank_signalflow)]

for i in range(len(clusters_rank_signalflow)):
    clusters_rank_signalflow[i]['annot'] = annot_names[i]

cluster_lvl = 7
clusters_rank_signalflow[cluster_lvl].sort_values(by=f'sum_{sort}', ascending=True, inplace=True)
clusters_rank_signalflow[cluster_lvl].reset_index(inplace=True, drop=True)

# %%
# plot memberships of each cluster at level 7, compare sorting by eye between walk-sort and signal-flow

cluster_lvl = 7
clusters_sf_cta = Celltype_Analyzer([Celltype(clusters_signalflow[cluster_lvl].loc[i].annot, clusters_signalflow[cluster_lvl].loc[i].skids) for i in range(len(clusters_signalflow[cluster_lvl]))])
clusters_rsf_cta = Celltype_Analyzer([Celltype(clusters_rank_signalflow[cluster_lvl].loc[i].annot, clusters_rank_signalflow[cluster_lvl].loc[i].skids) for i in range(len(clusters_rank_signalflow[cluster_lvl]))])
clusters_ws_cta = Celltype_Analyzer([Celltype(clusters_walksort[cluster_lvl].loc[i].annot, clusters_walksort[cluster_lvl].loc[i].skids) for i in range(len(clusters_walksort[cluster_lvl]))])

_, celltypes = Celltype_Analyzer.default_celltypes()

clusters_sf_cta.set_known_types(celltypes)
clusters_rsf_cta.set_known_types(celltypes)
clusters_ws_cta.set_known_types(celltypes)

# plot signal-flow sorted clusters (level 7)
figsize = (6,3)
path = 'plots/clusters-sorted-by-signal-flow.pdf'
celltype_colors = [x.get_color() for x in clusters_sf_cta.get_known_types()]
all_memberships = clusters_sf_cta.memberships()
all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs

ind = np.arange(0, len(clusters_sf_cta.Celltypes))
fig, ax = plt.subplots(1,1,figsize=figsize)
plt.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
bottom = all_memberships.iloc[0, :]
for i in range(1, len(all_memberships.index)):
    plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
    bottom = bottom + all_memberships.iloc[i, :]
plt.savefig(path, format='pdf', bbox_inches='tight')

# plot walk-sort sorted clusters (level 7)
figsize = (6,3)
path = 'plots/clusters-sorted-by-walk-sort.pdf'
celltype_colors = [x.get_color() for x in clusters_ws_cta.get_known_types()]
all_memberships = clusters_ws_cta.memberships()
all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs

ind = np.arange(0, len(clusters_ws_cta.Celltypes))
fig, ax = plt.subplots(1,1,figsize=figsize)
plt.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
bottom = all_memberships.iloc[0, :]
for i in range(1, len(all_memberships.index)):
    plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
    bottom = bottom + all_memberships.iloc[i, :]
plt.savefig(path, format='pdf', bbox_inches='tight')


# plot rank signal flow sorted clusters (level 7)
figsize = (6,3)
path = 'plots/clusters-sorted-by-rank-signal-flow.pdf'
celltype_colors = [x.get_color() for x in clusters_rsf_cta.get_known_types()]
all_memberships = clusters_rsf_cta.memberships()
all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs

ind = np.arange(0, len(clusters_rsf_cta.Celltypes))
fig, ax = plt.subplots(1,1,figsize=figsize)
plt.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
bottom = all_memberships.iloc[0, :]
for i in range(1, len(all_memberships.index)):
    plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
    bottom = bottom + all_memberships.iloc[i, :]
plt.savefig(path, format='pdf', bbox_inches='tight')
# we decided to go with signal-flow for simplicity; a reviewer wanted us to reduce methods complexity
# this will allow us to remove one of the two sorting algorithms used

# %%
