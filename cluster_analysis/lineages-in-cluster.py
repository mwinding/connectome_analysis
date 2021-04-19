#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import connectome_tools.process_matrix as pm
import connectome_tools.cascade_analysis as casc

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
                '\\*DPlal1-3_med_l akira',
                'unknown lineage']
lineages = [x for x in lineages if x not in remove_these ]
lineages.sort()
lineage_skids = [list(np.intersect1d(pymaid.get_skids_by_annotation(x), brain)) for x in lineages]

lineage_skids_summed = [lineage_skids[i] + lineage_skids[i+1] for i in np.arange(0, len(lineages), 2)]
lineages_oneside = [lineages[i] for i in np.arange(0, len(lineages), 2)]

# load cluster data
clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)

# separate meta file with median_node_visits from sensory for each node
# determined using iterative random walks
meta_with_order = pd.read_csv('data/meta_data_w_order.csv', index_col = 0, header = 0)

def cluster_order(lvl_label_str, meta_with_order):
    lvl = clusters.groupby(lvl_label_str)
    order_df = []
    for key in lvl.groups:
        skids = lvl.groups[key]        
        node_visits = meta_with_order.loc[skids, :].median_node_visits
        order_df.append([key, list(skids), np.nanmean(node_visits)])

    order_df = pd.DataFrame(order_df, columns = ['cluster', 'skids', 'node_visit_order'])
    order_df = order_df.sort_values(by = 'node_visit_order')
    order_df.reset_index(inplace=True, drop=True)

    return(order_df, list(order_df.cluster))

lvl7, order_7 = cluster_order('lvl7_labels', meta_with_order)
lvl5, order_5 = cluster_order('lvl5_labels', meta_with_order)

# %%
# known cell types in clusters

celltypes, celltype_names, celltype_colors = pm.Promat.celltypes()
celltype_colors = celltype_colors + ['tab:gray']
all_clusters = [casc.Celltype(lvl7.cluster[i], lvl7.skids[i]) for i in range(0, len(lvl7))]
all_celltypes = [casc.Celltype(celltype_names[i], celltypes[i]) for i in range(0, len(celltypes))]
cluster_analyze = casc.Celltype_Analyzer(all_clusters)
cluster_analyze.set_known_types(all_celltypes)
memberships = cluster_analyze.memberships()
memberships = memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,15,12,13,14], :]
celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,15,12,13,14]]

ind = np.arange(0, len(all_clusters))
plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
bottom = memberships.iloc[0, :]
for i in range(1, len(memberships.index)):
    plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
    bottom = bottom + memberships.iloc[i, :]
plt.savefig('cluster_analysis/plots/celltypes-clusters.pdf', format='pdf', bbox_inches='tight')

# %%
# lineages in clusters

all_clusters = [casc.Celltype(lvl7.cluster[i], lvl7.skids[i]) for i in range(0, len(lvl7))]
all_lineage_skids = [casc.Celltype(lineages_oneside[i], lineage_skids_summed[i]) for i in range(0, len(lineage_skids_summed))]
cluster_analyze = casc.Celltype_Analyzer(all_clusters)
cluster_analyze.set_known_types(all_lineage_skids)
lineage_memberships = cluster_analyze.memberships()

memberships_sort = lineage_memberships.iloc[0:len(lineage_memberships.index)-1, :]
memberships_sort.sort_values(by=list(memberships_sort.columns), ascending=False, inplace=True)

plt.rcParams['font.size'] = 3

g = sns.jointplot(data=memberships_sort, x='0-1-1-0-0-0-0-1 (47)', y='0-1-1-0-0-0-0-1 (47)', kind='hist')
g.ax_marg_y.cla()
g.ax_marg_x.cla()
sns.heatmap(memberships_sort, ax=g.ax_joint, cbar=False)

g.ax_marg_x.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
bottom = memberships.iloc[0, :]
for i in range(1, len(memberships.index)):
    g.ax_marg_x.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
    bottom = bottom + memberships.iloc[i, :]
g.ax_marg_x.set(xlim = (-1, len(ind)), xticks=([]), yticks=([]))
g.ax_marg_y.axis('off')
g.ax_marg_x.axis('off')

plt.savefig('cluster_analysis/plots/lineage-clusters.pdf', format='pdf', bbox_inches='tight')

plt.rcParams['font.size'] = 5
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.stripplot(y=(memberships_sort>0).sum(axis=0), orient='v', ax=ax, s= 3)
ax.set(ylabel='Number of lineages per cluster', xticks=([]))
plt.savefig('cluster_analysis/plots/lineages-per-clusters.pdf', format='pdf', bbox_inches='tight')

thresholds = [0.025, 0.05, 0.1, 0.2, 0.5]
for threshold in thresholds:
    fig, ax = plt.subplots(1,1,figsize=(2,2))
    sns.stripplot(y=(memberships_sort>threshold).sum(axis=0), orient='v', ax=ax, s= 3)
    ax.set(ylabel='Number of lineages per cluster', xticks=([]))
    plt.savefig(f'cluster_analysis/plots/lineages-per-clusters_threshold-{threshold}.pdf', format='pdf', bbox_inches='tight')
# %%
