# object for analysing hit_histograms from cascades run using TraverseDispatcher

import numpy as np
import pandas as pd
import sys
from connectome_tools.process_matrix import Promat

class Analyze_Cluster():

    def __init__(self, cluster_info_path, cluster_sort_path, lvl_label_str):
        # load cluster data
        self.clusters_info = pd.read_csv(cluster_info_path, index_col = 0, header = 0)

        # separate meta file with median_node_visits from sensory for each node
        # determined using iterative random walks
        self.meta_with_order = pd.read_csv(cluster_sort_path, index_col = 0, header = 0)
        self.clusters, self.cluster_order = self.cluster_order(lvl_label_str)

    def cluster_order(self, lvl_label_str):
        lvl = self.clusters_info.groupby(lvl_label_str)
        order_df = []
        for key in lvl.groups:
            skids = lvl.groups[key]
            node_visits = self.meta_with_order.loc[skids, :].median_node_visits
            order_df.append([key, list(skids), np.nanmean(node_visits)])

        order_df = pd.DataFrame(order_df, columns = ['cluster', 'skids', 'node_visit_order'])
        order_df = order_df.sort_values(by = 'node_visit_order')
        order_df.reset_index(inplace=True, drop=True)

        return(order_df, list(order_df.cluster))
'''
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
'''