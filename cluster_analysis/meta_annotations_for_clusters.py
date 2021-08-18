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
import connectome_tools.process_skeletons as skel
import connectome_tools.cluster_analysis as clust

# %%
# set up annotation names and organize clusters

cluster_lvls = [0,1,2,3,4,5,6,7]
clusters = [clust.Analyze_Cluster(x).cluster_df for x in cluster_lvls]
annot_names = [list(map(lambda x: (f'{x[0]}_clusterID-{x[1]:.0f}_walksort-{x[2]:.3f}'), zip(clust.index, clust.cluster, clust.sum_walk_sort))) for clust in clusters]

for i in range(len(clusters)):
    clusters[i]['annot'] = annot_names[i]

# %%
# writing annotations for clusters

for i in range(len(clusters)):
    for j in range(len(clusters[i].index)):
        annot_skids = clusters[i].loc[j, 'skids']
        annot = clusters[i].loc[j, 'annot']
        pymaid.add_annotations(annot_skids, [annot])

    annots = list(clusters[i].annot)
    pymaid.add_meta_annotations(annots, f'mw brain clusters level {i}')

pymaid.add_meta_annotations([f'mw brain clusters level {lvl}' for lvl in cluster_lvls], 'mw brain clusters')

# %%
#