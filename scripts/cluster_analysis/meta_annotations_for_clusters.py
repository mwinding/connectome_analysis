# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Analyze_Cluster

# %%
# set up annotation names and organize clusters

skids = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
skids = list(np.setdiff1d(skids, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

sort='signal_flow'
cluster_lvls = [0,1,2,3,4,5,6,7]
clusters = [Analyze_Cluster(x, meta_data_path='data/graphs/meta_data.csv', skids=skids, sort=sort).cluster_df for x in cluster_lvls]
annot_names = [list(map(lambda x: (f'{x[0]}_level-{cluster_lvls[i]}_clusterID-{x[1]:.0f}_{sort.replace("_", "-")}_{x[2]:.3f}'), zip(clust.index, clust.cluster, clust.loc[:, f'sum_{sort}']))) for i, clust in enumerate(clusters)]

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