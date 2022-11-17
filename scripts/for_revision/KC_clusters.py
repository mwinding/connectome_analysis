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

# KC clusters
annots = ['30_level-7_clusterID-152_signal-flow_0.347',
            '33_level-7_clusterID-154_signal-flow_0.277',
            '34_level-7_clusterID-153_signal-flow_0.239',
            '36_level-7_clusterID-151_signal-flow_0.202',
            '39_level-7_clusterID-47_signal-flow_0.181',
            '40_level-7_clusterID-91_signal-flow_0.179',
            '42_level-7_clusterID-94_signal-flow_0.154',
            '49_level-7_clusterID-50_signal-flow_-0.017']

# KC lineages
lineages = ['Lineage MBNB A', 'Lineage MBNB B', 'Lineage MBNB C', 'Lineage MBNB D']
lineages = [Celltype(annot, pymaid.get_skids_by_annotation(annot)) for annot in lineages]
KC_cts = [Celltype(annot, pymaid.get_skids_by_annotation(annot)) for annot in annots]

KC_cta = Celltype_Analyzer(KC_cts)
KC_cta.set_known_types(lineages)
KC_cta.memberships(raw_num=True)

lr_cts = [Celltype('left', pymaid.get_skids_by_annotation('mw left')), Celltype('right', pymaid.get_skids_by_annotation('mw right'))]
KC_cta.set_known_types(lr_cts)
KC_cta.memberships(raw_num=True)

claw_cts = ['mw KC_subclass_1claw', 'mw KC_subclass_2claw', 'mw KC_subclass_3claw', 'mw KC_subclass_4claw', 'mw KC_subclass_5claw', 'mw KC_subclass_6claw']
claw_cts = [Celltype(annot.replace('mw KC_subclass_', ''), pymaid.get_skids_by_annotation(annot)) for annot in claw_cts]
KC_cta.set_known_types(claw_cts)
KC_cta.memberships(raw_num=True)
# %%
# what about level 6?
lvl=6
clusters_lvl6 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl6.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

KC_lvl6_clusters = [clusters_lvl6.cluster_cta.Celltypes[i] for i in range(len((memberships.loc['KCs']==1).values)) if (memberships.loc['KCs']==1).values[i]==True]
KC_lvl6_clusters_cta = Celltype_Analyzer(KC_lvl6_clusters)
KC_lvl6_clusters_cta.set_known_types(claw_cts)
KC_lvl6_clusters_cta.memberships(raw_num=True)

# level 4
lvl=4
clusters_lvl4 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl4.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

KC_lvl4_clusters = [clusters_lvl4.cluster_cta.Celltypes[i] for i in range(len((memberships.loc['KCs']==1).values)) if (memberships.loc['KCs']==1).values[i]==True]
KC_lvl4_clusters_cta = Celltype_Analyzer(KC_lvl4_clusters)
KC_lvl4_clusters_cta.set_known_types(claw_cts)
KC_lvl4_clusters_cta.memberships(raw_num=True)

# %%
# similar quesiton with MBONs and MBINs, do they split by some kind of functional group?

# level 7
lvl=7
clusters_lvl7 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl7.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

MBON_types_cts = ['mw MBON subclass_appetitive', 'mw MBON subclass_aversive', 'mw MBON subclass_neither']
MBON_types_cts = [Celltype(annot.replace('mw ', ''), pymaid.get_skids_by_annotation(annot)) for annot in MBON_types_cts]

MBON_lvl7_clusters = [clusters_lvl7.cluster_cta.Celltypes[i] for i in range(len((memberships.loc['MBONs']>0).values)) if (memberships.loc['MBONs']>0).values[i]==True]
MBON_lvl7_clusters_cta = Celltype_Analyzer(MBON_lvl7_clusters)
MBON_lvl7_clusters_cta.set_known_types(MBON_types_cts)
MBON_lvl7_clusters_cta.memberships(raw_num=True)

# level 8
lvl=8
clusters_lvl8 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl8.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

MBON_types_cts = ['mw MBON subclass_appetitive', 'mw MBON subclass_aversive', 'mw MBON subclass_neither']
MBON_types_cts = [Celltype(annot.replace('mw ', ''), pymaid.get_skids_by_annotation(annot)) for annot in MBON_types_cts]

MBON_lvl8_clusters = [clusters_lvl8.cluster_cta.Celltypes[i] for i in range(len((memberships.loc['MBONs']>0).values)) if (memberships.loc['MBONs']>0).values[i]==True]
MBON_lvl8_clusters_cta = Celltype_Analyzer(MBON_lvl8_clusters)
MBON_lvl8_clusters_cta.set_known_types(MBON_types_cts)
MBON_lvl8_clusters_cta.memberships(raw_num=True)

# level 7 MBINs
lvl=7
clusters_lvl7 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl7.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

# level 8 MBINs
lvl=8
clusters_lvl8 = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')

_, celltypes = Celltype_Analyzer.default_celltypes()
celltype_colors = [x.color for x in celltypes] + ['tab:gray']
cluster_analyze = clusters_lvl8.cluster_cta
cluster_analyze.set_known_types(celltypes)
memberships = cluster_analyze.memberships()

# %%
