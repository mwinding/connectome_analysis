#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load clusters and adj

cluster_lvl = 7
clusters = clust.Analyze_Cluster(cluster_lvl=cluster_lvl-1) # 0 indexing

# set up weak and strong edge adjacency matrices
ad_adj = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accessory')

ad_adj_weak = ad_adj.copy()
ad_adj_weak[ad_adj_weak>1] = 0

ad_adj_strong = ad_adj.copy()
ad_adj_strong[ad_adj_strong<5] = 0

# pull connectivity
strong_connect = clusters.cluster_cta.connectivity(ad_adj_strong)
weak_connect = clusters.cluster_cta.connectivity(ad_adj_weak)

# %%
# generate data

strong_df = []
for i in strong_connect.index:
    intra = strong_connect.loc[i, i]
    inter = sum(strong_connect.loc[i, strong_connect.columns.drop(i)])
    total = intra + inter

    intra = intra/total
    inter = inter/total
    strong_df.append([i, intra, inter])

strong_df = pd.DataFrame(strong_df, columns = ['cluster', 'strong_intra', 'strong_inter'])
strong_df.set_index('cluster', drop=True, inplace=True)

weak_df = []
for i in weak_connect.index:
    intra = weak_connect.loc[i, i]
    inter = sum(weak_connect.loc[i, weak_connect.columns.drop(i)])
    total = intra + inter

    intra = intra/total
    inter = inter/total
    weak_df.append([i, intra, inter])

weak_df = pd.DataFrame(weak_df, columns = ['cluster', 'weak_intra', 'weak_inter'])
weak_df.set_index('cluster', drop=True, inplace=True)

data_df = pd.concat([strong_df, weak_df], axis=1)
data_df.drop(65, inplace=True) # drop last cluster which has no connectivity
data_df['cluster'] = data_df.index

plot_df = pd.melt(data_df, id_vars='cluster', var_name='connectivity', value_name='fraction')
sns.barplot(data = plot_df, x='connectivity', y='fraction')
plt.savefig(f'cluster_analysis/plots/strong-vs-weak-connectivity_clusters.pdf', format='pdf', bbox_inches='tight')

# %%
# generate data type 2

strong_df = []
for i in strong_connect.index:
    intra = strong_connect.loc[i, i]
    inter = strong_connect.loc[i, strong_connect.columns.drop(i)]
    total = intra + sum(inter)

    strong_df.append([i, intra/total, 'strong_intra'])

    for sample in inter:
        strong_df.append([i, sample/total, 'strong_inter'])

strong_df = pd.DataFrame(strong_df, columns = ['cluster', 'fraction', 'connectivity'])
strong_df.set_index('cluster', drop=True, inplace=True)

weak_df = []
for i in weak_connect.index:
    intra = weak_connect.loc[i, i]
    inter = weak_connect.loc[i, weak_connect.columns.drop(i)]
    total = intra + sum(inter)

    weak_df.append([i, intra/total, 'weak_intra'])

    for sample in inter:
        weak_df.append([i, sample/total, 'weak_inter'])

weak_df = pd.DataFrame(weak_df, columns = ['cluster', 'fraction', 'connectivity'])
weak_df.set_index('cluster', drop=True, inplace=True)

data_df = pd.concat([strong_df, weak_df], axis=0)
data_df.drop(65, inplace=True) # drop last cluster which has no connectivity
data_df['cluster'] = data_df.index

fig, ax = plt.subplots(1,1)
sns.barplot(data = data_df, x='connectivity', y='fraction', ax=ax)
plt.savefig(f'cluster_analysis/plots/strong-vs-weak-connectivity_clusters_type2.pdf', format='pdf', bbox_inches='tight')

# %%
# third plot: how many clusters are targeted by weak and strong connections

data_df = pd.concat([(strong_connect>0).sum(axis=1), (weak_connect>0).sum(axis=1)], axis=1)
data_df.columns = ['strong', 'weak']
data_df['clusters'] = data_df.index

data_df = pd.melt(data_df, id_vars='clusters', var_name='connectivity', value_name='num_clusters')

fig, ax = plt.subplots(1,1)
sns.barplot(data = data_df, x='connectivity', y='num_clusters', ax=ax)
plt.savefig(f'cluster_analysis/plots/strong-vs-weak-connectivity_clusters_type3.pdf', format='pdf', bbox_inches='tight')

# %%
