# %%
# identify clusters with higher than average aa, da, or dd connections (input or output)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'arial'

cluster_lvl = 7 
clusters = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_lvl}', split=True, return_celltypes=True)

adj_ad = Promat.pull_adj(type_adj='ad', data_date=data_date)
adj_aa = Promat.pull_adj(type_adj='aa', data_date=data_date)
adj_dd = Promat.pull_adj(type_adj='dd', data_date=data_date)
adj_da = Promat.pull_adj(type_adj='da', data_date=data_date)

# %%
'''
# looking at connection type by cluster
# doesn't work properly

clusters_cta = Celltype_Analyzer(clusters)

cluster_ad = clusters_cta.connectivity(adj = adj_ad)
cluster_aa = clusters_cta.connectivity(adj = adj_aa)
cluster_dd = clusters_cta.connectivity(adj = adj_dd)
cluster_da = clusters_cta.connectivity(adj = adj_da)

outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)

cluster_output_data = []
for i in outputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_output_data.append([i, cluster.name, outputs.loc[i, 'axon_output']+outputs.loc[i, 'dendrite_output']])

cluster_output_df = pd.DataFrame(cluster_output_data, columns=['skid', 'cluster', 'output'])
summed_outputs = cluster_output_df.groupby('cluster').sum().loc[cluster_ad.iloc[0, :].index]

fracs = []
fracs_ad = []
for i in range(len(cluster_ad)):
    cluster_summed = cluster_ad + cluster_aa + cluster_dd + cluster_da
    boolean = cluster_summed.iloc[i, :]>0
    modulatory = (cluster_aa.iloc[i, :][boolean]+cluster_dd.iloc[i, :][boolean]+cluster_da.iloc[i, :][boolean])
    frac = modulatory/summed_outputs[boolean].output
    frac = [[x, cluster_summed.iloc[i, :].name] for x in frac]
    fracs.append(frac)

    frac_ad = cluster_ad.iloc[i, :][boolean]/summed_outputs[boolean].output
    frac_ad = [[x, cluster_summed.iloc[i, :].name] for x in frac_ad]
    fracs_ad.append(frac_ad)

fracs_data = [x for sublist in fracs for x in sublist]
fracs_df = pd.DataFrame(fracs_data, columns=['fraction_aa-dd-da', 'cluster'])

fracs_ad_data = [x for sublist in fracs_ad for x in sublist]
fracs_ad_df = pd.DataFrame(fracs_ad_data, columns=['fraction_ad', 'cluster'])

fig, ax = plt.subplots(1,1,figsize=(6,3))
sns.barplot(data=fracs_df, x='cluster', y='fraction_aa-dd-da', errwidth=0.5, ax=ax)
plt.savefig('plots/clusters_fraction-aa-da-dd-outputs.png', format='png')

# %%
# connection type per individual neuron
# then combine into cluster

plt.rcParams['font.size'] = 10

# outputs
non_ad = adj_aa.sum(axis=1) + adj_dd.sum(axis=1) + adj_da.sum(axis=1)
outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)

cluster_output_data = []
for i in outputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_output_data.append([i, cluster.name, non_ad.loc[i], outputs.loc[i, 'axon_output']+outputs.loc[i, 'dendrite_output'], non_ad.loc[i]/(outputs.loc[i, 'axon_output']+outputs.loc[i, 'dendrite_output'])])

cluster_output_df = pd.DataFrame(cluster_output_data, columns=['skid', 'cluster', 'non_ad', 'output', 'fraction_non_ad'])

fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_output_df, x='cluster', y='fraction_non_ad', order = cluster_ad.iloc[0, :].index, errwidth=0.5, ax=ax)
ax.set(ylim=[0,1], xticklabels=[], ylabel='Fraction of Output (average per cluster)', title='Axo-axonic, dendro-dendritic, and dendro-axonic connections')
plt.savefig('plots/clusters_fraction-aa-da-dd-outputs.png', format='png')


# inputs
non_ad = adj_dd.sum(axis=0)
inputs = pd.read_csv(f'data/adj/inputs_{data_date}.csv', index_col=0)

cluster_input_data = []
for i in inputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_input_data.append([i, cluster.name, non_ad.loc[i] + inputs.loc[i, 'axon_input'], inputs.loc[i, 'axon_input']+inputs.loc[i, 'dendrite_input'], (non_ad.loc[i] + inputs.loc[i, 'axon_input'])/(inputs.loc[i, 'axon_input']+inputs.loc[i, 'dendrite_input'])])

cluster_input_df = pd.DataFrame(cluster_input_data, columns=['skid', 'cluster', 'non_ad', 'output', 'fraction_non_ad'])

fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_input_df, x='cluster', y='fraction_non_ad', order = cluster_ad.iloc[0, :].index, errwidth=0.5, ax=ax)
ax.set(ylim=[0,1], xticklabels=[], ylabel='Fraction of Input (average per cluster)', title='Axo-axonic, dendro-dendritic, and dendro-axonic connections')
plt.savefig('plots/clusters_fraction-aa-da-dd-inputs.png', format='png')
'''
# %%
# input-output connectivity per cluster
import scipy.stats as stats
import scikit_posthocs as sp

########
# fraction of ad connections
con_type = 'ad'
connections = adj_ad.sum(axis=1) + adj_ad.sum(axis=0)
outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)
inputs = pd.read_csv(f'data/adj/inputs_{data_date}.csv', index_col=0)
inputs_outputs = outputs.sum(axis=1) + inputs.sum(axis=1)

# collecting data into df
cluster_io_data = []
for i in inputs_outputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_io_data.append([i, cluster.name, connections.loc[i], inputs_outputs.loc[i], connections.loc[i]/inputs_outputs.loc[i]])

cluster_io_df = pd.DataFrame(cluster_io_data, columns=['skid', 'cluster', f'{con_type}', 'io', f'{con_type}_frac'])

# performing kruskal test (nonparametric ANOVA) to determine if there are clusters that are different
stats_boolean = (cluster_io_df.groupby('cluster').count()>=5).skid # requires >=5 samples for kruskal test
fvalue, pvalue = stats.kruskal(*cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean])
print(fvalue, pvalue)
if(pvalue<0.05):
    print(f'Fraction of {con_type} connections are significantly different across clusters (somewhere...)')

# posthoc dunn test to identify which pairwise comparisons are different; didn't use in the end
posthoc = sp.posthoc_dunn(cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean].values, p_adjust = 'bonferroni')
print(f'The mean number of significant pairwise differences for {con_type} were {np.mean((posthoc>0.05).sum(axis=1))} +/- {np.std((posthoc>0.05).sum(axis=1))}')

# plot per cluster data
fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_io_df, x='cluster', y=f'{con_type}_frac', order = [x.name for x in clusters], errwidth=0.5, ax=ax)
ax.set(ylim=[0,1], xticklabels=[], ylabel=f'Fraction of {con_type} connectivity (average per cluster)')
plt.savefig(f'plots/clusters_fraction-{con_type}.pdf', format='pdf')

##########
# fraction of aa connections
con_type = 'aa'
cutoff = 0.3
connections = adj_aa.sum(axis=1) + adj_aa.sum(axis=0)
outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)
inputs = pd.read_csv(f'data/adj/inputs_{data_date}.csv', index_col=0)
inputs_outputs = outputs.sum(axis=1) + inputs.sum(axis=1)

# collecting data into df
cluster_io_data = []
for i in inputs_outputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_io_data.append([i, cluster.name, connections.loc[i], inputs_outputs.loc[i], connections.loc[i]/inputs_outputs.loc[i]])

cluster_io_df = pd.DataFrame(cluster_io_data, columns=['skid', 'cluster', f'{con_type}', 'io', f'{con_type}_frac'])

# performing kruskal test (nonparametric ANOVA) to determine if there are clusters that are different
stats_boolean = (cluster_io_df.groupby('cluster').count()>=5).skid # requires >=5 samples for kruskal test
fvalue, pvalue = stats.kruskal(*cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean])
print(fvalue, pvalue)
if(pvalue<0.05):
    print(f'Fraction of {con_type} connections are significantly different across clusters (somewhere...)')

# posthoc dunn test to identify which pairwise comparisons are different; didn't use in the end
posthoc = sp.posthoc_dunn(cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean].values, p_adjust = 'bonferroni')
print(f'The mean number of significant pairwise differences for {con_type} were {np.mean((posthoc>0.05).sum(axis=1))} +/- {np.std((posthoc>0.05).sum(axis=1))}')

# plot per cluster data
fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_io_df, x='cluster', y=f'{con_type}_frac', order = [x.name for x in clusters], errwidth=0.5, ax=ax)
max_con = max(cluster_io_df.groupby('cluster').mean().loc[:, f'{con_type}_frac']) # identify tallest bar
plt.axhline(y=max_con/2, color='k', alpha=0.1) # draw line at half the height of tallest bar
plt.savefig(f'plots/clusters_fraction-{con_type}.pdf', format='pdf')

# identify all clusters that have bars half as tall as tallest in the plot
high_aa_clusters = cluster_io_df.groupby('cluster').mean()
high_aa_clusters = high_aa_clusters[high_aa_clusters.aa_frac>(max_con/2)]

##########
# fraction of da connections
con_type = 'da'
cutoff = 0.04
connections = adj_da.sum(axis=1) + adj_da.sum(axis=0)
outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)
inputs = pd.read_csv(f'data/adj/inputs_{data_date}.csv', index_col=0)
inputs_outputs = outputs.sum(axis=1) + inputs.sum(axis=1)

# collecting data into df
cluster_io_data = []
for i in inputs_outputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_io_data.append([i, cluster.name, connections.loc[i], inputs_outputs.loc[i], connections.loc[i]/inputs_outputs.loc[i]])

cluster_io_df = pd.DataFrame(cluster_io_data, columns=['skid', 'cluster', f'{con_type}', 'io', f'{con_type}_frac'])

# performing kruskal test (nonparametric ANOVA) to determine if there are clusters that are different
stats_boolean = (cluster_io_df.groupby('cluster').count()>=5).skid # requires >=5 samples for kruskal test
fvalue, pvalue = stats.kruskal(*cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean])
print(fvalue, pvalue)
if(pvalue<0.05):
    print(f'Fraction of {con_type} connections are significantly different across clusters (somewhere...)')

# posthoc dunn test to identify which pairwise comparisons are different; didn't use in the end
posthoc = sp.posthoc_dunn(cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean].values, p_adjust = 'bonferroni')
print(f'The mean number of significant pairwise differences for {con_type} were {np.mean((posthoc>0.05).sum(axis=1))} +/- {np.std((posthoc>0.05).sum(axis=1))}')

# plot per cluster data
fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_io_df, x='cluster', y=f'{con_type}_frac', order = [x.name for x in clusters], errwidth=0.5, ax=ax)
max_con = max(cluster_io_df.groupby('cluster').mean().loc[:, f'{con_type}_frac'])
plt.axhline(y=max_con/2, color='k', alpha=0.1)
plt.savefig(f'plots/clusters_fraction-{con_type}.pdf', format='pdf')

# identify all clusters that have bars half as tall as tallest in the plot
high_da_clusters = cluster_io_df.groupby('cluster').mean()
high_da_clusters = high_da_clusters[high_da_clusters.da_frac>(max_con/2)]

###########
# fraction of dd connections
con_type = 'dd'
cutoff = 0.1
connections = adj_dd.sum(axis=1) + adj_dd.sum(axis=0)
outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)
inputs = pd.read_csv(f'data/adj/inputs_{data_date}.csv', index_col=0)
inputs_outputs = outputs.sum(axis=1) + inputs.sum(axis=1)

# collecting data into df
cluster_io_data = []
for i in inputs_outputs.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_io_data.append([i, cluster.name, connections.loc[i], inputs_outputs.loc[i], connections.loc[i]/inputs_outputs.loc[i]])

cluster_io_df = pd.DataFrame(cluster_io_data, columns=['skid', 'cluster', f'{con_type}', 'io', f'{con_type}_frac'])

# performing kruskal test (nonparametric ANOVA) to determine if there are clusters that are different
stats_boolean = (cluster_io_df.groupby('cluster').count()>=5).skid # requires >=5 samples for kruskal test
fvalue, pvalue = stats.kruskal(*cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean])
print(fvalue, pvalue)
if(pvalue<0.05):
    print(f'Fraction of {con_type} connections are significantly different across clusters (somewhere...)')

# posthoc dunn test to identify which pairwise comparisons are different; didn't use in the end
posthoc = sp.posthoc_dunn(cluster_io_df.groupby('cluster')[f'{con_type}'].apply(list)[stats_boolean].values, p_adjust = 'bonferroni')
print(f'The mean number of significant pairwise differences for {con_type} were {np.mean((posthoc>0.05).sum(axis=1))} +/- {np.std((posthoc>0.05).sum(axis=1))}')

# plot per cluster data
fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_io_df, x='cluster', y=f'{con_type}_frac', order = [x.name for x in clusters], errwidth=0.5, ax=ax)
ax.set(ylim=[0,0.4], xticklabels=[], ylabel=f'Fraction of {con_type} connectivity (average per cluster)')
max_con = max(cluster_io_df.groupby('cluster').mean().loc[:, f'{con_type}_frac'])
plt.axhline(y=max_con/2, color='k', alpha=0.1)
plt.savefig(f'plots/clusters_fraction-{con_type}.pdf', format='pdf')

# identify all clusters that have bars half as tall as tallest in the plot
high_dd_clusters = cluster_io_df.groupby('cluster').mean()
high_dd_clusters = high_dd_clusters[high_dd_clusters.dd_frac>(max_con/2)]

# %%
# which neurons are in each prominent cluster

# pull skids for each cluster
high_aa_clusters_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in high_aa_clusters.index]
high_aa_clusters_skids = [x for sublist in high_aa_clusters_skids for x in sublist]

high_da_clusters_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in high_da_clusters.index]
high_da_clusters_skids = [x for sublist in high_da_clusters_skids for x in sublist]

high_dd_clusters_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in high_dd_clusters.index]
high_dd_clusters_skids = [x for sublist in high_dd_clusters_skids for x in sublist]

high_aa_clusters_ct = Celltype('aa_clusters', high_aa_clusters_skids)
high_da_clusters_ct = Celltype('da_clusters', high_da_clusters_skids)
high_dd_clusters_ct = Celltype('dd_clusters', high_dd_clusters_skids)

# make Celltype_Analyzer to plot celltypes within
cta = Celltype_Analyzer([high_aa_clusters_ct, high_da_clusters_ct, high_dd_clusters_ct])

_, celltypes = Celltype_Analyzer.default_celltypes()
cta.set_known_types(celltypes)
num_df = cta.memberships(raw_num=True)

# pull official celltype colors
official_order = ['sensories', 'ascendings', 'PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
colors = list(pymaid.get_annotated('mw brain simple colors').name)
colors_names = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
color_sort = [np.where(x.replace('mw brain ', '')==np.array(official_order))[0][0] for x in colors_names]
colors = [element for _, element in sorted(zip(color_sort, colors))]
colors = colors + ['tab:gray']

# donut plot of aa cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(num_df.loc[official_order, 'aa_clusters (172)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_aa_celltypes.pdf', format='pdf', bbox_inches='tight')

# donut plot of dd cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(num_df.loc[official_order, 'dd_clusters (92)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_dd_celltypes.pdf', format='pdf', bbox_inches='tight')

# donut plot of da cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(num_df.loc[official_order, 'da_clusters (80)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_da_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# combination of 3 or 4 connection types

# fraction of ad connections
adj_ad_bin = adj_ad.copy()
adj_aa_bin = adj_aa.copy()
adj_dd_bin = adj_dd.copy()
adj_da_bin = adj_da.copy()

adj_ad_bin[adj_ad_bin>0]=1
adj_aa_bin[adj_aa_bin>0]=1
adj_dd_bin[adj_dd_bin>0]=1
adj_da_bin[adj_da_bin>0]=1

# determine total number of connections per neuron
connections = adj_ad_bin + adj_aa_bin + adj_dd_bin + adj_da_bin
total_connections = connections.sum(axis=0) + connections.sum(axis=1)

# how many triple and quadruple connections are there per neuron
connections3 = (connections==3).sum(axis=1) + (connections==3).sum(axis=0)
connections4 = (connections==4).sum(axis=1) + (connections==4).sum(axis=0)

connections3 = connections3/total_connections
connections4 = connections4/total_connections

# make df
cluster_conn_data = []
for i in total_connections.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_conn_data.append([i, cluster.name, connections3.loc[i], connections4.loc[i]])

cluster_conn_df = pd.DataFrame(cluster_conn_data, columns=['skid', 'cluster', 'conn3', 'conn4'])

# performing kruskal test (nonparametric ANOVA) to determine if there are clusters that are different
stats_boolean = (cluster_conn_df.groupby('cluster').count()>=5).skid # requires >=5 samples for kruskal test
fvalue, pvalue = stats.kruskal(*cluster_conn_df.groupby('cluster')['conn3'].apply(list)[stats_boolean])
print(fvalue, pvalue)
if(pvalue<0.05):
    print(f'Fraction of 3 connections are significantly different across clusters (somewhere...)')

stats_boolean = (cluster_conn_df.groupby('cluster').count()>=5).skid # requires >=5 samples for kruskal test
fvalue, pvalue = stats.kruskal(*cluster_conn_df.groupby('cluster')['conn4'].apply(list)[stats_boolean])
print(fvalue, pvalue)
if(pvalue<0.05):
    print(f'Fraction of 4 connections are significantly different across clusters (somewhere...)')

# plot per cluster data
fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_conn_df, x='cluster', y=f'conn3', order = [x.name for x in clusters], errwidth=0.5, ax=ax)
ax.set(ylim=[0,0.05], xticklabels=[], ylabel=f'Fraction of 3 connections (average per cluster)')
conn3_max = max(cluster_conn_df.groupby('cluster').mean().conn3)
plt.axhline(y=conn3_max/2, color='k', alpha=0.1)
plt.savefig(f'plots/clusters_fraction-conn3.pdf', format='pdf')

fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_conn_df, x='cluster', y=f'conn4', order = [x.name for x in clusters], errwidth=0.5, ax=ax)
ax.set(ylim=[0,0.05], xticklabels=[], ylabel=f'Fraction of 4 connections (average per cluster)')
conn4_max = max(cluster_conn_df.groupby('cluster').mean().conn4)
plt.axhline(y=conn4_max/2.4, color='k', alpha=0.1)
plt.savefig(f'plots/clusters_fraction-conn4.pdf', format='pdf')

# identify clusters of interest
conn3_clusters = cluster_conn_df.groupby('cluster').mean()[cluster_conn_df.groupby('cluster').mean().conn3>(conn3_max/2)].index
conn4_clusters = cluster_conn_df.groupby('cluster').mean()[cluster_conn_df.groupby('cluster').mean().conn4>(conn4_max/2.4)].index

# %%
# plot neurons with 3/4 connections

# all skids with 3-connections or 4-connections
conn3_skids = list(cluster_conn_df[cluster_conn_df.conn3>0].skid)
conn4_skids = list(cluster_conn_df[cluster_conn_df.conn4>0].skid)

conn3_ct = Celltype('conn3_neurons', conn3_skids)
conn4_ct = Celltype('conn4_neurons', conn4_skids)

# skids from clusters
conn3_clusters_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in conn3_clusters]
conn3_clusters_skids = [x for sublist in conn3_clusters_skids for x in sublist]
conn3_clusters_skids = np.intersect1d(conn3_clusters_skids, conn3_skids)

conn4_clusters_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in conn4_clusters]
conn4_clusters_skids = [x for sublist in conn4_clusters_skids for x in sublist]
conn4_clusters_skids = np.intersect1d(conn4_clusters_skids, conn4_skids)

conn3_clusters_ct = Celltype('conn3_clusters_neurons', conn3_clusters_skids)
conn4_clusters_ct = Celltype('conn4_clusters_neurons', conn4_clusters_skids)

conns_cta = Celltype_Analyzer([conn3_clusters_ct, conn4_clusters_ct, conn3_ct, conn4_ct])
conns_cta.set_known_types(celltypes)

conns_cta.plot_memberships('plots/conn3-conn4_celltypes.pdf', figsize=(3,3))

conns_df = conns_cta.memberships(raw_num=True)

# donut plot of aa cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(conns_df.loc[official_order, 'conn3_clusters_neurons (25)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_conn3_celltypes.pdf', format='pdf', bbox_inches='tight')

# donut plot of dd cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(conns_df.loc[official_order, 'conn4_clusters_neurons (21)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_conn4_celltypes.pdf', format='pdf', bbox_inches='tight')
