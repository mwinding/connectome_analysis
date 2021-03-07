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
import numpy.random as random
import networkx as nx

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import networkx as nx
from tqdm import tqdm

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

adj_ad = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj_aa = pd.read_csv('VNC_interaction/data/brA1_axon-axon.csv', header = 0, index_col = 0)
adj_da = pd.read_csv('VNC_interaction/data/brA1_dendrite-axon.csv', header = 0, index_col = 0)
adj_dd = pd.read_csv('VNC_interaction/data/brA1_dendrite-dendrite.csv', header = 0, index_col = 0)

adj_type = 'ad'
adj = adj_ad
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

# remove A1 except for ascendings
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

# load inputs and pair data
inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs
pairs.drop(1121, inplace=True) # remove duplicate rightid

adj_mat = pm.Adjacency_matrix(adj.values, adj.index, pairs, inputs, adj_type)

# load previously generated paths
all_edges_combined = pd.read_csv(f'network_analysis/csv/{adj_type}_all_paired_edges.csv', index_col=0)

# flip contralateral axon/contralateral dendrite neurons so they act as ipsilateral
contra_contras = np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), 
                                pymaid.get_skids_by_annotation('mw contralateral dendrite'))
all_edges_combined.loc[[x in contra_contras for x in all_edges_combined.upstream_pair_id], 'type'] = ['ipsilateral']*sum([x in contra_contras for x in all_edges_combined.upstream_pair_id])

# build networkx Graph
graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')

# %%
# neuron degree, in-degree, out-degree
out_degree = all_edges_combined.groupby('upstream_pair_id').count().downstream_pair_id
in_degree = all_edges_combined.groupby('downstream_pair_id').count().upstream_pair_id

all_skids = np.unique(list(out_degree.index) + list(in_degree.index))

in_degree_missing = []
out_degree_missing = []
for skid in all_skids:
    if((skid in out_degree) & (skid in in_degree)):
        continue
    if((skid in out_degree) & (skid not in in_degree)):
        in_degree_missing.append((skid, 0))
    if((skid not in out_degree) & (skid in in_degree)):
        out_degree_missing.append((skid, 0))

out_degree = list(zip(out_degree.index, out_degree.values)) + out_degree_missing
in_degree = list(zip(in_degree.index, in_degree.values)) + in_degree_missing

def take_first(x):
    return(x[0])

in_degree = sorted(in_degree, key=take_first)
out_degree = sorted(out_degree, key=take_first)

# degrees include ipsi and contra connections as separate
degree = pd.DataFrame(in_degree, columns=['skid', 'in_degree'])
degree['out_degree'] = [x[1] for x in out_degree]
degree['total_degree'] = degree.in_degree + degree.out_degree

# node to node degrees
neurons = pd.DataFrame(graph.G.in_degree, columns=['skid', 'in_degree'])
neurons['out_degree'] = [x[1] for x in graph.G.out_degree]
neurons['total_degree'] = neurons.in_degree + neurons.out_degree

sns.scatterplot(data=neurons, x='in_degree', y='out_degree')

fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.distplot(neurons.total_degree, kde=False, ax=ax)
ax.set_yscale('log')

#pymaid.add_annotations(list(neurons[neurons.total_degree>50].skid), 'mw degree >50')
#pymaid.add_annotations(list(neurons[(neurons.total_degree>40) & (neurons.total_degree<=50)].skid), 'mw degree 40-50')
#pymaid.add_annotations(list(neurons[(neurons.total_degree>30) & (neurons.total_degree<=40)].skid), 'mw degree 30-40')

# %%
# betweenness centrality

centrality = nx.betweenness_centrality(graph.G, k=1000)
neurons['betweenness_centrality'] = [x[1] for x in centrality.items()]

sns.jointplot(data = neurons, x='total_degree', y='betweenness_centrality', s=4, alpha = 0.5)
#pymaid.add_annotations(list(neurons[neurons.betweenness_centrality>0.01].skid), 'mw betweenness_centrality >0.1')

# %%
# participation coefficients for clusters
from joblib import Parallel, delayed

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

def participation_coeff(neuron, in_degree, out_degree, total_degree, clusters, cluster_edges):

    input_part_coeff = 1
    output_part_coeff = 1
    part_coeff = 1

    for i, cluster_skids in enumerate(clusters.skids):
        output_edges = [x for x in cluster_edges[i] if (x[1] in cluster_skids) & (x[0]==neuron)]
        input_edges = [x for x in cluster_edges[i] if (x[0] in cluster_skids) & (x[1]==neuron)]

        if(in_degree>0):
            input_frac = len(input_edges)/in_degree
            input_part_coeff = input_part_coeff - input_frac**2
        if(out_degree>0):
            output_frac = len(output_edges)/out_degree
            output_part_coeff = output_part_coeff - output_frac**2
        if(total_degree>0):
            total_frac = (len(output_edges) + len(input_edges))/total_degree
            part_coeff = part_coeff - total_frac**2

        if(in_degree==0):
            input_part_coeff=np.nan
        if(out_degree==0):
            output_part_coeff=np.nan
        if(total_degree==0):
            part_coeff=np.nan

    return(input_part_coeff, output_part_coeff, part_coeff)

# generate list of edges associated with each cluster to make edge searches more efficient
cluster_edges = []
for cluster_skids in lvl5.skids:
    edges = [x for x in graph.G.edges if (x[1] in cluster_skids) | (x[0] in cluster_skids)]
    cluster_edges.append(edges)

participation_coeff = Parallel(n_jobs=-1)(delayed(participation_coeff)(neurons.skid[i], neurons.in_degree[i], neurons.out_degree[i], neurons.total_degree[i], lvl5, cluster_edges) for i in tqdm(range(0,len(neurons.skid))))
participation_coeff_df = pd.DataFrame(participation_coeff, columns = ['part_in', 'part_out', 'part'])
participation_coeff_df.columns = ['part_in', 'part_out', 'part']
#participation_coeff_df.to_csv('network_analysis/csv/participation_coefficients.csv')

neurons['part_in'] = participation_coeff_df.part_in
neurons['part_out'] = participation_coeff_df.part_out
neurons['part'] = participation_coeff_df.part

# %%
# within module score

# calculate average inner-module edges for all neurons within a module; and standard deviation
within_module_in_scores = []
within_module_out_scores = []
within_module_total_scores = []

for i, cluster_skids in enumerate(lvl5.skids):

    within_module_in_score = []
    within_module_out_score = []
    within_module_total_score = []

    for skid in cluster_skids:

        output_edges = [x for x in cluster_edges[i] if (x[1] in cluster_skids) & (x[0]==skid)]
        input_edges = [x for x in cluster_edges[i] if (x[0] in cluster_skids) & (x[1]==skid)]

        within_module_in_score.append(len(input_edges))
        within_module_out_score.append(len(output_edges))
        within_module_total_score.append(len(output_edges) + len(input_edges))
    
    within_module_in_scores.append(within_module_in_score)
    within_module_out_scores.append(within_module_out_score)
    within_module_total_scores.append(within_module_total_score)

within_module_out_mean = [np.mean(x) for x in within_module_out_scores]
within_module_out_std = [np.std(x) for x in within_module_out_scores]

within_module_in_mean = [np.mean(x) for x in within_module_in_scores]
within_module_in_std = [np.std(x) for x in within_module_in_scores]

within_module_mean = [np.mean(x) for x in within_module_total_scores]
within_module_std = [np.std(x) for x in within_module_total_scores]

# calculate within module z-store for each neuron

def within_zscore(neuron, clustered_neurons, lvl5, cluster_edges, zscore_stats):

    within_module_in_mean = zscore_stats[0]
    within_module_in_std = zscore_stats[1]
    within_module_out_mean = zscore_stats[2] 
    within_module_out_std = zscore_stats[3] 
    within_module_mean = zscore_stats[4] 
    within_module_std = zscore_stats[5] 

    # give neurons zscore = np.nan if not in a cluster
    if(neuron not in clustered_neurons):
        return(np.nan, np.nan, np.nan)

    i = np.where([neuron in x for x in lvl5.skids])[0][0]

    output_edges = [x for x in cluster_edges[i] if (x[1] in lvl5.skids[i]) & (x[0]==neuron)]
    input_edges = [x for x in cluster_edges[i] if (x[0] in lvl5.skids[i]) & (x[1]==neuron)]

    in_zscore = np.nan
    out_zscore = np.nan
    total_zscore = np.nan

    if(within_module_in_std[i]>0):
        in_zscore = (len(input_edges)-within_module_in_mean[i])/within_module_in_std[i]
    if(within_module_out_std[i]>0):
        out_zscore = (len(output_edges)-within_module_out_mean[i])/within_module_out_std[i]
    if(within_module_std[i]>0):
        total_zscore = ((len(output_edges) + len(input_edges))-within_module_mean[i])/within_module_std[i]

    return(in_zscore, out_zscore, total_zscore)

clustered_neurons = [x for sublist in lvl5.skids for x in sublist]
zscore_stats = [within_module_in_mean, within_module_in_std, within_module_out_mean, within_module_out_std, within_module_mean, within_module_std]

zscores = Parallel(n_jobs=-1)(delayed(within_zscore)(neurons.skid[i], clustered_neurons, lvl5, cluster_edges, zscore_stats) for i in tqdm(range(0,len(neurons.skid))))

neurons['within_zscore'] = [x[2] for x in zscores]

# %%
# identify interesting neurons

#neurons.to_csv('network_analysis/csv/neuron_properties.csv')
neurons = pd.read_csv('network_analysis/csv/neuron_properties.csv', header=0, index_col=0)

'''
types=[]
for neuron in neurons.skid:
    values = neurons[neurons.skid==neuron]
    type_ = 'none'
    if((values.total_degree.values[0]>30) & (values.betweenness_centrality.values[0]>0.01)):
        type_ = 'hub'
    if((values.total_degree.values[0]<=30) & (values.betweenness_centrality.values[0]>0.01)):
        type_ = 'relay'

    types.append(type_)
'''

types=[]
hubs=[]
for neuron in neurons.skid:
    values = neurons[neurons.skid==neuron]
    type_ = 'none'
    hub = 0

    if((values.within_zscore.values[0]==np.nan) | (values.part.values[0]==np.nan)):
        types.append(type_)
        hubs.append(hub)
        continue

    if((values.within_zscore.values[0]>=2.5) & (values.part.values[0]<=0.3)):
        type_ = 'provincal hub'
        hub = 1
    if((values.within_zscore.values[0]>=2.5) & (values.part.values[0]>0.3) & (values.part.values[0]<=0.75)):
        type_ = 'connector hub'
        hub = 1
    if((values.within_zscore.values[0]>=2.5) & (values.part.values[0]>0.75)):
        type_ = 'kinless hub'
        hub = 1

    if((values.within_zscore.values[0]<2.5) & (values.part.values[0]<=0.05)):
        type_ = 'ultra-peripheral node'
        hub = 0
    if((values.within_zscore.values[0]<2.5) & (values.part.values[0]>0.05) & (values.part.values[0]<=0.62)):
        type_ = 'peripheral node'
        hub = 0
    if((values.within_zscore.values[0]<2.5) & (values.part.values[0]>0.62) & (values.part.values[0]<=0.8)):
        type_ = 'non-hub connector node'
        hub = 0
    if((values.within_zscore.values[0]<2.5) & (values.part.values[0]>0.8)):
        type_ = 'kinless connector node'
        hub = 0

    types.append(type_)
    hubs.append(hub)

neurons['types'] = types
neurons['hub'] = hubs

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'arial'

g = sns.pairplot(neurons.loc[:, ['total_degree', 'betweenness_centrality', 'within_zscore', 'part', 'types']], hue='types', 
                            hue_order = ['connector hub', 'provincal hub', 'kinless hub', 'ultra-peripheral node', 'peripheral node', 'non-hub connector node', 'kinless connector node'], 
                            diag_kind='hist', diag_kws = {'alpha':0.55, 'bins':40}, plot_kws=dict(edgecolor='none', alpha=1.0, s=5))
g = sns.pairplot(neurons.loc[:, ['total_degree', 'betweenness_centrality', 'within_zscore', 'part', 'hub']], hue='hub', 
                            diag_kind='hist', diag_kws = {'alpha':0.55, 'bins':10}, plot_kws=dict(edgecolor='none', alpha=1.0, s=5))
plt.savefig('network_analysis/plots/hubs_neuron-metrics.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
g = sns.scatterplot(data = neurons.loc[:, ['total_degree', 'betweenness_centrality', 'within_zscore', 'part', 'types']], 
                x='part', y='within_zscore', hue='types', hue_order = ['connector hub', 'provincal hub', 'kinless hub', 'ultra-peripheral node', 'peripheral node', 'non-hub connector node', 'kinless connector node'], 
                s=8, alpha = 1.0, edgecolor='none', ax=ax)
g.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
g.set(xlabel='participation coefficient')
plt.savefig('network_analysis/plots/within-mod-zscore_vs_participation-coefficient.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
g = sns.scatterplot(data = neurons.loc[:, ['total_degree', 'betweenness_centrality', 'within_zscore', 'part', 'types']], 
                x='part', y='within_zscore',
                s=8, alpha = 1.0, edgecolor='none', ax=ax)
g.set(xlabel='participation coefficient')
plt.savefig('network_analysis/plots/within-mod-zscore_vs_participation-coefficient_no_hubs.pdf', format='pdf', bbox_inches='tight')

#%%

connector_hubs = neurons[neurons.hub==1].skid

hubs_in_cluster = []
for hub in connector_hubs:
    if(hub not in clustered_neurons):
        hubs_in_cluster.append(np.nan)
        continue
    i = np.where([hub in x for x in lvl5.skids])[0][0]
    hubs_in_cluster.append(i)

sorted(hubs_in_cluster)

# %%
# hubs based on degree

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.scatterplot(data = neurons, x='in_degree', y='out_degree', s=6, alpha = 0.8, edgecolor='none', ax=ax)
ax.set(xlim=(-1, 140), ylim=(-1, 140), xticks=([0, 20, 40, 60, 80, 100, 120, 140]))
plt.axvline(x=20, color='grey', linewidth=0.5)
plt.axhline(y=20, color='grey', linewidth=0.5)
plt.savefig('network_analysis/plots/in_out_degree.pdf', format='pdf', bbox_inches='tight')

g = sns.jointplot(data=neurons, x='in_degree', y='out_degree', xlim=(-1, 140), ylim=(-1, 140), s=4, alpha = 1.0)
g.ax_joint.axvline(x=20, color='grey', linewidth=0.5)
g.ax_joint.axhline(y=20, color='grey', linewidth=0.5)
g.ax_marg_x.axvline(x=20, color='grey', linewidth=0.5)
g.ax_marg_y.axhline(y=20, color='grey', linewidth=0.5)
plt.savefig('network_analysis/plots/in_out_degree_joint.pdf', format='pdf', bbox_inches='tight')

types = []
hubs = []
for skid in neurons.skid:
    values = neurons[neurons.skid==skid]
    type_ = 'none'
    hub = 0

    if((values.in_degree.values[0]>20) & (values.out_degree.values[0]>20)):
        type_ = 'in-out_hub'
        hub = 1
    if((values.in_degree.values[0]>20) & (values.out_degree.values[0]<=20)):
        type_ = 'in_hub'
        hub = 1
    if((values.in_degree.values[0]<=20) & (values.out_degree.values[0]>20)):
        type_ = 'out_hub'
        hub = 1
    
    types.append(type_)
    hubs.append(hub)

neurons['types'] = types
neurons['hub'] = hubs

g = sns.jointplot(data=neurons, x='in_degree', y='out_degree', hue='types', xlim=(-1, 140), ylim=(-1, 140), s=4, edgecolor='none', alpha = 1.0)
g.plot_marginals(sns.histplot, fill=None)
g.ax_joint.axvline(x=20, color='grey', linewidth=0.5)
g.ax_joint.axhline(y=20, color='grey', linewidth=0.5)
g.ax_marg_x.axvline(x=20, color='grey', linewidth=0.5)
g.ax_marg_y.axhline(y=20, color='grey', linewidth=0.5)
plt.savefig('network_analysis/plots/in_out_degree_joint_hue.pdf', format='pdf', bbox_inches='tight')

g = sns.pairplot(neurons.loc[:, ['total_degree', 'betweenness_centrality', 'within_zscore', 'part', 'types']], hue='types', markers=['.', '^', 'v', 'X'], 
                            diag_kind='hist', diag_kws = {'alpha':0.55, 'bins':20}, plot_kws=dict(edgecolor='none', alpha=1.0, s=20))
plt.savefig('network_analysis/plots/hubs_metrics.pdf', format='pdf', bbox_inches='tight')

# add a couple more metrics

# %%
# where are hubs in clusters?

lvl = lvl7
order = order_7

cluster_ids = []
for skid in neurons.skid:
    cluster_id='none'
    for i, cluster_skids in enumerate(lvl.skids):
        if(skid in cluster_skids):
            cluster_id = lvl.iloc[i].cluster
    
    cluster_ids.append(cluster_id)

neurons['cluster']=cluster_ids

neurons.groupby(['cluster', 'types']).count()

hubs_in_clusters = pd.DataFrame(neurons.pivot_table(index='types',
                                        columns = 'cluster',
                                        values='hub',
                                        fill_value=0, 
                                        aggfunc='count').unstack(), columns = ['counts'])

total_in_clusters = pd.DataFrame(neurons.pivot_table(columns='cluster',
                                        values='types',
                                        fill_value=0, 
                                        aggfunc='count').unstack(), columns = ['counts'])

for cluster in order:
    hubs_in_clusters.loc[cluster, 'counts'] = (hubs_in_clusters.loc[cluster, :]/total_in_clusters.loc[cluster].values[0][0]).values

hubs_in_clusters.loc['none', 'counts'] = (hubs_in_clusters.loc['none', 'counts']/sum(hubs_in_clusters.loc['none', 'counts'])).values
#hubs_in_clusters.reset_index(inplace=True)
ind = [x for x in range(0, len(total_in_clusters))]

hub_types = np.unique([x[1] for x in hubs_in_clusters.index])
hubs_in_clusters_plot = [hubs_in_clusters.loc[(slice(None), x), :] for x in hub_types]

plt.bar(ind, [x[0] for x in hubs_in_clusters_plot[0].values], color = sns.color_palette()[3])
bottom = np.array([x[0] for x in hubs_in_clusters_plot[0].values])

plt.bar(ind, [x[0] for x in hubs_in_clusters_plot[1].values], bottom = bottom, color = sns.color_palette()[1])
bottom = bottom + np.array([x[0] for x in hubs_in_clusters_plot[1].values])

plt.bar(ind, [x[0] for x in hubs_in_clusters_plot[3].values], bottom = bottom, color = sns.color_palette()[2])
bottom = bottom + np.array([x[0] for x in hubs_in_clusters_plot[3].values])

plt.bar(ind, [x[0] for x in hubs_in_clusters_plot[2].values], bottom = bottom, color = '#ACAAC8')

# %%
# cell-type identity of hubs

# %%
# local vs connector hubs?

