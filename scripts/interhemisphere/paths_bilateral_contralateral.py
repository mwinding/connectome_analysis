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

import cmasher as cmr

import connectome_tools.process_matrix as pm
import connectome_tools.cascade_analysis as casc

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

#adj = mg.adj  # adjacency matrix from the "mg" object
adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
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
        order_df.append([key, np.nanmean(node_visits)])

    order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
    order_df = order_df.sort_values(by = 'node_visit_order')

    return(lvl, list(order_df.cluster))

lvl7, order_7 = cluster_order('lvl7_labels', meta_with_order)
lvl6, order_6 = cluster_order('lvl6_labels', meta_with_order)
lvl5, order_5 = cluster_order('lvl5_labels', meta_with_order)
lvl4, order_4 = cluster_order('lvl4_labels', meta_with_order)

# %%
# load ipsi, bilateral, contra; set up adj matrix

import connectome_tools.process_matrix as pm

ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

# use for contra
adj_mat = pm.Adjacency_matrix(adj.values, adj.index, pairs, inputs, 'axo-dendritic')

# use for ipsi
# normal adj but removing all contra edges

# %%
# downstream of contralateral neurons
from tqdm import tqdm

contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)[0]
bi_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]
ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)[0]

threshold = 0.01
hops = 3

ds_contra_list = []
for i in tqdm(range(len(contra_pairs))):
    ds_contra = adj_mat.downstream_multihop(list(contra_pairs.loc[i]), threshold, min_members = 0, hops=hops, allow_source_ds=True)
    ds_contra_list.append(ds_contra)

ds_bi_list = []
for i in tqdm(range(len(bi_pairs))):
    ds_bi = adj_mat.downstream_multihop(list(bi_pairs.loc[i]), threshold, min_members = 0, hops=hops, allow_source_ds=True)
    ds_bi_list.append(ds_bi)

ds_ipsi_list = []
for i in tqdm(range(len(ipsi_pairs))):
    ds_ipsi = adj_mat.downstream_multihop(source = list(ipsi_pairs.loc[i]), threshold = threshold, min_members = 0, hops=hops, allow_source_ds=True)
    ds_ipsi_list.append(ds_ipsi)
'''
us_contra_list = []
for i in tqdm(range(len(contra_pairs))):
    us_contra = adj_mat.upstream(list(contra_pairs.loc[i]), threshold, min_members = 0, hops=hops)
    us_contra_list.append(us_contra)
'''
# %%
# group by cluster

def cluster_group(order, lvl, pairs):
    cluster_loc = []
    for i in range(len(order)):
        for j in range(len(pairs)):
            lvl_skids = list(lvl.groups[order[i]])
            if(len(np.intersect1d(list(pairs.loc[j]), lvl_skids))>0):
                cluster_loc.append([pairs.leftid.loc[j], order[i]])
    cluster_loc = pd.DataFrame(cluster_loc, columns = ['pairid', 'cluster'])
    return(cluster_loc)

cluster7_loc = cluster_group(order_7, lvl7, contra_pairs)
cluster6_loc = cluster_group(order_6, lvl6, contra_pairs)
cluster5_loc = cluster_group(order_5, lvl5, contra_pairs)
cluster4_loc = cluster_group(order_4, lvl4, contra_pairs)

contra_pairs['cluster_lvl7'] = cluster7_loc.cluster
contra_pairs['cluster_lvl6'] = cluster6_loc.cluster
contra_pairs['cluster_lvl5'] = cluster5_loc.cluster
contra_pairs['cluster_lvl4'] = cluster4_loc.cluster

celltypes, celltype_names = pm.Promat.celltypes([ipsi, bilateral, contra], ['ipsi', 'bilateral', 'contra'])

# identify types of neurons in each layer downstream of each contra-pair
contra_type_layers_list = []
contra_type_layers_skids_list = []
for celltype in celltypes:
    type_layers, type_layers_skids = adj_mat.layer_id(ds_contra_list, contra_pairs.leftid, celltype)
    contra_type_layers_list.append(type_layers)
    contra_type_layers_skids_list.append(type_layers_skids)

# plot 3-hops downstream of each contra neuron
layer_colors = ['Greens', 'Greens', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greens', 'Blues', 'Purples', 'Blues', 'Reds', 'Purples', 'Reds', 'Blues', 'Greens', 'Oranges']
layer_vmax = [200, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 50, 50, 500, 500, 500]
save_path = 'interhemisphere/plots/contra_downstream/'

adj_mat.plot_layer_types(layer_types=contra_type_layers_list, layer_names=celltype_names, layer_colors=layer_colors,
                        layer_vmax=layer_vmax, pair_ids=contra_pairs.leftid, figsize=(.5*hops/3, 1.5), save_path=save_path, threshold=0.01, hops=hops)
# %%
# combine paths of contra neurons in same cluster

contra_pairs_lvl = contra_pairs.set_index(['cluster_lvl7', 'leftid'], drop=False)

ds_contra_cluster_list = []
for cluster in np.unique(contra_pairs_lvl.loc[:, 'cluster_lvl7']):
    leftid = list(contra_pairs_lvl.loc[(cluster, slice(None)), :].leftid)
    cluster_list=[]
    for id in leftid:
        index = np.where(contra_pairs_lvl.leftid==id)[0][0]
        cluster_list.append(ds_contra_list[index])

    combined = []
    for i in range(len(cluster_list[0])):
        cluster_hop = [x[i] for x in cluster_list if len(x)>=(i+1)]
        cluster_hop = np.unique([x for sublist in cluster_hop for x in sublist])
        combined.append(cluster_hop)
    ds_contra_cluster_list.append(combined)

ds_contra_1o = []
for i, ds_contra in enumerate(ds_contra_cluster_list):
    ds_contra_1o.append(casc.Celltype(f'cluster{i}_contra', list(ds_contra[0])))

# general downstream partners
celltypes, celltype_names = pm.Promat.celltypes()
all_celltypes = list(map(lambda pair: casc.Celltype(pair[0], pair[1]), zip(celltype_names, celltypes)))
ds_contra_1o_analyzer = casc.Celltype_Analyzer(ds_contra_1o)
ds_contra_1o_analyzer.set_known_types(all_celltypes)
ds_contra_1o_memberships = ds_contra_1o_analyzer.memberships()
ds_contra_1o_memberships = ds_contra_1o_memberships[ds_contra_1o_memberships.sum(axis=1)!=0]

fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','KC','MB-FBN', 'CN', 'pre-dSEZ', 'pre-dVNC', 'RGN', 'dSEZ', 'dVNC', 'unknown']
colors = ['#1D79B7', '#D4E29E', '#FF9852', '#F9EB4D', 'black', '#C144BC', '#8C7700', '#77CDFC','#E0B1AD', 'tab:purple','#C47451', '#A52A2A', 'tab:grey']
ind = [x for x in range(0, len(ds_contra_1o_memberships.columns))]

fig, ax = plt.subplots(1,1, figsize=(1.5,2))

# summary plot of 1st order upstream of dVNCs
plts = []
plt1 = plt.bar(ind, ds_contra_1o_memberships.iloc[0,:], color = colors[0])
bottom = ds_contra_1o_memberships.iloc[0,:]
plts.append(plt1)

for i in range(1, len(ds_contra_1o_memberships.index)):
    plt1 = plt.bar(ind, ds_contra_1o_memberships.iloc[i,:], bottom = bottom, color = colors[i])
    bottom = bottom + ds_contra_1o_memberships.iloc[i,:]
    plts.append(plt1)

plt.savefig('interhemisphere/plots/summary_contra_partners_by_cluster.pdf', format='pdf', bbox_inches='tight')

# bilateral/contra/ipsi partners
celltypes = [ipsi, bilateral, contra]
celltype_names = ['ipsi', 'bilateral', 'contra']
all_celltypes = list(map(lambda pair: casc.Celltype(pair[0], pair[1]), zip(celltype_names, celltypes)))
ds_contra_1o_analyzer = casc.Celltype_Analyzer(ds_contra_1o)
ds_contra_1o_analyzer.set_known_types(all_celltypes)
ds_contra_1o_memberships = ds_contra_1o_analyzer.memberships()
ds_contra_1o_memberships = ds_contra_1o_memberships[ds_contra_1o_memberships.sum(axis=1)!=0]

fraction_types_names = ['ipsi', 'bilateral', 'contra', 'unknown']
colors = ['blue', 'orange', 'green','tab:grey']
ind = [x for x in range(0, len(ds_contra_1o_memberships.columns))]

fig, ax = plt.subplots(1,1, figsize=(1.5,2))

# summary plot of 1st order upstream of dVNCs
plts = []
plt1 = plt.bar(ind, ds_contra_1o_memberships.iloc[0,:], color = colors[0])
bottom = ds_contra_1o_memberships.iloc[0,:]
plts.append(plt1)

for i in range(1, len(ds_contra_1o_memberships.index)):
    plt1 = plt.bar(ind, ds_contra_1o_memberships.iloc[i,:], bottom = bottom, color = colors[i])
    bottom = bottom + ds_contra_1o_memberships.iloc[i,:]
    plts.append(plt1)

plt.savefig('interhemisphere/plots/summary_contra_partners_by_cluster_to_bi_contra.pdf', format='pdf', bbox_inches='tight')

# plot paths

# %%
# what fraction of ds-network is ipsi, contra, bilateral?

ds_total = [len(x[0]) for x in ds_contra_list]

ds_bilateral = [len(x) for x in contra_type_layers_skids_list[-1].loc[0]]
ds_contra = [len(x) for x in contra_type_layers_skids_list[-2].loc[0]]
ds_ipsi = [len(x) for x in contra_type_layers_skids_list[-3].loc[0]]
ds_ipsi_skids = [x for x in contra_type_layers_skids_list[-3].loc[0]]

frac_bilateral = np.array(ds_bilateral)/np.array(ds_total)
frac_contra = np.array(ds_contra)/np.array(ds_total)
frac_ipsi = np.array(ds_ipsi)/np.array(ds_total)

threshold = 0.01

# total number of partners upstream
ipsi_input_total = (adj_mat.adj_pairwise.loc[('pairs'), ('pairs', ipsi)]>threshold).sum(axis=0)
bilateral_input_total = (adj_mat.adj_pairwise.loc[('pairs'), ('pairs', bilateral)]>threshold).sum(axis=0)
contralateral_input_total = (adj_mat.adj_pairwise.loc[('pairs'), ('pairs', contra)]>threshold).sum(axis=0)

# number of numbers upstream
ipsi_ipsi = (adj_mat.adj_pairwise.loc[('pairs', ipsi), ('pairs', ipsi)]>threshold).sum(axis=0)
ipsi_bilateral = (adj_mat.adj_pairwise.loc[('pairs', ipsi), ('pairs', bilateral)]>threshold).sum(axis=0)
ipsi_contra = (adj_mat.adj_pairwise.loc[('pairs', ipsi), ('pairs', contra)]>threshold).sum(axis=0)

contra_ipsi = (adj_mat.adj_pairwise.loc[('pairs', contra), ('pairs', ipsi)]>threshold).sum(axis=0)
contra_bilateral = (adj_mat.adj_pairwise.loc[('pairs', contra), ('pairs', bilateral)]>threshold).sum(axis=0)
contra_contra = (adj_mat.adj_pairwise.loc[('pairs', contra), ('pairs', contra)]>threshold).sum(axis=0)

bi_ipsi = (adj_mat.adj_pairwise.loc[('pairs', bilateral), ('pairs', ipsi)]>threshold).sum(axis=0)
bi_bilateral = (adj_mat.adj_pairwise.loc[('pairs', bilateral), ('pairs', bilateral)]>threshold).sum(axis=0)
bi_contra = (adj_mat.adj_pairwise.loc[('pairs', bilateral), ('pairs', contra)]>threshold).sum(axis=0)
'''
data = pd.concat([pd.DataFrame(zip(ipsi_ipsi, ['ipsi-ipsi']*len(ipsi_ipsi)), columns = ['number_us', 'connection_type']), 
                    pd.DataFrame(zip(bi_ipsi, ['bilateral-ipsi']*len(bi_ipsi)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(contra_ipsi, ['contra-ipsi']*len(contra_contra)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(ipsi_bilateral, ['ipsi-bilateral']*len(ipsi_bilateral)), columns = ['number_us', 'connection_type']), 
                    pd.DataFrame(zip(bi_bilateral, ['bilateral-bilateral']*len(bi_bilateral)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(contra_bilateral, ['contra-bilateral']*len(contra_bilateral)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(ipsi_contra, ['ipsi-contra']*len(ipsi_contra)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(bi_contra, ['bilateral-contra']*len(bi_contra)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(contra_contra, ['contra-contra']*len(contra_contra)), columns = ['number_us', 'connection_type'])], 
                    axis=0)
'''
data = pd.concat([ pd.DataFrame(zip(bi_ipsi, ['bilateral-ipsi']*len(bi_ipsi)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(bi_bilateral, ['bilateral-bilateral']*len(bi_bilateral)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(bi_contra, ['bilateral-contra']*len(bi_contra)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(contra_ipsi, ['contra-ipsi']*len(contra_contra)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(contra_bilateral, ['contra-bilateral']*len(contra_bilateral)), columns = ['number_us', 'connection_type']),
                    pd.DataFrame(zip(contra_contra, ['contra-contra']*len(contra_contra)), columns = ['number_us', 'connection_type'])], 
                    axis=0)

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.boxplot(x='connection_type', y='number_us', data=data, fliersize=0.1, ax=ax, linewidth=0.5)
plt.xticks(rotation=45, ha='right')
plt.savefig('interhemisphere/plots/ipsi-contra-bilateral_interactions/number_upstream_partners.pdf', format='pdf', bbox_inches='tight')


# number of neurons downstream
ipsi_ipsi = (adj_mat.adj_pairwise.loc[('pairs', ipsi), ('pairs', ipsi)]>threshold).sum(axis=1)
ipsi_bilateral = (adj_mat.adj_pairwise.loc[('pairs', ipsi), ('pairs', bilateral)]>threshold).sum(axis=1)
ipsi_contra = (adj_mat.adj_pairwise.loc[('pairs', ipsi), ('pairs', contra)]>threshold).sum(axis=1)

contra_ipsi = (adj_mat.adj_pairwise.loc[('pairs', contra), ('pairs', ipsi)]>threshold).sum(axis=1)
contra_bilateral = (adj_mat.adj_pairwise.loc[('pairs', contra), ('pairs', bilateral)]>threshold).sum(axis=1)
contra_contra = (adj_mat.adj_pairwise.loc[('pairs', contra), ('pairs', contra)]>threshold).sum(axis=1)

bi_ipsi = (adj_mat.adj_pairwise.loc[('pairs', bilateral), ('pairs', ipsi)]>threshold).sum(axis=1)
bi_bilateral = (adj_mat.adj_pairwise.loc[('pairs', bilateral), ('pairs', bilateral)]>threshold).sum(axis=1)
bi_contra = (adj_mat.adj_pairwise.loc[('pairs', bilateral), ('pairs', contra)]>threshold).sum(axis=1)

data = pd.concat([pd.DataFrame(zip(ipsi_bilateral, ['ipsi-bilateral']*len(ipsi_bilateral)), columns = ['number_ds', 'connection_type']), 
                    pd.DataFrame(zip(bi_bilateral, ['bilateral-bilateral']*len(bi_bilateral)), columns = ['number_ds', 'connection_type']),
                    pd.DataFrame(zip(contra_bilateral, ['contra-bilateral']*len(contra_bilateral)), columns = ['number_ds', 'connection_type']),
                    pd.DataFrame(zip(ipsi_contra, ['ipsi-contra']*len(ipsi_contra)), columns = ['number_ds', 'connection_type']),
                    pd.DataFrame(zip(bi_contra, ['bilateral-contra']*len(bi_contra)), columns = ['number_ds', 'connection_type']),
                    pd.DataFrame(zip(contra_contra, ['contra-contra']*len(contra_contra)), columns = ['number_ds', 'connection_type'])], 
                    axis=0)

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.boxplot(x='connection_type', y='number_ds', data=data, fliersize=0.1, ax=ax, linewidth=0.5)
ax.set(ylim=(-1, 25))
plt.xticks(rotation=45, ha='right')
plt.savefig('interhemisphere/plots/ipsi-contra-bilateral_interactions/number_downstream_partners.pdf', format='pdf', bbox_inches='tight')

# fraction of inputs from ipsi/contra/bi
# fraction of outputs ipsi/contra/bi

# %%
# self-loops? contra? bilateral? vs ipsi
contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)[0]
bilateral_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]
ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)[0]

self_loop_contra = []
for i in range(len(contra_pairs)):
    self_loop_contra.append(adj_mat.adj_pairwise.loc[('pairs', contra_pairs.leftid[i]), ('pairs', contra_pairs.leftid[i])])

self_loop_bilateral = []
for i in range(len(bilateral_pairs)):
    self_loop_bilateral.append(adj_mat.adj_pairwise.loc[('pairs', bilateral_pairs.leftid[i]), ('pairs', bilateral_pairs.leftid[i])])

self_loop_ipsi = []
for i in range(len(ipsi_pairs)):
    self_loop_ipsi.append(adj_mat.adj_pairwise.loc[('pairs', bilateral_pairs.leftid[i]), ('pairs', bilateral_pairs.leftid[i])])

sum(np.array(self_loop_contra)>0.01)/len(self_loop_contra)
sum(np.array(self_loop_bilateral)>0.01)/len(self_loop_bilateral)


celltypes, celltype_names = pm.Promat.celltypes([ipsi, bilateral, contra], ['ipsi', 'bilateral', 'contra'])

# identify types of neurons in each layer downstream of each contra-pair
def self_loops(ds_source_list, source_pairs, type_cell):

    self_loop_hops = []
    for i, pair in enumerate(source_pairs.leftid):
        hops=[pair, 0, type_cell]
        for hop, ds in enumerate(ds_source_list[i]):
            if(pair in ds):
                hops=[pair, hop+1, type_cell]
    
        self_loop_hops.append(hops)

    self_loop_hops = pd.DataFrame(self_loop_hops, columns =['pairid', 'hops', 'type'])
    return(self_loop_hops)

# contra
contra_type_layers_list = []
contra_type_layers_skids_list = []
for celltype in celltypes:
    type_layers, type_layers_skids = adj_mat.layer_id(ds_contra_list, contra_pairs.leftid, celltype)
    contra_type_layers_list.append(type_layers)
    contra_type_layers_skids_list.append(type_layers_skids)

# for bilaterals
bi_type_layers_list = []
bi_type_layers_skids_list = []
for celltype in celltypes:
    type_layers, type_layers_skids = adj_mat.layer_id(ds_bi_list, bi_pairs.leftid, celltype)
    bi_type_layers_list.append(type_layers)
    bi_type_layers_skids_list.append(type_layers_skids)

# for ipsi
ipsi_type_layers_list = []
ipsi_type_layers_skids_list = []
for celltype in celltypes:
    type_layers, type_layers_skids = adj_mat.layer_id(ds_ipsi_list, ipsi_pairs.leftid, celltype)
    ipsi_type_layers_list.append(type_layers)
    ipsi_type_layers_skids_list.append(type_layers_skids)

contra_self_loops = self_loops(ds_contra_list, contra_pairs, 'contra')
bi_self_loops = self_loops(ds_bi_list, bi_pairs, 'bilateral')
ipsi_self_loops = self_loops(ds_ipsi_list, ipsi_pairs, 'ipsi')
#data = pd.concat([ipsi_self_loops, bi_self_loops, contra_self_loops], axis=0)

self_loop_hops = pd.DataFrame([[sum(ipsi_self_loops.hops==1)/len(ipsi_self_loops), sum(ipsi_self_loops.hops==2)/len(ipsi_self_loops)],
                    [sum(bi_self_loops.hops==1)/len(bi_self_loops), sum(bi_self_loops.hops==2)/len(bi_self_loops)],
                    [sum(contra_self_loops.hops==1)/len(contra_self_loops), sum(contra_self_loops.hops==2)/len(contra_self_loops)]],
                    columns = ['Direct', '2-Hop'], index = ['Ipsilateral', 'Bilateral', 'Contralateral'])

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.heatmap(self_loop_hops, annot=True, fmt='.2', ax=ax, cbar=False, vmax=0.20, cmap='Blues')
plt.savefig('interhemisphere/plots/self_loops.pdf', format='pdf', bbox_inches='tight')

# %%
# simple 1-hop upstream, 1-hop downstream of ipsi, bilateral, contra



# %%
# multihop matrix of contra -> contra, contra->bilateral, bilateral->bilateral, bilateral->contra

contra_contra_layers, contra_contra_skids = adj_mat.layer_id(ds_contra_list, contra_pairs.leftid, contra)

contra_contra_skids = contra_contra_skids.T
contra_contra_mat, contra_contra_mat_plotting = adj_mat.hop_matrix(contra_contra_skids, contra_pairs.leftid, contra_pairs.leftid)



 # %%
# rich club

binary_mat_input = (adj_mat.adj_pairwise.loc['pairs', 'pairs']>threshold).sum(axis=1)
binary_mat_output = (adj_mat.adj_pairwise.loc['pairs', 'pairs']>threshold).sum(axis=0)

data = pd.DataFrame(binary_mat_input, index=binary_mat_output.index, columns=['input_counts'])
data['output_counts'] =  binary_mat_output
data['sum_degree'] = data.input_counts + data.output_counts
celltype_skid = []
for skid in data.index:
    if(skid in ipsi):
        celltype_skid.append('ipsilateral')
    if(skid in bilateral):
        celltype_skid.append('bilateral')
    if(skid in contra):
        celltype_skid.append('contralateral')
    if((skid not in ipsi) & (skid not in bilateral) & (skid not in contra)):
        celltype_skid.append('unknown')

data['celltype'] = celltype_skid

sns.scatterplot(x='input_counts', y='output_counts', data=data, hue='celltype')
# %%
