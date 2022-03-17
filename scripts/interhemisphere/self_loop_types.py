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

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import networkx as nx 


# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)
MBON = pymaid.get_skids_by_annotation('mw MBON')

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
pairs.drop(1121, inplace=True) # remove duplicate rightid
ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)[0]
bi_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]
contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)[0]

# %%
# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csvs/all_paired_edges.csv', index_col=0)

# %%
# load into network x object

dVNC_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs)
dSEZ_pairs = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs)
RGN_pairs = pm.Promat.load_pairs_from_annotation('mw RGN', pairs)
ipsi_pairs = pm.Promat.load_pairs_from_annotation('mw ipsilateral axon', pairs).leftid
bilateral_pairs = pm.Promat.load_pairs_from_annotation('mw bilateral axon', pairs).leftid
contra_pairs = pm.Promat.load_pairs_from_annotation('mw contralateral axon', pairs).leftid

# build directed networkx graph
graph = gp.Analyze_Nx_G(all_edges_combined)

# make sure pair_ids are in G
ipsi_pairs = np.intersect1d(ipsi_pairs, graph.G.nodes)
bilateral_pairs = np.intersect1d(bilateral_pairs, graph.G.nodes)
contra_pairs = np.intersect1d(contra_pairs, graph.G.nodes)
#%%
# generate self loop paths

cutoff = 3

ipsi_paths = [graph.all_simple_self_loop_paths(pair_id, cutoff) for pair_id in ipsi_pairs]
bi_paths = [graph.all_simple_self_loop_paths(pair_id, cutoff) for pair_id in bilateral_pairs]
contra_paths = [graph.all_simple_self_loop_paths(pair_id, cutoff) for pair_id in contra_pairs]

paths = [ipsi_paths, bi_paths, contra_paths]
paths_names = ['ipsi', 'bilateral', 'contralateral']
pair_types = [ipsi_pairs, bilateral_pairs, contra_pairs]

paths_length = []
for i, paths_list in enumerate(paths):
    for j, path in enumerate(paths_list):
        if(len(path)==0):
            paths_length.append([paths_names[i], pair_types[i][j], 0, 'none'])
        if(len(path)>0):
            for subpath in path:
                edge_types = path_edge_attributes(G, subpath, 'edge_type', include_skids=False)
                if((sum(edge_types=='contralateral')%2)==0): # if there is an even number of contralateral edges
                    paths_length.append([paths_names[i], pair_types[i][j], len(subpath)-1, 'self'])
                if((sum(edge_types=='contralateral')%2)==1): # if there is an odd number of contralateral edges
                    paths_length.append([paths_names[i], pair_types[i][j], len(subpath)-1, 'pair'])

paths_length = pd.DataFrame(paths_length, columns = ['neuron_type', 'skid', 'path_length', 'loop_type'])
loop_type_counts = paths_length.groupby(['neuron_type', 'skid', 'path_length', 'loop_type']).size()
loop_type_counts.loc[('contralateral', 19298625, 1, 'self')]=False # added in this fake data so that the appropriate row appears with a 0.0 (instead of not appearing at all)
loop_type_counts.loc[('ipsi', 40045, 1, 'pair')]=False # same as above
loop_type_counts = loop_type_counts>0
total_loop_types = loop_type_counts.groupby(['neuron_type', 'path_length','loop_type']).sum()
total_loop_types.loc['bilateral'] = (total_loop_types.loc['bilateral']/len(bilateral_pairs)).values
total_loop_types.loc['contralateral'] = (total_loop_types.loc['contralateral']/len(contra_pairs)).values
total_loop_types.loc['ipsi'] = (total_loop_types.loc['ipsi']/len(ipsi_pairs)).values

self_loops = total_loop_types.loc[(slice(None), [1,2], 'self')]
pair_loops = total_loop_types.loc[(slice(None), [1,2], 'pair')]


data_self = pd.DataFrame([total_loop_types.loc[('ipsi', slice(None), 'self')].values, 
                            total_loop_types.loc[('bilateral', slice(None), 'self')].values, 
                            total_loop_types.loc[('contralateral', slice(None), 'self')].values], 
                            index=paths_names, columns=['Direct', '2-Hops', '3-Hops'])

data_pair = pd.DataFrame([total_loop_types.loc[('ipsi', slice(None), 'pair')].values, 
                            total_loop_types.loc[('bilateral', slice(None), 'pair')].values, 
                            total_loop_types.loc[('contralateral', slice(None), 'pair')].values], 
                            index=paths_names, columns=['Direct', '2-Hops', '3-Hops'])

no_loops = pd.DataFrame([total_loop_types.loc[('ipsi', slice(None), 'none')].values, 
                            total_loop_types.loc[('bilateral', slice(None), 'none')].values, 
                            total_loop_types.loc[('contralateral', slice(None), 'none')].values], 
                            index=paths_names, columns=[''])

fig, axs = plt.subplots(1,2, figsize=(2,.75), sharey=True)

ax = axs[0]
sns.heatmap(data_self, annot=True, fmt = '.1%', cmap='Blues', ax=ax, cbar=False, vmax=0.3)
ax.set(title='Self-Loop Prevalence')
plt.yticks(rotation=45)

ax = axs[1]
sns.heatmap(data_pair, annot=True, fmt = '.1%', cmap='Purples', ax=ax, cbar=False, vmax=0.4)
ax.set(title='Pair-Loop Prevalence')
plt.savefig('interhemisphere/plots/interhemisphere_crossings/self_loops.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(2/6,.75), sharey=True)
sns.heatmap(1-no_loops, annot=True, fmt = '.1%', cmap='Greens', ax=ax, cbar=False, vmax=.6, vmin=.2)
ax.set(title='Displays Loop in 3-Hops')
plt.savefig('interhemisphere/plots/interhemisphere_crossings/loops.pdf', format='pdf', bbox_inches='tight')


# without contra edges

paths_length_no_contra = []
for i, paths_list in enumerate(paths):
    for j, path in enumerate(paths_list):
        if(len(path)==0):
            paths_length_no_contra.append([paths_names[i], pair_types[i][j], 0, 'none'])
        if(len(path)>0):
            for subpath in path:
                edge_types = path_edge_attributes(G, subpath, 'edge_type', include_skids=False)
                if (sum(edge_types=='contralateral')>0):
                    paths_length_no_contra.append([paths_names[i], pair_types[i][j], 0, 'none'])
                if(sum(edge_types=='contralateral')==0): 
                    paths_length_no_contra.append([paths_names[i], pair_types[i][j], len(subpath)-1, 'self'])

paths_length_no_contra = pd.DataFrame(paths_length_no_contra, columns = ['neuron_type', 'skid', 'path_length', 'loop_type'])
loop_type_counts = paths_length_no_contra.groupby(['neuron_type', 'skid', 'path_length', 'loop_type']).size()
loop_type_counts.loc[('contralateral', 19298625, 1, 'self')]=False # added in this fake data so that the appropriate row appears with a 0.0 (instead of not appearing at all)
loop_type_counts.loc[('contralateral', 19298625, 2, 'self')]=False # added in this fake data so that the appropriate row appears with a 0.0 (instead of not appearing at all)
loop_type_counts.loc[('contralateral', 19298625, 3, 'self')]=False # added in this fake data so that the appropriate row appears with a 0.0 (instead of not appearing at all)
loop_type_counts = loop_type_counts>0
total_loop_types_no_contra = loop_type_counts.groupby(['neuron_type', 'path_length','loop_type']).sum()
total_loop_types_no_contra.loc['bilateral'] = (total_loop_types_no_contra.loc['bilateral']/len(bilateral_pairs)).values
total_loop_types_no_contra.loc['contralateral'] = (total_loop_types_no_contra.loc['contralateral']/len(contra_pairs)).values
total_loop_types_no_contra.loc['ipsi'] = (total_loop_types_no_contra.loc['ipsi']/len(ipsi_pairs)).values

data_self = pd.DataFrame([total_loop_types_no_contra.loc[('ipsi', slice(None), 'self')].values, 
                            total_loop_types_no_contra.loc[('bilateral', slice(None), 'self')].values, 
                            total_loop_types_no_contra.loc[('contralateral', slice(None), 'self')].values], 
                            index=paths_names, columns=['Direct', '2-Hops', '3-Hops'])

no_loops_no_contra = pd.DataFrame([total_loop_types_no_contra.loc[('ipsi', slice(None), 'none')].values, 
                            total_loop_types_no_contra.loc[('bilateral', slice(None), 'none')].values, 
                            total_loop_types_no_contra.loc[('contralateral', slice(None), 'none')].values], 
                            index=paths_names, columns=[''])

fig, ax = plt.subplots(1,1, figsize=(1,.75), sharey=True)
sns.heatmap(data_self, annot=True, fmt = '.1%', cmap='Blues', ax=ax, cbar=False, vmax=0.3)
ax.set(title='Self-Loop Prevalence')
plt.yticks(rotation=45)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/self_loops_without_contra_edges.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(2/6,.75), sharey=True)
sns.heatmap(1-no_loops_no_contra, annot=True, fmt = '.1%', cmap='Greens', ax=ax, cbar=False, vmax=.6, vmin=.2)
ax.set(title='Displays Loop in 3-Hops')
plt.savefig('interhemisphere/plots/interhemisphere_crossings/loops_without_contra_edges.pdf', format='pdf', bbox_inches='tight')

#%%
# plot loops with or without contra edges, bilateral-contra edges, all interhemisphere edges

loops_no_contra = pd.DataFrame([[1-total_loop_types_no_contra.loc[('ipsi', slice(None), 'none')].values[0], 'no_inter', 'ipsi'], 
                            [1-total_loop_types_no_contra.loc[('bilateral', slice(None), 'none')].values[0], 'no_inter', 'bilateral'], 
                            [1-total_loop_types_no_contra.loc[('contralateral', slice(None), 'none')].values[0], 'no_inter', 'contralateral']], 
                            index=paths_names, columns=['loop_fraction', 'condition', 'cell_type'])

loops = pd.DataFrame([[1-total_loop_types.loc[('ipsi', slice(None), 'none')].values[0], 'control', 'ipsi'], 
                            [1-total_loop_types.loc[('bilateral', slice(None), 'none')].values[0], 'control', 'bilateral'], 
                            [1-total_loop_types.loc[('contralateral', slice(None), 'none')].values[0], 'control', 'contralateral']], 
                            index=paths_names, columns=['loop_fraction', 'condition', 'cell_type'])

data = pd.concat([loops, loops_no_contra], axis=0)

height = 1.5
width = 1.5
sns.catplot(data=data, x='cell_type', y='loop_fraction', hue='condition', kind='bar', height=height, aspect=width/height)
plt.xticks(rotation=45, ha='right')
plt.savefig('interhemisphere/plots/interhemisphere_crossings/loops_without_contra_edges_barplot.pdf', format='pdf', bbox_inches='tight')
# %%
# outdated 

'''
def two_hop_edges(pairs_list, threshold, adj_mat):

    all_edges_list=[]
    for pair_id in tqdm(pairs_list):
        _, ds, ds_edges = adj_mat.downstream(pair_id, threshold)
        ds_edges, _ = adj_mat.edge_threshold(ds_edges, threshold, 'downstream')
        initial_overthres_ds_edges = ds_edges[ds_edges.overthres==True]
        initial_overthres_ds_edges.reset_index(inplace=True)
        ds_partners = initial_overthres_ds_edges.downstream_pair_id

        all_edges = []
        for i, partner in enumerate(ds_partners):
            _, ds, edges = adj_mat.downstream(partner, threshold)
            ds_edges, ds_partners = adj_mat.edge_threshold(edges, threshold, 'downstream')
            overthres_ds_edges = ds_edges[ds_edges.overthres==True]
            overthres_ds_edges.reset_index(inplace=True)

            for j in range(len(overthres_ds_edges)):
                path = pd.concat([initial_overthres_ds_edges.iloc[i, :], overthres_ds_edges.iloc[j, :]], axis=1).T
                path.reset_index(inplace=True)
                path = path.drop(labels=['index', 'level_0', 'overthres'], axis=1)
                path['path'] = f'path-{pair_id}_{i}_{j}'
                all_edges.append(path)
        if(len(all_edges)>0):
            all_edges = pd.concat(all_edges, axis=0)
            all_edges.set_index(['path'], inplace=True)
        all_edges_list.append(all_edges)

    return(all_edges_list)

threshold = 0.01

#CNs = pymaid.get_skids_by_annotation('mw CN')
#N_pairs = pm.Promat.extract_pairs_from_list(CNs, pairs)[0].iloc[0:5, :]
#contra_edges = two_hop_edges(CN_pairs.leftid, threshold, adj_mat)
contra_edges = two_hop_edges(contra_pairs.leftid, threshold, adj_mat)
contra_edges_combined = [x for x in contra_edges if type(x)==pd.DataFrame]
contra_edges_combined = pd.concat(contra_edges_combined, axis=0)
'''
# %%
adj_mat = pm.Adjacency_matrix(adj.values, adj.index, pairs, inputs, 'axo-dendritic')

KCs = list(pd.read_json('interhemisphere/data/KC-2020-01-14.json').skeleton_id)
left = list(pd.read_json('interhemisphere/data/hemisphere-L-2020-3-9.json').skeleton_id)
right = list(pd.read_json('interhemisphere/data/hemisphere-R-2020-3-9.json').skeleton_id)
MBON = list(pd.read_json('interhemisphere/data/MBON-2019-12-9.json').skeleton_id)

uPN_left = 7865696
MBON_left = 16223537
threshold = 0.01
KC_oneside = 16630385

_, ds, ds_edges = adj_mat.downstream(uPN_left, threshold)
edges, skids = adj_mat.edge_threshold(ds_edges, threshold, 'downstream', include_nonpaired = KCs, left=left, right=right)

_, ds, ds_edges = adj_mat.downstream(KC_oneside, threshold)
edges, skids = adj_mat.edge_threshold(ds_edges, threshold, 'downstream', include_nonpaired = KCs, left=left, right=right)
