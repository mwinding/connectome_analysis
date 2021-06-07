#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

from pymaid_creds import url, name, password, token
import pymaid

import numpy as np
import pandas as pd

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import networkx as nx

rm = pymaid.CatmaidInstance(url, token, name, password)
# %%
brain = pymaid.get_skids_by_annotation('mw brain neurons')
pairs = pm.Promat.get_pairs()

# import graphs of axon-dendrite split data; generated by Ben Pedigo's scripts
G = nx.readwrite.graphml.read_graphml('data/graphs/G.graphml', node_type=int)
Gad = nx.readwrite.graphml.read_graphml('data/graphs/Gad.graphml', node_type=int)
Gaa = nx.readwrite.graphml.read_graphml('data/graphs/Gaa.graphml', node_type=int)
Gdd = nx.readwrite.graphml.read_graphml('data/graphs/Gdd.graphml', node_type=int)
Gda = nx.readwrite.graphml.read_graphml('data/graphs/Gda.graphml', node_type=int)

# generate adjacency matrices
adj_all = pd.DataFrame(nx.adjacency_matrix(G=G, weight = 'weight').todense(), columns = G.nodes, index = G.nodes)
adj_ad = pd.DataFrame(nx.adjacency_matrix(G=Gad, weight = 'weight').todense(), columns = Gad.nodes, index = Gad.nodes)
adj_aa = pd.DataFrame(nx.adjacency_matrix(G=Gaa, weight = 'weight').todense(), columns = Gaa.nodes, index = Gaa.nodes)
adj_dd = pd.DataFrame(nx.adjacency_matrix(G=Gdd, weight = 'weight').todense(), columns = Gdd.nodes, index = Gdd.nodes)
adj_da = pd.DataFrame(nx.adjacency_matrix(G=Gda, weight = 'weight').todense(), columns = Gda.nodes, index = Gda.nodes)

# add back in skids with no edges (with rows/columns of 0); simplifies some later analysis
def refill_adjs(adj, adj_all):
    skids_diff = np.setdiff1d(adj_all.index, adj.index)
    adj = adj.append(pd.DataFrame([[0]*adj.shape[1]]*len(skids_diff), index = skids_diff, columns = adj.columns)) # add in rows with 0s
    for skid in skids_diff:
        adj[skid]=[0]*len(adj.index)
    return(adj)

adj_ad = refill_adjs(adj_ad, adj_all)
adj_aa = refill_adjs(adj_aa, adj_all)
adj_dd = refill_adjs(adj_dd, adj_all)
adj_da = refill_adjs(adj_da, adj_all)

# export adjacency matrices
adj_all.to_csv('data/adj/all-neurons_all-all.csv')
adj_ad.to_csv('data/adj/all-neurons_ad.csv')
adj_aa.to_csv('data/adj/all-neurons_aa.csv')
adj_dd.to_csv('data/adj/all-neurons_dd.csv')
adj_da.to_csv('data/adj/all-neurons_da.csv')

# import input data and export as simplified csv
meta_data = pd.read_csv('data/graphs/meta_data.csv', index_col = 0)
inputs = meta_data.loc[:, ['axon_input', 'dendrite_input']]
outputs = meta_data.loc[:, ['axon_output', 'dendrite_output']]

# exporting input data
inputs.to_csv('data/graphs/inputs.csv')
outputs.to_csv('data/graphs/outputs.csv')

# making some custom adjacencies without certain edge types
adj_allaa = adj_ad + adj_da + adj_dd
adj_ad_da = adj_ad + adj_da

#convert column names to int for easier indexing
adj_all.columns = adj_all.columns.astype(int)
adj_ad.columns = adj_ad.columns.astype(int)
adj_aa.columns = adj_aa.columns.astype(int)
adj_da.columns = adj_da.columns.astype(int)
adj_dd.columns = adj_dd.columns.astype(int)
adj_allaa.columns = adj_allaa.columns.astype(int)
adj_ad_da.columns = adj_ad_da.columns.astype(int)

# %%
# prune out A1 neurons from adjacency matrices (optional)

# remove A1 except for ascendings, also paritally differentiated neurons
A1_ascending = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain ascendings').name))
A1_ascending = [x for sublist in A1_ascending for x in sublist]
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
prune_all = A1_local + pymaid.get_skids_by_annotation('mw partially differentiated') #+ pymaid.get_skids_by_annotation('mw brain accessory neurons')
pruned_index = [list(np.setdiff1d(x.index, A1_local))  for x in [adj_all, adj_ad, adj_aa, adj_dd, adj_da, adj_allaa, adj_ad_da]]

# remove all local A1 skids from adjacency matrix
adj_all = adj_all.loc[pruned_index[0], pruned_index[0]] 
adj_ad = adj_ad.loc[pruned_index[1], pruned_index[1]] 
adj_aa = adj_aa.loc[pruned_index[2], pruned_index[2]] 
adj_dd = adj_dd.loc[pruned_index[3], pruned_index[3]] 
adj_da = adj_da.loc[pruned_index[4], pruned_index[4]] 
adj_allaa = adj_allaa.loc[pruned_index[5], pruned_index[5]] 
adj_ad_da = adj_ad_da.loc[pruned_index[6], pruned_index[6]] 

# %%
# load adj matrices

adj_all_mat = pm.Adjacency_matrix(adj_all, inputs, 'summed')
adj_ad_mat = pm.Adjacency_matrix(adj_ad, inputs, 'ad')
adj_aa_mat = pm.Adjacency_matrix(adj_aa, inputs, 'aa')
adj_dd_mat = pm.Adjacency_matrix(adj_dd, inputs, 'dd')
adj_da_mat = pm.Adjacency_matrix(adj_da, inputs, 'da')
adj_allaa_mat = pm.Adjacency_matrix(adj_allaa, inputs, 'all-aa')
adj_ad_da_mat = pm.Adjacency_matrix(adj_ad_da, inputs, 'ad_da')

# %%
# generate all paired and nonpaired edges from each matrix with threshold
# export as paired edges between paired neurons (collapse left/right hemispheres, except for nonpaired neurons)
# export as normal edge list, but with pair-wise threshold

adjs = [adj_all_mat, adj_ad_mat, adj_aa_mat, adj_dd_mat, adj_da_mat, adj_allaa_mat, adj_ad_da_mat]
adjs_names = ['summed', 'ad', 'aa', 'dd', 'da', 'all-aa', 'ad_da']

threshold = 0.01
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

for i, adj_mat in enumerate(adjs):
    matrix_pairs = pm.Promat.extract_pairs_from_list(adj_mat.skids)
    matrix_nonpaired = list(np.intersect1d(matrix_pairs[2].nonpaired, left+right)) # ignore unipolar neurons, not in set of brain neurons
    all_sources = list(matrix_pairs[0].leftid) + matrix_nonpaired

    all_edges_combined = adj_mat.threshold_edge_list(all_sources, matrix_nonpaired, threshold, left, right) # currently generates edge list for all paired -> paired/nonpaired, nonpaired -> paired/nonpaired
    all_edges_combined.to_csv(f'data/edges_threshold/{adjs_names[i]}_all-paired-edges.csv')
    all_edges_split = adj_mat.split_paired_edges(all_edges_combined, left, right)
    all_edges_split.to_csv(f'data/edges_threshold/pairwise-threshold_{adjs_names[i]}_all-edges.csv')

# %%
# load data for proofreading purposes

paired_edge_lists = [pd.read_csv(f'data/edges_threshold/{name}_all-paired-edges.csv', index_col=0) for name in adjs_names]
edge_lists = [pd.read_csv(f'data/edges_threshold/pairwise-threshold_{name}_all-edges.csv', index_col=0) for name in adjs_names]
# %%
