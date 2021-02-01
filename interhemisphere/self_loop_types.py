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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, name, password, token)

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

ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)[0]
bi_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]
contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)[0]

adj_mat = pm.Adjacency_matrix(adj.values, adj.index, pairs, inputs, 'axo-dendritic')
# %%
# generate paths for whole brain
from tqdm import tqdm
from joblib import Parallel, delayed

def hop_edges(pair_id, threshold, adj_mat, edges_only=False):

    _, ds, ds_edges = adj_mat.downstream(pair_id, threshold)
    ds_edges, _ = adj_mat.edge_threshold(ds_edges, threshold, 'downstream')
    overthres_ds_edges = ds_edges[ds_edges.overthres==True]
    overthres_ds_edges.reset_index(inplace=True)
    overthres_ds_edges.drop(labels=['index', 'overthres'], axis=1, inplace=True)

    if(edges_only==False):
        return(overthres_ds_edges, np.unique(overthres_ds_edges.downstream_pair_id))
    if(edges_only):
        return(overthres_ds_edges)

br = pymaid.get_skids_by_annotation('mw brain neurons')
br_pairs = pm.Promat.extract_pairs_from_list(br, pairs)[0]
matrix_pairs = pm.Promat.extract_pairs_from_list(adj_mat.skids, pairs)[0]

threshold = 0.01
all_paths = Parallel(n_jobs=-1)(delayed(hop_edges)(pair, threshold, adj_mat, edges_only=True) for pair in tqdm(matrix_pairs.leftid))
all_paths_combined = [x for x in all_paths if type(x)==pd.DataFrame]
all_paths_combined = pd.concat(all_paths_combined, axis=0)
all_paths_combined.reset_index(inplace=True, drop=True)

all_paths_combined.to_csv('interhemisphere/csvs/all_paired_edges.csv')

# %%
# load into network x object

import networkx as nx 

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dVNC_pairs = pm.Promat.extract_pairs_from_list(dVNC, pairs)[0]

dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dSEZ_pairs = pm.Promat.extract_pairs_from_list(dSEZ, pairs)[0]
# directed graph
G = nx.DiGraph()

# build the graph
for i in range(len(all_paths_combined)):
    G.add_edge(all_paths_combined.iloc[i].upstream_pair_id, all_paths_combined.iloc[i].downstream_pair_id, 
                weight = np.mean([all_paths_combined.iloc[i].left, all_paths_combined.iloc[i].right]), 
                edge_type = all_paths_combined.iloc[i].type)

#G.add_edges_from(all_paths_combined.loc[:, ['upstream_pair_id', 'downstream_pair_id']].values)

def path_edge_attributes(path, attribute_name, include_skids=True):
    if(include_skids):
        return [(u,v,G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])]
    if(include_skids==False):
        return np.array([(G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])])

paths = nx.all_simple_paths(G, 3410499, dVNC_pairs.leftid.values, cutoff=6) # ORN-42b to dVNCs
crossing_count = (sum(path_edge_attributes(path, 'edge_type', False)=='contralateral') for path in paths)
sns.violinplot(list(crossing_count), orient='v')

paths = nx.all_simple_paths(G, 3410499, dSEZ_pairs.leftid.values, cutoff=6) # ORN-42b to dSEZs
crossing_count = (sum(path_edge_attributes(path, 'edge_type', False)=='contralateral') for path in paths)
sns.violinplot(list(crossing_count), orient='v')

# %%
# self-loop paths



# %%
#
'''
def multi_hop_edges(pair_id, threshold, adj_mat):
    
    edges_list = []
    edges_1, ds_pairs_1 = hop_edges(pair_id, threshold, adj_mat)
    
    for ds_pair_1 in tqdm(ds_pairs_1):
        edges_2, ds_pairs_2 = hop_edges(ds_pair_1, threshold, adj_mat)
        for ds_pair_2 in tqdm(ds_pairs_2):
            edges_3, ds_pairs_3 = hop_edges(ds_pair_2, threshold, adj_mat)
            for ds_pair_3 in ds_pairs_3:
                edges_4, ds_pairs_4 = hop_edges(ds_pair_3, threshold, adj_mat)
                for ds_pair_4 in ds_pairs_4:
                    edges_5, ds_pairs_5 = hop_edges(ds_pair_4, threshold, adj_mat)
                    for ds_pair_5 in ds_pairs_5:
                        edges_6, _ = hop_edges(ds_pair_5, threshold, adj_mat)
                        path = pd.concat([edges_1[edges_1.downstream_pair_id==ds_pair_1], 
                                            edges_2[edges_2.downstream_pair_id==ds_pair_2], 
                                            edges_3[edges_3.downstream_pair_id==ds_pair_3],
                                            edges_4[edges_4.downstream_pair_id==ds_pair_4], 
                                            edges_5[edges_5.downstream_pair_id==ds_pair_5]], 
                                            axis=0)
                        #path['path'] = f'path-{pair_id}_{i}_{j}'
                        edges_list.append(path)
    
    return(edges_list)
                    
test = multi_hop_edges(CN33[0], 0.05, adj_mat)
test_combined = [x for x in test if type(x)==pd.DataFrame]
test_combined = pd.concat(test_combined, axis=0)
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


# %%
