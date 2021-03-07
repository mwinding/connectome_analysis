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

adj_ad = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj_aa = pd.read_csv('VNC_interaction/data/brA1_axon-axon.csv', header = 0, index_col = 0)
adj_da = pd.read_csv('VNC_interaction/data/brA1_dendrite-axon.csv', header = 0, index_col = 0)
adj_dd = pd.read_csv('VNC_interaction/data/brA1_dendrite-dendrite.csv', header = 0, index_col = 0)

adj_type = 'all'
adj_all = adj_ad + adj_aa + adj_da + adj_dd
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

# remove A1 except for ascendings
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj_ad.index, A1_local)) 

# remove all local A1 skids from adjacency matrix
adj_all = adj_all.loc[pruned_index, pruned_index] 
adj_ad = adj_ad.loc[pruned_index, pruned_index] 
adj_aa = adj_aa.loc[pruned_index, pruned_index] 
adj_dd = adj_dd.loc[pruned_index, pruned_index] 
adj_da = adj_da.loc[pruned_index, pruned_index] 
adj_allaa = adj_allaa.loc[pruned_index, pruned_index] 
adj_ad_da = adj_ad_da.loc[pruned_index, pruned_index] 

# load inputs and pair data
inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs
pairs.drop(1121, inplace=True) # remove duplicate rightid

# %%
# load adj matrices

adj_all_mat = pm.Adjacency_matrix(adj_all.values, adj_all.index, pairs, inputs, 'all')
adj_ad_mat = pm.Adjacency_matrix(adj_ad.values, adj_ad.index, pairs, inputs, 'ad')
adj_aa_mat = pm.Adjacency_matrix(adj_aa.values, adj_aa.index, pairs, inputs, 'aa')
adj_dd_mat = pm.Adjacency_matrix(adj_dd.values, adj_dd.index, pairs, inputs, 'dd')
adj_da_mat = pm.Adjacency_matrix(adj_da.values, adj_da.index, pairs, inputs, 'da')
adj_allaa_mat = pm.Adjacency_matrix(adj_allaa.values, adj_allaa.index, pairs, inputs, 'all-aa')
adj_ad_da_mat = pm.Adjacency_matrix(adj_ad_da.values, adj_ad_da.index, pairs, inputs, 'ad_da')

# %%
# generate all paired and nonpaired edges from each matrix with threshold
adjs = [adj_all_mat, adj_ad_mat, adj_aa_mat, adj_dd_mat, adj_da_mat, adj_allaa_mat, adj_ad_da_mat]
adjs_names = ['all', 'ad', 'aa', 'dd', 'da', 'all-aa', 'ad_da']

threshold = 0.01
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

for i, adj_mat in enumerate(adjs):
    matrix_pairs = pm.Promat.extract_pairs_from_list(adj_mat.skids, pairs)
    matrix_nonpaired = list(np.intersect1d(matrix_pairs[2].nonpaired, left+right)) # ignore unipolar neurons
    all_sources = list(matrix_pairs[0].leftid) + matrix_nonpaired

    all_edges_combined = adj_mat.threshold_edge_list(all_sources, matrix_nonpaired, threshold, left, right) # currently generates edge list for all paired -> paired/nonpaired, nonpaired -> paired/nonpaired
    all_edges_combined.to_csv(f'network_analysis/csv/{adjs_names[i]}_all_paired_edges.csv')

# %%
