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

#adj_type = 'all'
#adj = adj_ad + adj_aa + adj_da + adj_dd

#adj_type = 'all-aa'
#adj = adj_ad + adj_da + adj_dd

#adj_type = 'ad'
#adj = adj_ad

#adj_type = 'aa'
#adj = adj_aa

#adj_type = 'dd'
#adj = adj_dd

#adj_type = 'da'
#adj = adj_da

adj_type = 'ad_da'
adj = adj_ad + adj_da

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
graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='undirected')

# %%
# generate rich-club coefficients for the thresholded graph and 100 graphs with 100*M double-edge swaps, where M=number of edges
from joblib import Parallel, delayed

rc = nx.rich_club_coefficient(graph.G, normalized=False, Q=100, seed=1)
rc = [x[1] for x in rc.items()]

def random_rc(graph, seed):
    Q = 100
    R = graph.copy()
    E = R.number_of_edges()
    nx.double_edge_swap(R,Q*E,max_tries=Q*E*10, seed=seed)
    R_rc = nx.rich_club_coefficient(R, normalized=False, Q=Q, seed=seed)
    return(R_rc)

# generating randomized graphs and calculating rich-club coefficient
n_init=8
R_rc_list = Parallel(n_jobs=-1)(delayed(random_rc)(graph.G, seed=i) for i in tqdm(range(0, n_init)))
R_rc_list_df = pd.DataFrame(R_rc_list)

# plot normalized rich-club plot
fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(x=[x for x in range(0, len(rc))], y=np.array(rc)/R_rc_list_df.mean(axis=0), ax=ax, linewidth=0.5)
ax.set(xlim=(0,80))
plt.savefig(f'network_analysis/plots/{adj_type}_rich-club-normalized_pair-threshold-graph_{n_init}-repeats.pdf', format='pdf', bbox_inches='tight')

# plot random with error bars vs observed
data = []
for i in range(len(R_rc_list_df.index)):
    for j in range(len(R_rc_list_df.columns)):
        data.append([R_rc_list_df.iloc[i, j], j, 'random'])

data = pd.DataFrame(data, columns = ['rich_club_coeff', 'degree', 'type'])
rc_data = [[x, i, 'observed'] for i, x in enumerate(rc)]

data = pd.concat([data, pd.DataFrame(rc_data, columns = ['rich_club_coeff', 'degree', 'type'])], axis=0)

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(data=data, x='degree', y='rich_club_coeff', hue='type', ci='sd', ax=ax, linewidth=0.5)
ax.set(xlim=(0,80))
ax.set(ylim=(0, .8))
plt.savefig(f'network_analysis/plots/{adj_type}_rich-club_pair-threshold-graph_{n_init}-repeats.pdf', format='pdf', bbox_inches='tight')

# %%
# generate rich-club coefficients for the raw graph and 100 graphs with 100*M double-edge swaps, where M=number of edges

all_edges = adj_mat.edge_list(exclude_loops=True)
all_edges_df = pd.DataFrame(all_edges, columns = ['upstream_pair_id', 'downstream_pair_id'])
graph_complete = pg.Analyze_Nx_G(all_edges_df, graph_type='undirected')

rc_complete = nx.rich_club_coefficient(graph_complete.G, normalized=False, Q=100, seed=1)
rc_complete = [x[1] for x in rc_complete.items()]

# generating randomized graphs
n_init = 8
R_rc_com_list = Parallel(n_jobs=-1)(delayed(random_rc)(graph_complete.G, seed=i) for i in tqdm(range(0, n_init)))
R_rc_com_list_df = pd.DataFrame(R_rc_com_list)

# plot normalized rich-club plot
fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(x=[x for x in range(0, len(rc_complete))], y=np.array(rc_complete)/R_rc_com_list_df.mean(axis=0), ax=ax, linewidth=0.5)
plt.savefig(f'network_analysis/plots/{adj_type}_raw_rich-club-normalized_pair-threshold-graph_{n_init}-repeats.pdf', format='pdf', bbox_inches='tight')

# plot random with error bars vs observed
data_com = []
for i in range(len(R_rc_com_list_df.index)):
    for j in range(len(R_rc_com_list_df.columns)):
        data_com.append([R_rc_com_list_df.iloc[i, j], j, 'random'])

data_com = pd.DataFrame(data_com, columns = ['rich_club_coeff', 'degree', 'type'])
rc_com_data = [[x, i, 'observed'] for i, x in enumerate(rc_complete)]

data_com = pd.concat([data_com, pd.DataFrame(rc_com_data, columns = ['rich_club_coeff', 'degree', 'type'])], axis=0)

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(data=data_com, x='degree', y='rich_club_coeff', hue='type', ci='sd', linewidth=0.5)
ax.set(ylim=(0, .8))
plt.savefig(f'network_analysis/plots/{adj_type}_raw_rich-club_pair-threshold-graph_{n_init}-repeats.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(data=data_com, x='degree', y='rich_club_coeff', hue='type', ci='sd', linewidth=0.25)
ax.set(ylim=(0, .8))
plt.savefig(f'network_analysis/plots/{adj_type}_raw_rich-club_pair-threshold-graph_{n_init}-repeats_thin_line.pdf', format='pdf', bbox_inches='tight')

# %%
