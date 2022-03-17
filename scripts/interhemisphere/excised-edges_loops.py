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
import gzip
import csv

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx 

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

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

ipsi_pair_ids = pm.Promat.load_pairs_from_annotation('mw ipsilateral axon', pairs, return_type='all_pair_ids')
bilateral_pair_ids = pm.Promat.load_pairs_from_annotation('mw bilateral axon', pairs, return_type='all_pair_ids')
contra_pair_ids = pm.Promat.load_pairs_from_annotation('mw contralateral axon', pairs, return_type='all_pair_ids')

dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')

sensories_pair_ids = [pm.Promat.load_pairs_from_annotation(x, pairs, return_type='all_pair_ids') for x in pymaid.get_annotated('mw brain inputs').name]
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]

# %%
# EXPERIMENT 1: removing edges from contralateral and bilateral neurons -> effect on self loops?
# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

# iterations for random edge removal as control
n_init = 8

# generate wildtype graph
wildtype = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')

# excise edges and generate graphs
e_contra_contra, e_contra_contra_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(contra_pair_ids, all_sensories), 'contralateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
e_bi_contra, e_bi_contra_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(bilateral_pair_ids, all_sensories), 'contralateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
e_bi_ipsi, e_bi_ipsi_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(bilateral_pair_ids, all_sensories), 'ipsilateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
e_all_contra, e_all_contra_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(bilateral_pair_ids + contra_pair_ids, all_sensories), 'contralateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))

# %%
# generate paths and identify loops
cutoff=3

experimental = [wildtype, e_contra_contra, e_bi_contra, e_bi_ipsi, e_all_contra]
experimental_graphs_loops = Parallel(n_jobs=-1)(delayed(pg.Prograph.self_loop_experiment)(experimental[i], ipsi_pair_ids, bilateral_pair_ids, contra_pair_ids, all_sensories, cutoff) for i in tqdm(range(len(experimental))))

e_contra_contra_control_loops = Parallel(n_jobs=-1)(delayed(pg.Prograph.self_loop_experiment)(e_contra_contra_control[i], ipsi_pair_ids, bilateral_pair_ids, contra_pair_ids, all_sensories, cutoff) for i in tqdm(range(len(e_contra_contra_control))))
e_bi_contra_control_loops = Parallel(n_jobs=-1)(delayed(pg.Prograph.self_loop_experiment)(e_bi_contra_control[i], ipsi_pair_ids, bilateral_pair_ids, contra_pair_ids, all_sensories, cutoff) for i in tqdm(range(len(e_bi_contra_control))))
e_bi_ipsi_control_loops = Parallel(n_jobs=-1)(delayed(pg.Prograph.self_loop_experiment)(e_bi_ipsi_control[i], ipsi_pair_ids, bilateral_pair_ids, contra_pair_ids, all_sensories, cutoff) for i in tqdm(range(len(e_bi_ipsi_control))))
e_all_contra_control_loops = Parallel(n_jobs=-1)(delayed(pg.Prograph.self_loop_experiment)(e_all_contra_control[i], ipsi_pair_ids, bilateral_pair_ids, contra_pair_ids, all_sensories, cutoff) for i in tqdm(range(len(e_all_contra_control))))

# %%
# plot data

def reorganize_data(loops_list, condition, append_here=None):
    ipsi = [x[0] for x in loops_list]
    bilateral = [x[1] for x in loops_list]
    contra = [x[2] for x in loops_list]

    if(append_here is None):
        ipsi_loops = split_self_pair_loops(ipsi, condition, '1_ipsi', append_here)
        bilateral_loops = split_self_pair_loops(bilateral, condition, '2_bilateral', append_here)
        contra_loops = split_self_pair_loops(contra, condition, '3_contra', append_here)

    if(append_here is not None):
        ipsi_loops = split_self_pair_loops(ipsi, condition, '1_ipsi', append_here[0])
        bilateral_loops = split_self_pair_loops(bilateral, condition, '2_bilateral', append_here[1])
        contra_loops = split_self_pair_loops(contra, condition, '3_contra', append_here[2])

    return(ipsi_loops, bilateral_loops, contra_loops)

def split_self_pair_loops(loops, condition, celltype, append_here=None):
    contains_loop = [[1-x.iloc[0], condition, celltype, 'any', -1] for x in loops]

    contains_loop_1pair = [[x.loc[(1, 'pair')], condition, celltype, 'pair_loop', 1] for x in loops]
    contains_loop_2pair = [[x.loc[(2, 'pair')], condition, celltype, 'pair_loop', 2] for x in loops]
    contains_loop_3pair = [[x.loc[(3, 'pair')], condition, celltype, 'pair_loop', 3] for x in loops]

    contains_loop_1self = [[x.loc[(1, 'self')], condition, celltype, 'self_loop', 1] for x in loops]
    contains_loop_2self = [[x.loc[(2, 'self')], condition, celltype, 'self_loop', 2] for x in loops]
    contains_loop_3self = [[x.loc[(3, 'self')], condition, celltype, 'self_loop', 3] for x in loops]

    if(append_here is None):
        data = pd.DataFrame(np.concatenate((contains_loop, contains_loop_1pair, contains_loop_2pair, contains_loop_3pair, contains_loop_1self, contains_loop_2self, contains_loop_3self)), columns = ['fraction_loops', 'condition', 'cell_type', 'loop_type', 'path_length'])
    if(append_here is not None):
        data = pd.DataFrame(np.concatenate((contains_loop, contains_loop_1pair, contains_loop_2pair, contains_loop_3pair, contains_loop_1self, contains_loop_2self, contains_loop_3self)), columns = ['fraction_loops', 'condition', 'cell_type', 'loop_type', 'path_length'])
        data = pd.concat([append_here, data], axis=0)
    return(data)

data = reorganize_data(e_contra_contra_control_loops, 'Cc-control')
data = reorganize_data(e_bi_contra_control_loops, 'Bc-control', data)
data = reorganize_data(e_bi_ipsi_control_loops, 'Bc-control', data)
data = reorganize_data(e_all_contra_control_loops, 'Bc-control', data)
data = reorganize_data([experimental_graphs_loops[0]], 'Wildtype', data)
data = reorganize_data([experimental_graphs_loops[1]], 'Contra-contra', data)
data = reorganize_data([experimental_graphs_loops[2]], 'Bilateral-contra', data)
data = reorganize_data([experimental_graphs_loops[3]], 'Bilateral-ipsi', data)
data = reorganize_data([experimental_graphs_loops[4]], 'All-contra', data)

data = pd.concat(data, axis=0)
data.fraction_loops = [float(x) for x in data.fraction_loops]

conditions = ['Wildtype', 'Contra-contra', 'Bilateral-contra', 'Bilateral-ipsi', 'All-contra']
fig, axs = plt.subplots(len(conditions),2, figsize=(2,0.75*len(conditions)), sharey=True, sharex=True)
fig.tight_layout(pad=0.5)
for i, condition in enumerate(conditions):
    ax = axs[i, 0]
    plot_data = data[(data.loop_type == 'self_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    sns.heatmap(plot_data, cmap='Blues', annot=True, fmt='.2f' ,ax=ax, cbar=False, vmax=.35)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

    ax = axs[i, 1]
    plot_data = data[(data.loop_type == 'pair_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    sns.heatmap(plot_data, cmap='Purples', annot=True, fmt='.2f', ax=ax, cbar=False, vmax=.35)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

plt.savefig('interhemisphere/plots/loops/compare-loops_excised-edges.pdf', format='pdf', bbox_inches='tight')


self_wt = data[(data.loop_type == 'self_loop') & (data.condition=='Wildtype')].pivot(index='cell_type', columns='path_length', values='fraction_loops')
pair_wt = data[(data.loop_type == 'pair_loop') & (data.condition=='Wildtype')].pivot(index='cell_type', columns='path_length', values='fraction_loops')

conditions = ['Wildtype', 'Contra-contra', 'Bilateral-contra', 'Bilateral-ipsi', 'All-contra']
fig, axs = plt.subplots(len(conditions),2, figsize=(2,0.75*len(conditions)), sharey=True, sharex=True)
fig.tight_layout(pad=0.5)
for i, condition in enumerate(conditions):
    ax = axs[i, 0]
    plot_data = data[(data.loop_type == 'self_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    if(i==0):
        sns.heatmap(plot_data, cmap='Blues', annot=True, fmt='.2f' ,ax=ax, cbar=False, vmax=.35)
    if(i>0):
        sns.heatmap(((plot_data-self_wt)/self_wt).fillna(0), annot=True, fmt='.2f' ,ax=ax, cmap=sns.cm.rocket_r, cbar=False, vmin=-1, vmax=0)

    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

    ax = axs[i, 1]
    plot_data = data[(data.loop_type == 'pair_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    if(i==0):
        sns.heatmap(plot_data, cmap='Purples', annot=True, fmt='.2f', ax=ax, cbar=False, vmax=.35)
    if(i>0):
        sns.heatmap(((plot_data-pair_wt)/pair_wt).fillna(0), annot=True, fmt='.2f', ax=ax, cmap=sns.cm.rocket_r, cbar=False, vmin=-1, vmax=0)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

plt.savefig('interhemisphere/plots/loops/compare-loops_excised-edges_normalized.pdf', format='pdf', bbox_inches='tight')

# %%
# simple plot for main figure

conditions = ['Wildtype', 'All-contra']
fig, axs = plt.subplots(len(conditions),2, figsize=(1.5,0.5*len(conditions)), sharey=True, sharex=True)
fig.tight_layout(pad=0.05)
for i, condition in enumerate(conditions):
    ax = axs[i, 0]
    plot_data = data[(data.loop_type == 'self_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    sns.heatmap(plot_data, cmap='Blues', annot=True, fmt='.2f' ,ax=ax, cbar=False, vmax=.25)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

    ax = axs[i, 1]
    plot_data = data[(data.loop_type == 'pair_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    sns.heatmap(plot_data, cmap='Purples', annot=True, fmt='.2f' ,ax=ax, cbar=False, vmax=.25)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

plt.savefig('interhemisphere/plots/loops/simple_compare-loops_excised-edges.pdf', format='pdf', bbox_inches='tight')

self_wt = data[(data.loop_type == 'self_loop') & (data.condition=='Wildtype')].pivot(index='cell_type', columns='path_length', values='fraction_loops')
pair_wt = data[(data.loop_type == 'pair_loop') & (data.condition=='Wildtype')].pivot(index='cell_type', columns='path_length', values='fraction_loops')
conditions = ['Wildtype', 'All-contra']
fig, axs = plt.subplots(len(conditions),2, figsize=(1.5,0.5*len(conditions)), sharey=True, sharex=True)
fig.tight_layout(pad=0.05)
for i, condition in enumerate(conditions):
    ax = axs[i, 0]
    plot_data = data[(data.loop_type == 'self_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    if(i==0):
        sns.heatmap(plot_data, cmap='Blues', annot=True, fmt='.3f', ax=ax, cbar=False, vmax=.35)
    if(i>0):
        sns.heatmap(((plot_data-self_wt)/self_wt).fillna(0), annot=True, fmt='.2f', ax=ax, cmap=sns.cm.rocket_r, cbar=False, vmin=-1, vmax=0)    
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

    ax = axs[i, 1]
    plot_data = data[(data.loop_type == 'pair_loop') & (data.condition==condition)].pivot(index='cell_type', columns='path_length', values='fraction_loops')
    if(i==0):
        sns.heatmap(plot_data, cmap='Purples', annot=True, fmt='.3f', ax=ax, cbar=False, vmax=.35)
    if(i>0):
        sns.heatmap(((plot_data-pair_wt)/pair_wt).fillna(0), annot=True, fmt='.2f', ax=ax, cmap=sns.cm.rocket_r, cbar=False, vmin=-1, vmax=0)    
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')    
    ax.set_xlabel('')

plt.savefig('interhemisphere/plots/loops/simple_compare-loops_excised-edges_normalized-change.pdf', format='pdf', bbox_inches='tight')

