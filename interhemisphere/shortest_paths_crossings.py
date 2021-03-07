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

# %%
# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

# %%
# load into network x object
graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')

# %%
# calculate shortest paths

dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')

target_names = ['dVNC', 'dSEZ', 'RGN']
targets = [dVNC_pair_ids, dSEZ_pair_ids, RGN_pair_ids]

sensories_pair_ids = [pm.Promat.load_pairs_from_annotation(x, pairs, return_type='all_pair_ids') for x in pymaid.get_annotated('mw brain inputs').name]
#sensories_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'asc-proprio', 'asc-mechano', 'asc-classII-III', 'asc-noci']
sensories_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd']
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]

all_sensories = list(np.intersect1d(all_sensories, graph.G.nodes))
dVNC_pair_ids = list(np.intersect1d(dVNC_pair_ids, graph.G.nodes))

cutoff=10

shortest_paths = []
for i in range(len(all_sensories)):
    sens_shortest_paths = []
    for j in range(len(dVNC_pair_ids)):
        try:
            shortest_path = nx.shortest_path(graph.G, all_sensories[i], dVNC_pair_ids[j])
            sens_shortest_paths.append(shortest_path)
        except:
            print(f'probably no path exists from {all_sensories[i]}-{dVNC_pair_ids[j]}') 

    shortest_paths.append(sens_shortest_paths)

all_shortest_paths = [x for sublist in shortest_paths for x in sublist]

# %%
# calculate crossing per path

graph_crossings = pg.Prograph.crossing_counts(graph.G, all_shortest_paths)

control_hists = []
total_paths = len(graph_crossings)
binwidth = 1
x_range = list(range(0, 7))
data = graph_crossings
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
hist = np.histogram(data, bins=bins)
for hist_pair in zip(hist[0], hist[0]/total_paths, [x for x in range(len(hist[0]))], ['control']*len(hist[0]), [0]*len(hist[0])):
    control_hists.append(hist_pair)

control_hists = pd.DataFrame(control_hists, columns = ['count', 'fraction', 'bin', 'condition', 'repeat'])

# plot as raw path counts
fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=control_hists, x='bin', y='count', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_dVNC.pdf', bbox_inches='tight')

# plot as fraction total paths

fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=control_hists, x='bin', y='fraction', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_dVNC_fraction-total-paths.pdf', bbox_inches='tight')
# %%

shortest_paths = []
for i in range(len(all_sensories)):
    sens_shortest_paths = []
    for j in range(len(dSEZ_pair_ids)):
        try:
            shortest_path = nx.shortest_path(graph.G, all_sensories[i], dSEZ_pair_ids[j])
            sens_shortest_paths.append(shortest_path)
        except:
            print(f'probably no path exists from {all_sensories[i]}-{dSEZ_pair_ids[j]}') 

    shortest_paths.append(sens_shortest_paths)

all_shortest_paths = [x for sublist in shortest_paths for x in sublist]

# calculate crossing per path

graph_crossings = pg.Prograph.crossing_counts(graph.G, all_shortest_paths)

control_hists = []
total_paths = len(graph_crossings)
binwidth = 1
x_range = list(range(0, 7))
data = graph_crossings
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
hist = np.histogram(data, bins=bins)
for hist_pair in zip(hist[0], hist[0]/total_paths, [x for x in range(len(hist[0]))], ['control']*len(hist[0]), [0]*len(hist[0])):
    control_hists.append(hist_pair)

control_hists = pd.DataFrame(control_hists, columns = ['count', 'fraction', 'bin', 'condition', 'repeat'])

# plot as raw path counts
fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=control_hists, x='bin', y='count', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_dSEZ.pdf', bbox_inches='tight')

# plot as fraction total paths

fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=control_hists, x='bin', y='fraction', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_dSEZ_fraction-total-paths.pdf', bbox_inches='tight')
# %%
#

shortest_paths = []
for i in range(len(all_sensories)):
    sens_shortest_paths = []
    for j in range(len(RGN_pair_ids)):
        try:
            shortest_path = nx.shortest_path(graph.G, all_sensories[i], RGN_pair_ids[j])
            sens_shortest_paths.append(shortest_path)
        except:
            print(f'probably no path exists from {all_sensories[i]}-{RGN_pair_ids[j]}') 

    shortest_paths.append(sens_shortest_paths)

all_shortest_paths = [x for sublist in shortest_paths for x in sublist]

# calculate crossing per path

graph_crossings = pg.Prograph.crossing_counts(graph.G, all_shortest_paths)

control_hists = []
total_paths = len(graph_crossings)
binwidth = 1
x_range = list(range(0, 7))
data = graph_crossings
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
hist = np.histogram(data, bins=bins)
for hist_pair in zip(hist[0], hist[0]/total_paths, [x for x in range(len(hist[0]))], ['control']*len(hist[0]), [0]*len(hist[0])):
    control_hists.append(hist_pair)

control_hists = pd.DataFrame(control_hists, columns = ['count', 'fraction', 'bin', 'condition', 'repeat'])

# plot as raw path counts
fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=control_hists, x='bin', y='count', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_RGN.pdf', bbox_inches='tight')

# plot as fraction total paths

fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=control_hists, x='bin', y='fraction', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_RGN_fraction-total-paths.pdf', bbox_inches='tight')
# %%
