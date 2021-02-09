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

rm = pymaid.CatmaidInstance(url, name, password, token)

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
# takes ~20s

n_init = 40

graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')
#shuffled_graphs = graph.generate_shuffled_graphs(n_init, graph_type='directed')
shuffled_graphs = Parallel(n_jobs=-1)(delayed(nx.readwrite.graphml.read_graphml)(f'interhemisphere/csv/shuffled_graphs/iteration-{i}.graphml', node_type=int, edge_key_type=str) for i in tqdm(range(n_init)))
shuffled_graphs = [pg.Analyze_Nx_G(edges=x.edges, graph=x) for x in shuffled_graphs]

# %%
# sensories to outputs paths and interhemisphere crossings
# generate the data and save it with pickle because it takes so long to generate
# total: ~40 min
dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')

target_names = ['dVNC', 'dSEZ', 'RGN']
targets = [dVNC_pair_ids, dSEZ_pair_ids, RGN_pair_ids]

sensories_pair_ids = [pm.Promat.load_pairs_from_annotation(x, pairs, return_type='all_pair_ids') for x in pymaid.get_annotated('mw brain inputs and ascending').name]
sensories_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'asc-proprio', 'asc-mechano', 'asc-classII-III', 'asc-noci']
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]

save_path = 'interhemisphere/plots/interhemisphere_crossings'
cutoff=5

# generate and save paths
'''
# generate and save paths for control
save_paths = [f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}',
                f'interhemisphere/csv/paths/all_paths_sens-to-dSEZ_cutoff{cutoff}',
                f'interhemisphere/csv/paths/all_paths_sens-to-RGN_cutoff{cutoff}']
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(graph.G, all_sensories, targets[i], cutoff=cutoff, save_path=save_paths[i]) for i in tqdm(range(len(save_paths))))

# generate and save paths for shuffled
save_path = f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}_shuffled-graph'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(shuffled_graphs[i].G, all_sensories, targets[0], cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
'''

# load graph paths and calculate interhemisphere crossings
graph_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}.csv.gz')
graph_crossings = pg.Prograph.crossing_counts(graph.G, graph_paths)

shuffled_graph_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}_shuffled-graph{i}.csv.gz') for i in tqdm(range(n_init)))
shuffled_graph_crossings = Parallel(n_jobs=-1)(delayed(pg.Prograph.crossing_counts)(shuffled_graphs[i].G, shuffled_graph_paths[i]) for i in tqdm(range(n_init)))

# save interhemisphere crossings
save_path = f'interhemisphere/csv/interhemisphere_crossings/all_paths_sens-to-dVNC_cutoff{cutoff}_control'
with gzip.open(save_path + '.csv.gz', 'wt') as f:
    writer = csv.writer(f)
    writer.writerow(graph_crossings)

# not working
'''
for i, shuffled_graph_crossing in enumerate(shuffled_graph_crossings):
    save_path = f'interhemisphere/csv/interhemisphere_crossings/all_paths_sens-to-dVNC_cutoff{cutoff}_shuffled-graphs{i}'
    with gzip.open(save_path + '.csv.gz', 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(shuffled_graph_crossing)
'''
# %%
# plot data

# open interhemisphere crossings if needed

shuffled_hists = []
for i, shuffled in enumerate(shuffled_graph_crossings):
    total_paths = len(shuffled)
    binwidth = 1
    x_range = list(range(0, 7))
    data = shuffled
    bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
    hist = np.histogram(data, bins=bins)
    for hist_pair in zip(hist[0], hist[0]/total_paths, [x for x in range(len(hist[0]))], ['shuffled']*len(hist[0]), [i]*len(hist[0])):
        shuffled_hists.append(hist_pair)

shuffled_hists = pd.DataFrame(shuffled_hists, columns = ['count', 'fraction', 'bin', 'condition', 'repeat'])

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

hists = pd.concat([shuffled_hists, control_hists], axis=0)
hists.to_csv(f'interhemisphere/csv/interhemisphere_crossings/df_all_paths_sens-to-dVNC_cutoff{cutoff}_shuffled-vs-control-graphs.csv')

# plot as raw path counts
fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=hists, x='bin', y='count', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/sens_to_dVNC_controls-vs-shuffled.pdf', bbox_inches='tight')

# plot as fraction total paths

fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=hists, x='bin', y='fraction', hue='condition', ax=ax)
plt.savefig('interhemisphere/plots/interhemisphere_crossings/sens_to_dVNC_controls-vs-shuffled_fraction-total-paths.pdf', bbox_inches='tight')

# %%
# load paths if needed
graph_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}.csv.gz')
shuffled_graph_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}_shuffled-graph{i}.csv.gz') for i in tqdm(range(n_init)))

graph_total_paths = len(graph_paths)
shuffled_graphs_total_paths = [len(x) for x in shuffled_graph_paths]

# %%
# number of each path length
control_path_lengths = []
total_paths = len(graph_paths)
data = [len(x) for x in graph_paths]
binwidth = 1
x_range = list(range(0, 7))
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
hist = np.histogram(data, bins=bins)
for hist_pair in zip(hist[0], hist[0]/total_paths, [x for x in range(len(hist[0]))], ['control']*len(hist[0]), [0]*len(hist[0])):
    control_path_lengths.append(hist_pair)

control_path_lengths = pd.DataFrame(control_path_lengths, columns = ['count', 'fraction', 'bin', 'condition', 'repeat'])

shuffled_hists = []
binwidth = 1
x_range = list(range(0, 7))
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
for i, shuffled in enumerate(shuffled_graph_paths):
    total_paths = len(shuffled)
    data = [len(x) for x in shuffled]
    hist = np.histogram(data, bins=bins)
    for hist_pair in zip(hist[0], hist[0]/total_paths, [x for x in range(len(hist[0]))], ['shuffled']*len(hist[0]), i*len(hist[0])):
        shuffled_hists.append(hist_pair)

shuffled_path_lengths = pd.DataFrame(shuffled_hists, columns = ['count', 'fraction', 'bin', 'condition', 'repeat'])

path_lengths_path = pd.concat([shuffled_path_lengths, control_path_lengths], axis=0)

# total number of paths
total_paths = control_path_lengths.groupby(['condition', 'repeat']).sum()