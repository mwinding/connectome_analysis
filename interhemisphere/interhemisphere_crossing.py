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
ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)[0]
bi_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]
contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)[0]

# %%
# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

# %%
# load into network x object

n_init = 100

graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')
#shuffled_graphs = graph.generate_shuffled_graphs(n_init, graph_type='directed')
shuffled_graphs = Parallel(n_jobs=-1)(delayed(nx.readwrite.graphml.read_graphml)(f'interhemisphere/csv/shuffled_graphs/iteration-{i}.graphml', node_type=int, edge_key_type=str) for i in tqdm(range(n_init)))
shuffled_graphs = [pg.Analyze_Nx_G(edges=x.edges, graph=x) for x in shuffled_graphs]


# %%
# sensories to outputs

dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')

target_names = ['dVNC', 'dSEZ', 'RGN']
targets = [dVNC_pair_ids, dSEZ_pair_ids, RGN_pair_ids]

sensories_pair_ids = [pm.Promat.load_pairs_from_annotation(x, pairs, return_type='all_pair_ids') for x in pymaid.get_annotated('mw brain inputs and ascending').name]
sensories_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'asc-proprio', 'asc-mechano', 'asc-classII-III', 'asc-noci']
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]

save_path = 'interhemisphere/plots/interhemisphere_crossings'
cutoff=6

all_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.crossing_counts)(graph.G, all_sensories, targets[i], cutoff=cutoff, plot=True, 
                                                                source_name='Sensories', target_name=target_names[i], save_path=save_path) for i in (range(len(targets))))

# comparison to random
all_paths_shuffled_test = Parallel(n_jobs=-1)(delayed(pg.Prograph.crossing_counts)(G = shuffled_graphs[i].G, source_list = all_sensories, targets = dVNC_pair_ids, cutoff=cutoff) for i in (range(100)))

shuffled_hists = []
for shuffled in all_paths_shuffled_test:
    binwidth = 1
    x_range = list(range(0, 7))
    data = shuffled[1]
    bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
    hist = np.histogram(data, bins=bins)
    for hist_pair in zip(hist[0], [x for x in range(len(hist[0]))], ['shuffled']*len(hist[0])):
        shuffled_hists.append(hist_pair)

shuffled_hists = pd.DataFrame(shuffled_hists, columns = ['count', 'bin', 'condition'])

control_hists = []
binwidth = 1
x_range = list(range(0, 7))
data = all_paths[0][1]
bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5
hist = np.histogram(data, bins=bins)
for hist_pair in zip(hist[0], [x for x in range(len(hist[0]))], ['control']*len(hist[0])):
    control_hists.append(hist_pair)

control_hists = pd.DataFrame(control_hists, columns = ['count', 'bin', 'condition'])

hists = pd.concat([shuffled_hists, control_hists], axis=0)
fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.barplot(data=hists, x='bin', y='count', hue='condition', ax=ax)

import pickle
output = open('interhemisphere/csv/interhemisphere_crossings/crossings_all.txt','wb')
data=hists
pickle.dump(data, output)
