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

#url = 'https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/drosophila/l1/seymour/'
rm = pymaid.CatmaidInstance(url, name, password, token)
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

import networkx as nx 

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dVNC_pairs = pm.Promat.extract_pairs_from_list(dVNC, pairs)[0]

dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dSEZ_pairs = pm.Promat.extract_pairs_from_list(dSEZ, pairs)[0]

RGN = pymaid.get_skids_by_annotation('mw RGN')
RGN_pairs = pm.Promat.extract_pairs_from_list(RGN, pairs)[0]

# directed graph
G = nx.DiGraph()
G_no_commissure = nx.DiGraph()
G_no_contra_edges = nx.DiGraph()
G_no_bilateral_contra_edges = nx.DiGraph()
G_no_bilateral_ipsi_edges = nx.DiGraph()

# build the graph
for i in range(len(all_edges_combined)):
    G.add_edge(all_edges_combined.iloc[i].upstream_pair_id, all_edges_combined.iloc[i].downstream_pair_id, 
                weight = np.mean([all_edges_combined.iloc[i].left, all_edges_combined.iloc[i].right]), 
                edge_type = all_edges_combined.iloc[i].type)

all_ipsi_edges = all_edges_combined[all_edges_combined.type=='ipsilateral']
all_ipsi_edges.reset_index(inplace=True, drop=True)
for i in range(len(all_ipsi_edges)):
    G_no_commissure.add_edge(all_ipsi_edges.iloc[i].upstream_pair_id, all_ipsi_edges.iloc[i].downstream_pair_id, 
                weight = np.mean([all_ipsi_edges.iloc[i].left, all_ipsi_edges.iloc[i].right]), 
                edge_type = all_ipsi_edges.iloc[i].type)

# %%
# sensories to outputs
from tqdm import tqdm
from joblib import Parallel, delayed

def path_edge_attributes(G_graph, path, attribute_name, include_skids=True):
    if(include_skids):
        return [(u,v,G_graph[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])]
    if(include_skids==False):
        return np.array([(G_graph[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])])

def crossing_counts(G_graph, source_list, targets, cutoff, plot=False, source_name=[], target_name=[], save_path = []):

    all_paths = [nx.all_simple_paths(G_graph, source, targets.leftid.values, cutoff=cutoff) for source in source_list]
    paths_list = [x for sublist in all_paths for x in sublist]
    paths_crossing_count = [sum(path_edge_attributes(G_graph, path, 'edge_type', False)=='contralateral') for path in paths_list]
    if(plot):
        # allows text to be editable in Illustrator
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        # font settings
        plt.rcParams['font.size'] = 5
        plt.rcParams['font.family'] = 'arial'

        binwidth = 1
        x_range = list(range(0, 7))
        data = paths_crossing_count
        bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5

        fig,ax = plt.subplots(1,1, figsize=(1.5, 2))
        sns.distplot(paths_crossing_count, bins=bins, kde=False, ax=ax, hist_kws={"rwidth":0.9,'alpha':0.75})
        ax.set(xlim = (-0.75, 6.75), ylabel=f'Number of Paths ({source_name} to {target_name})', xlabel='Number of Interhemisphere Crossings', xticks=[i for i in range(7)])
        plt.savefig(f'{save_path}/{source_name}-to-{target_name}_distribution.pdf', format='pdf', bbox_inches='tight')

    return(paths_list, paths_crossing_count)

sensories = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs and ascending').name]
sensories_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'asc-proprio', 'asc-mechano', 'asc-classII-III', 'asc-noci']
sensories_pairs = [pm.Promat.extract_pairs_from_list(x, pairs)[0] for x in sensories]

save_path = 'interhemisphere/plots/interhemisphere_crossings'

target_names = ['dVNC', 'dSEZ', 'RGN']
targets = [dVNC_pairs, dSEZ_pairs, RGN_pairs]
all_paths_ORN = Parallel(n_jobs=-1)(delayed(crossing_counts)(G, sensories_pairs[0].leftid, targets[i], cutoff=6, plot=True, 
                                                                source_name='ORN', target_name=target_names[i], save_path=save_path) for i in (range(len(targets))))

all_paths_thermo = Parallel(n_jobs=-1)(delayed(crossing_counts)(G, sensories_pairs[1].leftid, targets[i], cutoff=6, plot=True, 
                                                                source_name='thermo', target_name=target_names[i], save_path=save_path) for i in (range(len(targets))))

all_paths_noci = Parallel(n_jobs=-1)(delayed(crossing_counts)(G, sensories_pairs[-1].leftid, targets[i], cutoff=6, plot=True, 
                                                                source_name='noci', target_name=target_names[i], save_path=save_path) for i in (range(len(targets))))

all_paths_ORN_no_commissure = Parallel(n_jobs=-1)(delayed(crossing_counts)(G_no_commissure, sensories_pairs[0].leftid, targets[i], cutoff=6, plot=True, 
                                                                source_name='no-comissure_ORN', target_name=target_names[i], save_path=save_path) for i in (range(len(targets))))

