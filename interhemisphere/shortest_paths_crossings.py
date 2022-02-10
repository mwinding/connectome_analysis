#%%

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random
import gzip 
import csv

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import connectome_tools.celltype as ct
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx 

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# load adjacency matrix, graphs, and pairs
adj = pm.Promat.pull_adj('ad', subgraph='brain and accessory')
pairs = pm.Promat.get_pairs()

ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
ad_edges_split = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

graph = pg.Analyze_Nx_G(ad_edges)
graph_split = pg.Analyze_Nx_G(ad_edges_split, split_pairs=True)

pairs = pm.Promat.get_pairs()

# %%
# calculate shortest paths

dVNC_pair_ids = list(pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_sorted').leftid)
dSEZ_pair_ids = list(pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_sorted').leftid)
RGN_pair_ids = list(pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_sorted').leftid)

target_names = ['dVNC', 'dSEZ', 'RGN']
targets = [dVNC_pair_ids, dSEZ_pair_ids, RGN_pair_ids]

sensories_names = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sensories_skids = [ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}') for name in sensories_names]
sensories_pair_ids = [pm.Promat.load_pairs_from_annotation(annot='', pairList=pairs, return_type='all_pair_ids', skids=celltype, use_skids=True) for celltype in sensories_skids]
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
# interhemisphere shortest paths to all outputs
output_pair_ids = dVNC_pair_ids + dSEZ_pair_ids + RGN_pair_ids

shortest_paths = []
for i in range(len(all_sensories)):
    sens_shortest_paths = []
    for j in range(len(output_pair_ids)):
        try:
            shortest_path = nx.shortest_path(graph.G, all_sensories[i], output_pair_ids[j])
            sens_shortest_paths.append(shortest_path)
        except:
            print(f'probably no path exists from {all_sensories[i]}-{output_pair_ids[j]}') 

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
fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.barplot(data=control_hists, x='bin', y='count', hue='condition', ax=ax)
ax.set(xlim=(-0.75, 7.75))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_output.pdf', bbox_inches='tight')

# plot as fraction total paths

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.barplot(data=control_hists, x='bin', y='fraction', hue='condition', ax=ax)
ax.set(xlim=(-0.75, 7.75), ylim=(0,0.3))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_output_fraction-total-paths.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.barplot(data=control_hists[control_hists.bin%2==0], x='bin', y='fraction', hue='condition', ax=ax)
ax.set(xlim=(-0.75, 7.75), ylim=(0,0.3))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_output_fraction-total-paths_even.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.barplot(data=control_hists[control_hists.bin%2==1], x='bin', y='fraction', hue='condition', ax=ax)
ax.set(xlim=(-0.75, 7.75), ylim=(0,0.3))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_output_fraction-total-paths_odd.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.barplot(data=control_hists.set_index('bin', drop=False).loc[[0,2,4,6,8,1,3,5,7], :], x=['0','2','4','6','8','1','3','5','7'], y='fraction', hue='condition', ax=ax)
ax.set(xlim=(-0.75, 7.75), ylim=(0,0.3))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/shortest-paths_sens_to_output_fraction-total-paths_even-odd.pdf', bbox_inches='tight')

# %%
# fraction of paths with crossings

data = pd.DataFrame([[1, control_hists[control_hists.bin==0].fraction.values[0], 'no_crossing'],
        [1,sum(control_hists[control_hists.bin>0].fraction), 'crossing']], columns=['celltype','fraction', 'condition'])

fig, ax = plt.subplots(1,1, figsize=(0.35,0.75))
sns.barplot(data = data, x = 'celltype', y='fraction', hue='condition', ax=ax)
ax.set(ylim=(0,1))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/no-crossing_vs_crossing.pdf', bbox_inches='tight')


data = pd.DataFrame([[1, sum(control_hists[control_hists.bin%2==0].fraction), 'ipsilateral'],
        [1,sum(control_hists[control_hists.bin%2==1].fraction), 'contralateral']], columns=['celltype','fraction', 'condition'])

fig, ax = plt.subplots(1,1, figsize=(0.35,0.75))
sns.barplot(data = data, x = 'celltype', y='fraction', hue='condition', ax=ax)
ax.set(ylim=(0,1))
plt.savefig('interhemisphere/plots/interhemisphere_crossings/ipsi-crossing_vs_contra-crossing.pdf', bbox_inches='tight')

# %%
