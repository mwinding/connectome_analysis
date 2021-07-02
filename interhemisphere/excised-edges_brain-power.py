#%%
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

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

import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import connectome_tools.cascade_analysis as casc
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx 
import pickle

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# load pairs
pairs = pm.Promat.get_pairs()

ipsi_pair_ids = pm.Promat.load_pairs_from_annotation('mw ipsilateral axon', pairs, return_type='all_pair_ids')
bilateral_pair_ids = pm.Promat.load_pairs_from_annotation('mw bilateral axon', pairs, return_type='all_pair_ids')
contra_pair_ids = pm.Promat.load_pairs_from_annotation('mw contralateral axon', pairs, return_type='all_pair_ids')

dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')
output_pair_ids = RGN_pair_ids + dSEZ_pair_ids + dVNC_pair_ids

all_sensories = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities')
all_outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')
all_sensories_left = list(pm.Promat.extract_pairs_from_list(all_sensories, pairs)[0].leftid)
all_outputs_left = list(pm.Promat.extract_pairs_from_list(all_outputs, pairs)[0].leftid)

# %%
##########
# EXPERIMENT 1: removing random number of ipsi vs contra edges, effect on nodes touched by all paths and number of raw visits from cascades
#

# load previously generated paths
all_edges_combined_split = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

left = pm.Promat.get_hemis('left')
right = pm.Promat.get_hemis('right')

# iterations for random edge removal
n_init = 8

# generate wildtype graph
split_graph = pg.Analyze_Nx_G(all_edges_combined_split, graph_type='directed', split_pairs=True)

# excise edges and generate graphs
random_ipsi500, random_contra500 = pg.Prograph.excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 500, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi1000, random_contra1000 = pg.Prograph.excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 1000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi2000, random_contra2000 = pg.Prograph.excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 2000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi4000, random_contra4000 = pg.Prograph.excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 4000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi8000, random_contra8000 = pg.Prograph.excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 8000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)

# %%
# generate and save paths through brain
cutoff = 5

save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_wildtype'
pg.Prograph.generate_save_simple_paths(split_graph.G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=save_path)

# generate and save paths
count = 500
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi500[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra500[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 1000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi1000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra1000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 2000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi2000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra2000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 4000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi4000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra4000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 8000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi8000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra8000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

# %%
# open data and count nodes

cutoff=5
n_init = 8
counts = [500, 1000, 2000, 4000, 8000]

'''
counts = [500, 1000, 2000, 4000, 8000]

removed_ipsi_nodes_list = []
removed_contra_nodes_list = []
for count in counts:
    random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
    random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
    
    ipsi_nodes_list = []
    for paths in random_ipsi_paths:
        ipsi_nodes_list.append(len(np.unique([x for sublist in paths for x in sublist])))
    
    contra_nodes_list = []
    for paths in random_contra_paths:
        contra_nodes_list.append(len(np.unique([x for sublist in paths for x in sublist])))
     
    removed_ipsi_nodes_list.append(ipsi_nodes_list)
    removed_contra_nodes_list.append(contra_nodes_list)
'''

def count_nodes_in_pathsList(pathsList):
    return([len(np.unique([x for sublist in paths for x in sublist])) for paths in pathsList])

removed_ipsi_nodes_list = []
removed_contra_nodes_list = []

count=500
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=1000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=2000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=4000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=8000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

control_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/all-paths-sens-to-alloutputs_cutoff{cutoff}_wildtype.csv.gz')
control_nodes = len(np.unique([x for sublist in control_paths for x in sublist]))

# generate appropriate dataframe

data = []
for i, iterations in enumerate(removed_ipsi_nodes_list):
    data.append(list(zip(iterations, [counts[i]]*len(counts), ['ipsi']*len(counts))))

for i, iterations in enumerate(removed_contra_nodes_list):
    data.append(list(zip(iterations, [counts[i]]*len(counts), ['contra']*len(counts))))
data = [x for sublist in data for x in sublist]

data = pd.DataFrame(data, columns = ['nodes', 'removed', 'type'])
data.loc[:, 'nodes'] = data.loc[:, 'nodes']/control_nodes

# plot data
fig, ax = plt.subplots(1,1, figsize=(1,1.25))
sns.pointplot(data = data, x='type', y='nodes', hue='removed', scale=0.25, ax=ax, errwidth=1)
ax.set(ylim=(0,1.1))
plt.savefig('interhemisphere/csv/paths/random-ipsi-contra-excised_brain-power/nodes-in-paths_left-to-left_removing-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
# EXPERIMENT 2: (depends on EXPERIMENT 1) 
# run cascades through graphs; count total "activation events" (visits) and also total nodes that receive over threshold visits

inputs = pd.read_csv('data/graphs/inputs.csv', header = 0, index_col = 0)

def fraction_input_to_raw_adj(adj, inputs):
    for column in adj.columns:
        den_input = inputs.loc[column].dendrite_input
        adj.loc[:, column] = adj.loc[:, column]*den_input
    return(adj)

def total_conversion(graph, inputs):
    df = pd.DataFrame(nx.adjacency_matrix(G=graph.G, weight = 'weight').todense(), columns = graph.G.nodes, index = graph.G.nodes)
    df = fraction_input_to_raw_adj(df, inputs)
    return(df)

# convert graphs to adj matrices; switch from %input to raw synapses for cascades
all_ipsi_graphs = [random_ipsi500, random_ipsi1000, random_ipsi2000, random_ipsi4000, random_ipsi8000]
all_contra_graphs = [random_contra500, random_contra1000, random_contra2000, random_contra4000, random_contra8000]

all_ipsi_adjs = []
for ipsi_graphs in all_ipsi_graphs:
    ipsi_adjs = Parallel(n_jobs=-1)(delayed(total_conversion)(ipsi_graphs[i], inputs) for i in tqdm(range(len(ipsi_graphs))))
    all_ipsi_adjs.append(ipsi_adjs)

all_contra_adjs = []
for ipsi_graphs in all_contra_graphs:
    ipsi_adjs = Parallel(n_jobs=-1)(delayed(total_conversion)(ipsi_graphs[i], inputs) for i in tqdm(range(len(ipsi_graphs))))
    all_contra_adjs.append(ipsi_adjs)

# save as pickle file because it's so slow
pickle.dump(all_ipsi_adjs, open('interhemisphere/cascades/left-left-adjs_ipsi-edges-removed.p', 'wb'))
pickle.dump(all_contra_adjs, open('interhemisphere/cascades/left-left-adjs_contra-edges-removed.p', 'wb'))

# %%
# cascades from edge-excised graphs
all_ipsi_adjs = pickle.load(open('interhemisphere/cascades/left-left-adjs_ipsi-edges-removed.p', 'rb'))
all_contra_adjs = pickle.load(open('interhemisphere/cascades/left-left-adjs_contra-edges-removed.p', 'rb'))

p = 0.05
max_hops = 8
n_init = 100
simultaneous = True

ipsi_hit_hist_list = []
for adjs in all_ipsi_adjs:
    hit_hist_list = []
    for adj in adjs:
        hit_hist = casc.Cascade_Analyzer.run_single_cascade(name='ipsi', source_skids=all_sensories_left, stop_skids=all_outputs_left, adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
        hit_hist_list.append(hit_hist)
    ipsi_hit_hist_list.append(hit_hist_list)

contra_hit_hist_list = []
for adjs in all_contra_adjs:
    hit_hist_list = []
    for adj in adjs:
        hit_hist = casc.Cascade_Analyzer.run_single_cascade(name='contra', source_skids=all_sensories_left, stop_skids=all_outputs_left, adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
        hit_hist_list.append(hit_hist)
    contra_hit_hist_list.append(hit_hist_list)

# save cascades as pickle file because it's so slow
pickle.dump(ipsi_hit_hist_list, open(f'interhemisphere/cascades/left-left-cascades_{n_init}-n_init_ipsi-removed-hit-hist-list.p', 'wb'))
pickle.dump(contra_hit_hist_list, open(f'interhemisphere/cascades/left-left-cascades_{n_init}-n_init_contra-removed-hit-hist-list.p', 'wb'))

# %%
# extract visit data
ipsi_hit_hist_list = pickle.load(open(f'interhemisphere/cascades/left-left-cascades_{n_init}-n_init_ipsi-removed-hit-hist-list.p', 'rb'))
contra_hit_hist_list = pickle.load(open(f'interhemisphere/cascades/left-left-cascades_{n_init}-n_init_contra-removed-hit-hist-list.p', 'rb'))

# generate appropriate dataframe

# plot data

