#%%

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random
import gzip
import csv

from contools import Celltype, Celltype_Analyzer, Promat, Prograph, Cascade_Analyzer, Analyze_Nx_G
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
pairs = Promat.get_pairs(pairs_path=pairs_path)

ipsi_pair_ids = Promat.load_pairs_from_annotation('mw ipsilateral axon', pairs, return_type='all_pair_ids')
#contra_contra = Promat.load_pairs_from_annotation(['mw contralateral axon', 'mw contralateral dendrite'], pairs, return_type='all_pair_ids')
bilateral_pair_ids = Promat.load_pairs_from_annotation('mw bilateral axon', pairs, return_type='all_pair_ids')
contra_pair_ids = Promat.load_pairs_from_annotation('mw contralateral axon', pairs, return_type='all_pair_ids')

dVNC_pair_ids = Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')
#output_pair_ids = RGN_pair_ids + dSEZ_pair_ids + dVNC_pair_ids
output_pair_ids = dVNC_pair_ids

all_sensories = Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities')
all_outputs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

left = pymaid.get_skids_by_annotation('mw left')
all_sensories_left = list(np.intersect1d(Promat.extract_pairs_from_list(all_sensories, pairs, return_pair_ids=True), left))
#all_outputs_left = list(pm.Promat.extract_pairs_from_list(all_outputs, pairs)[0].leftid)
all_outputs_left = dVNC_pair_ids

# %%
##########
# EXPERIMENT 1: removing random number of ipsi vs contra edges, effect on nodes touched by all paths and number of raw visits from cascades
#

# load previously generated paths
all_edges_combined_split = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=False)

contra_contra = np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')) # neurons with somas on opposite cf. to neurites
left = Promat.get_hemis(left_annot='mw left', right_annot='mw right', side='left', neurons_to_flip=contra_contra)
right = Promat.get_hemis(left_annot='mw left', right_annot='mw right', side='right', neurons_to_flip=contra_contra)

# iterations for random edge removal
n_init = 10

# generate wildtype graph
split_graph = Analyze_Nx_G(all_edges_combined_split, graph_type='directed', split_pairs=True)

def excise_ipsi_contra_edge_experiment_whole_brain(edges, edge_count, n_init, seed, exclude_nodes=[], split_pairs=False):
    # preparing edge lists
    excised_random_ipsi_edges_list = Prograph.random_excise_edges_type(edges, 'ipsilateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=exclude_nodes)
    excised_random_contra_edges_list = Prograph.random_excise_edges_type(edges, 'contralateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=exclude_nodes)

    # loading into graphs
    random_ipsi_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_ipsi_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_ipsi_edges_list))))
    random_contra_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_contra_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_contra_edges_list))))

    return(random_ipsi_graphs, random_contra_graphs)

# excise edges and generate graphs
random_ipsi500, random_contra500 = excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 500, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi1000, random_contra1000 = excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 1000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi2000, random_contra2000 = excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 2000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi4000, random_contra4000 = excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 4000, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True)
random_ipsi8000, random_contra8000 = excise_ipsi_contra_edge_experiment_whole_brain(all_edges_combined_split, 7487, n_init, 0, exclude_nodes=(all_sensories+all_outputs), split_pairs=True) # not quite 8000, more like ~7500

# %%
# generate and save paths through brain
cutoff = 5
#outputs = 'alloutputs'
outputs = 'dVNCs'

save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_wildtype'
Prograph.generate_save_simple_paths(split_graph.G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=save_path)

# generate and save paths
count = 500
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_ipsi500[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_contra500[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 1000
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_ipsi1000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_contra1000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 2000
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_ipsi2000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_contra2000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 4000
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_ipsi4000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_contra4000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 7487
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_ipsi8000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(random_contra8000[i].G, all_sensories_left, all_outputs_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

# %%
# open data and count nodes

cutoff=5
n_init = 8
counts = [500, 1000, 2000, 4000, 7487]

'''
counts = [500, 1000, 2000, 4000, 7487]

removed_ipsi_nodes_list = []
removed_contra_nodes_list = []
for count in counts:
    random_ipsi_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
    random_contra_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
    
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
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=1000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=2000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=4000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

count=7487
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList(random_ipsi_paths))
removed_contra_nodes_list.append(count_nodes_in_pathsList(random_contra_paths))

control_paths = pg.Prograph.open_simple_paths(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_wildtype.csv.gz')
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
plt.savefig('data/csv/paths/random-ipsi-contra-excised/nodes-in-paths_left-to-left_removing-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
# nodes in left hemisphere in paths


def count_nodes_in_pathsList_side(pathsList, side):
    nodes = [len(np.unique([x for sublist in paths for x in sublist if x in side])) for paths in pathsList]
    return(nodes)

n_init=8
outputs = 'dVNCs'
cutoff=5
left = pm.Promat.get_hemis('left')

removed_ipsi_nodes_list = []
removed_contra_nodes_list = []

count=500
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, left))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, left))

count=1000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, left))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, left))

count=2000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, left))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, left))

count=4000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, left))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, left))

count=7487
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, left))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, left))

control_paths = pg.Prograph.open_simple_paths(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_wildtype.csv.gz')
control_nodes = len(np.unique([x for sublist in control_paths for x in sublist if x in left]))

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
plt.savefig('data/csv/paths/random-ipsi-contra-excised/left_nodes-in-paths_left-to-left_removing-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
# nodes in right hemisphere in paths

def count_nodes_in_pathsList_side(pathsList, side):
    nodes = [len(np.unique([x for sublist in paths for x in sublist if x in side])) for paths in pathsList]
    return(nodes)

right = pm.Promat.get_hemis('right')
n_init=8
outputs = 'dVNCs'
cutoff=5

removed_ipsi_nodes_list = []
removed_contra_nodes_list = []

count=500
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, right))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, right))

count=1000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, right))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, right))

count=2000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, right))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, right))

count=4000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, right))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, right))

count=7487
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-ipsi-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_minus-edges-contra-{count}-N{i}.csv.gz') for i in tqdm(range(n_init)))
removed_ipsi_nodes_list.append(count_nodes_in_pathsList_side(random_ipsi_paths, right))
removed_contra_nodes_list.append(count_nodes_in_pathsList_side(random_contra_paths, right))

control_paths = pg.Prograph.open_simple_paths(f'data/csv/paths/random-ipsi-contra-excised/all-paths-sens-to-{outputs}_cutoff{cutoff}_wildtype.csv.gz')
control_nodes = len(np.unique([x for sublist in control_paths for x in sublist if x in right]))

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
plt.savefig('data/csv/paths/random-ipsi-contra-excised/right_nodes-in-paths_left-to-left_removing-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
# EXPERIMENT 2: (depends on EXPERIMENT 1) 
# run cascades through graphs; count total "activation events" (visits) and also total nodes that receive over threshold visits

inputs = pd.read_csv(f'data/adj/inputs_{data_date}.csv', header = 0, index_col = 0)

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
pickle.dump(all_ipsi_adjs, open('data/cascades/edge_excision/adjs_ipsi-edges-removed.p', 'wb'))
pickle.dump(all_contra_adjs, open('data/cascades/edge_excision/adjs_contra-edges-removed.p', 'wb'))

# %%
# cascades from edge-excised graphs
all_ipsi_adjs = pickle.load(open('data/cascades/edge_excision/adjs_ipsi-edges-removed.p', 'rb'))
all_contra_adjs = pickle.load(open('data/cascades/edge_excision/adjs_contra-edges-removed.p', 'rb'))

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True

ipsi_hit_hist_list = []
for adjs in all_ipsi_adjs:
    hit_hist_list = []
    for adj in adjs:
        hit_hist = Cascade_Analyzer.run_single_cascade(name='ipsi', source_skids=all_sensories, stop_skids=all_outputs, adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
        hit_hist_list.append(hit_hist)
    ipsi_hit_hist_list.append(hit_hist_list)

contra_hit_hist_list = []
for adjs in all_contra_adjs:
    hit_hist_list = []
    for adj in adjs:
        hit_hist = Cascade_Analyzer.run_single_cascade(name='contra', source_skids=all_sensories, stop_skids=all_outputs, adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
        hit_hist_list.append(hit_hist)
    contra_hit_hist_list.append(hit_hist_list)

# save cascades as pickle file because it's so slow
pickle.dump(ipsi_hit_hist_list, open(f'data/cascades/edge_excision/all-all-cascades_{n_init}-n_init_ipsi-removed-hit-hist-list.p', 'wb'))
pickle.dump(contra_hit_hist_list, open(f'data/cascades/edge_excision/all-all-cascades_{n_init}-n_init_contra-removed-hit-hist-list.p', 'wb'))

# %%
n_init = 100

dVNCs = pymaid.get_skids_by_annotation('mw dVNC')
dSEZs = pymaid.get_skids_by_annotation('mw dSEZ')
RGNs = pymaid.get_skids_by_annotation('mw RGN')

# extract visit data
ipsi_hit_hist_list = pickle.load(open(f'data/cascades/edge_excision/all-all-cascades_{n_init}-n_init_ipsi-removed-hit-hist-list.p', 'rb'))
contra_hit_hist_list = pickle.load(open(f'data/cascades/edge_excision/all-all-cascades_{n_init}-n_init_contra-removed-hit-hist-list.p', 'rb'))

# control graph and cascade
control_adj = total_conversion(split_graph, inputs)
control_cascade = Cascade_Analyzer.run_single_cascade(name='control', source_skids=all_sensories, stop_skids=all_outputs, adj=control_adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
hit_hist = control_cascade.skid_hit_hist
control_count_dVNCs = sum(hit_hist.loc[np.intersect1d(dVNCs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
control_count_dSEZs = sum(hit_hist.loc[np.intersect1d(dSEZs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
control_count_RGNs = sum(hit_hist.loc[np.intersect1d(RGNs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
control_count_all = sum(hit_hist.loc[:, 1:10].sum(axis=1)>=(n_init/2))

counts = [500,1000,2000,4000,7487]

# generate appropriate dataframes
output_counts = []
for i, hit_hists in enumerate(ipsi_hit_hist_list):
    for j, hit_hist in enumerate(hit_hists):
        hit_hist = hit_hist.skid_hit_hist
        count_dVNCs = sum(hit_hist.loc[np.intersect1d(dVNCs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
        count_dSEZs = sum(hit_hist.loc[np.intersect1d(dSEZs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
        count_RGNs = sum(hit_hist.loc[np.intersect1d(RGNs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
        count_all = sum(hit_hist.loc[:, 1:10].sum(axis=1)>=(n_init/2))
        output_counts.append([count_dVNCs/control_count_dVNCs, 'dVNC', counts[i], j, 'ipsi'])
        output_counts.append([count_dSEZs/control_count_dSEZs, 'dSEZ', counts[i], j, 'ipsi'])
        output_counts.append([count_RGNs/control_count_RGNs, 'RGN', counts[i], j, 'ipsi'])
        output_counts.append([count_all/control_count_all, 'all', counts[i], j, 'ipsi'])

for i, hit_hists in enumerate(contra_hit_hist_list):
    for j, hit_hist in enumerate(hit_hists):
        hit_hist = hit_hist.skid_hit_hist
        count_dVNCs = sum(hit_hist.loc[np.intersect1d(dVNCs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
        count_dSEZs = sum(hit_hist.loc[np.intersect1d(dSEZs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
        count_RGNs = sum(hit_hist.loc[np.intersect1d(RGNs, hit_hist.index), :].sum(axis=1)>=(n_init/2))
        count_all = sum(hit_hist.loc[:, 1:10].sum(axis=1)>=(n_init/2))
        output_counts.append([count_dVNCs/control_count_dVNCs, 'dVNC', counts[i], j, 'contra'])
        output_counts.append([count_dSEZs/control_count_dSEZs, 'dSEZ', counts[i], j, 'contra'])
        output_counts.append([count_RGNs/control_count_RGNs, 'RGN', counts[i], j, 'contra'])
        output_counts.append([count_all/control_count_all, 'all', counts[i], j, 'contra'])

output_node_counts = pd.DataFrame(output_counts, columns = ['count', 'output_type', 'edges_removed', 'N', 'edge_type'])

# number of nodes activated
size=(0.6, 1)
palette = [sns.color_palette()[0], sns.color_palette()[2]]
output_types = ['dVNC', 'dSEZ', 'RGN', 'all']
for output_type in output_types:
    fig, ax = plt.subplots(1,1,figsize=size)
    sns.lineplot(data=output_node_counts[output_node_counts.output_type==output_type], x='edges_removed', y='count', hue='edge_type', palette=palette, errorbar='sd', err_style='bars', linewidth=0.5, err_kws={'elinewidth':0.5})
    ax.set(ylim=(0, 1), xticks=(0, 2000, 4000, 6000, 8000), xlim = (0, 8200))
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'plots/random-edge-excision_{output_type}_activated-nodes-in-cascades.pdf', format='pdf', bbox_inches='tight')

# stats on data
from scipy.stats import mannwhitneyu
data = output_node_counts

mannwhitneyu_nodes_tests = []
for output_type in output_types:
    for edge_count in counts:
        data1 = data[(data.output_type==output_type) & (data.edges_removed==edge_count) & (data.edge_type=='ipsi')].loc[:, 'count'].values
        data2 = data[(data.output_type==output_type) & (data.edges_removed==edge_count) & (data.edge_type=='contra')].loc[:, 'count'].values
        test = mannwhitneyu(data1, data2, alternative='two-sided', method='exact')
        mannwhitneyu_nodes_tests.append([output_type, edge_count, test[1], data1, data2])

mannwhitneyu_nodes_tests = pd.DataFrame(mannwhitneyu_nodes_tests, columns=['output_type', 'edges_removed', 'mann_whitney_pvalue', 'data1', 'data2'])
mannwhitneyu_nodes_tests

# %%
# number of visits total
hit_hist = control_cascade.skid_hit_hist
control_count_dVNCs = sum(hit_hist.loc[np.intersect1d(dVNCs, hit_hist.index), :].sum(axis=1))
control_count_dSEZs = sum(hit_hist.loc[np.intersect1d(dSEZs, hit_hist.index), :].sum(axis=1))
control_count_RGNs = sum(hit_hist.loc[np.intersect1d(RGNs, hit_hist.index), :].sum(axis=1))
control_count_all = sum(hit_hist.loc[:, 1:10].sum(axis=1))

counts = [500,1000,2000,4000,7487]

# generate appropriate dataframes
output_counts = []
for i, hit_hists in enumerate(ipsi_hit_hist_list):
    for j, hit_hist in enumerate(hit_hists):
        hit_hist = hit_hist.skid_hit_hist
        count_dVNCs = sum(hit_hist.loc[np.intersect1d(dVNCs, hit_hist.index), :].sum(axis=1))
        count_dSEZs = sum(hit_hist.loc[np.intersect1d(dSEZs, hit_hist.index), :].sum(axis=1))
        count_RGNs = sum(hit_hist.loc[np.intersect1d(RGNs, hit_hist.index), :].sum(axis=1))
        count_all = sum(hit_hist.loc[:, 1:10].sum(axis=1))
        output_counts.append([count_dVNCs/control_count_dVNCs, 'dVNC', counts[i], j, 'ipsi'])
        output_counts.append([count_dSEZs/control_count_dSEZs, 'dSEZ', counts[i], j, 'ipsi'])
        output_counts.append([count_RGNs/control_count_RGNs, 'RGN', counts[i], j, 'ipsi'])
        output_counts.append([count_all/control_count_all, 'all', counts[i], j, 'ipsi'])

for i, hit_hists in enumerate(contra_hit_hist_list):
    for j, hit_hist in enumerate(hit_hists):
        hit_hist = hit_hist.skid_hit_hist
        count_dVNCs = sum(hit_hist.loc[np.intersect1d(dVNCs, hit_hist.index), :].sum(axis=1))
        count_dSEZs = sum(hit_hist.loc[np.intersect1d(dSEZs, hit_hist.index), :].sum(axis=1))
        count_RGNs = sum(hit_hist.loc[np.intersect1d(RGNs, hit_hist.index), :].sum(axis=1))
        count_all = sum(hit_hist.loc[:, 1:10].sum(axis=1))
        output_counts.append([count_dVNCs/control_count_dVNCs, 'dVNC', counts[i], j, 'contra'])
        output_counts.append([count_dSEZs/control_count_dSEZs, 'dSEZ', counts[i], j, 'contra'])
        output_counts.append([count_RGNs/control_count_RGNs, 'RGN', counts[i], j, 'contra'])
        output_counts.append([count_all/control_count_all, 'all', counts[i], j, 'contra'])

output_visit_counts = pd.DataFrame(output_counts, columns = ['count', 'output_type', 'edges_removed', 'N', 'edge_type'])

# number of visits total
output_type = 'dVNC'
palette = [sns.color_palette()[0], sns.color_palette()[2]]
output_types = ['dVNC', 'dSEZ', 'RGN', 'all']

for output_type in output_types:
    fig, ax = plt.subplots(1,1,figsize=size)
    sns.lineplot(data=output_visit_counts[output_visit_counts.output_type==output_type], x='edges_removed', y='count', hue='edge_type', palette=palette, err_style='bars', linewidth=0.5, errorbar='sd', err_kws={'elinewidth':0.5})
    ax.set(ylim=(0, 1), xticks=(0, 2000, 4000, 6000, 8000), xlim = (0, 8200))
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'plots/random-edge-excision_{output_type}_total-visits-in-cascades.pdf', format='pdf', bbox_inches='tight')

# stats on data
data = output_visit_counts

mannwhitneyu_visits_tests = []
for output_type in output_types:
    for edge_count in counts:
        data1 = data[(data.output_type==output_type) & (data.edges_removed==edge_count) & (data.edge_type=='ipsi')].loc[:, 'count'].values
        data2 = data[(data.output_type==output_type) & (data.edges_removed==edge_count) & (data.edge_type=='contra')].loc[:, 'count'].values
        test = mannwhitneyu(data1, data2, alternative='two-sided', method='exact')
        mannwhitneyu_visits_tests.append([output_type, edge_count, test[1], data1, data2])

mannwhitneyu_visits_tests = pd.DataFrame(mannwhitneyu_visits_tests, columns=['output_type', 'edges_removed', 'mann_whitney_pvalue', 'data1', 'data2'])
mannwhitneyu_visits_tests

# %%
