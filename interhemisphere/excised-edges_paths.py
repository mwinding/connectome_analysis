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

ipsi_pair_ids = pm.Promat.load_pairs_from_annotation('mw ipsilateral axon', pairs, return_type='all_pair_ids')
bilateral_pair_ids = pm.Promat.load_pairs_from_annotation('mw bilateral axon', pairs, return_type='all_pair_ids')
contra_pair_ids = pm.Promat.load_pairs_from_annotation('mw contralateral axon', pairs, return_type='all_pair_ids')

dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')

sensories_pair_ids = [pm.Promat.load_pairs_from_annotation(x, pairs, return_type='all_pair_ids') for x in pymaid.get_annotated('mw brain inputs').name]
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]
# %%
# EXPERIMENT 1: removing edges from contralateral and bilateral neurons -> effect on path length?
# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

# iterations for random edge removal as control
n_init = 40

# excise edges and generate graphs
e_contra_contra, e_contra_contra_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(contra_pair_ids, all_sensories), 'contralateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
e_bi_contra, e_bi_contra_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(bilateral_pair_ids, all_sensories), 'contralateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
e_bi_ipsi, e_bi_ipsi_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(bilateral_pair_ids, all_sensories), 'ipsilateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
e_all_contra, e_all_contra_control = pg.Prograph.excise_edge_experiment(all_edges_combined, np.setdiff1d(bilateral_pair_ids + contra_pair_ids, all_sensories), 'contralateral', n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))

# %%
# this chunk is incomplete
# write all graphs to graphml

# read all graph from graphml
graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')

shuffled_graphs = Parallel(n_jobs=-1)(delayed(nx.readwrite.graphml.read_graphml)(f'interhemisphere/csv/shuffled_graphs/iteration-{i}.graphml', node_type=int, edge_key_type=str) for i in tqdm(range(n_init)))
shuffled_graphs = [pg.Analyze_Nx_G(edges=x.edges, graph=x) for x in shuffled_graphs]
# %%
# generate and save paths
cutoff=5

# generate and save paths for experimental
save_path = [f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-contra',
                f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-contra',
                f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-ipsi',
                f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-all-contra']

experimental = [e_contra_contra, e_bi_contra, e_bi_ipsi, e_all_contra]
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(experimental[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=save_path[i]) for i in tqdm((range(len(experimental)))))

# generate and save paths for controls
save_path = f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-contra_CONTROL-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(e_contra_contra_control[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

save_path = f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-contra_CONTROL-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(e_bi_contra_control[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

save_path = f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-ipsi_CONTROL-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(e_bi_ipsi_control[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

save_path = f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-all-contra_CONTROL-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(e_all_contra_control[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

# %%
# analyze paths: total and count per # hops
def process_paths(excise_paths, control_paths, edges_removed):
    excise_count = len(excise_paths)
    control_counts = [len(x) for x in control_paths]

    path_counts_data = []
    for row in zip(control_counts, [f'control-{edges_removed}']*len(control_counts)):
        path_counts_data.append(row)

    path_counts_data.append([excise_count, f'excised-{edges_removed}'])
    path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition'])
    path_counts_data.to_csv(f'interhemisphere/csv/paths/processed/excised_graph_{edges_removed}.csv')

    # count per # hops
    excise_path_counts = [len(x) for x in excise_paths]
    control_path_counts = [[len(x) for x in path] for path in control_paths]

    path_counts_length_data = []
    for i, path_length in enumerate(control_path_counts):
        for row in zip(path_length, [f'control-{edges_removed}']*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    for row in zip(excise_path_counts, [f'excised-{edges_removed}']*len(excise_path_counts), [0]*len(excise_path_counts)):
        path_counts_length_data.append(row)

    path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'N'])

    path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
    path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'path_length']).count()
    path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/processed/excised_graph_{edges_removed}_path_lengths.csv')

cutoff=5
n_init = 40

excise_Cc_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-contra.csv.gz')
control_Cc_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-contra_CONTROL-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths(excise_Cc_paths, control_Cc_paths, edges_removed='Contra-contra')

excise_Bc_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-contra.csv.gz')
control_Bc_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-contra_CONTROL-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths(excise_Bc_paths, control_Bc_paths, edges_removed='Bilateral-contra')

excise_Bi_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-ipsi.csv.gz')
control_Bi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-bilateral-ipsi_CONTROL-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths(excise_Bi_paths, control_Bi_paths, edges_removed='Bilateral-ipsi')

excise_Ac_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-all-contra.csv.gz')
control_Ac_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/excised-graph_all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-all-contra_CONTROL-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths(excise_Ac_paths, control_Ac_paths, edges_removed='All-contra')

# wildtype paths
graph_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/all_paths_sens-to-dVNC_cutoff{cutoff}.csv.gz')
excise_count = len(graph_paths)
path_counts_data = []
path_counts_data.append([excise_count, f'wildtype'])
path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition'])
path_counts_data.to_csv(f'interhemisphere/csv/paths/processed/wildtype.csv')

path_counts_length_data = []
excise_path_counts = [len(x) for x in graph_paths]
for row in zip(excise_path_counts, [f'wildtype']*len(excise_path_counts), [0]*len(excise_path_counts)):
    path_counts_length_data.append(row)

path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'N'])
path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'path_length']).count()
path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/processed/wildtype_path_lengths.csv')


# %%
##########
# EXPERIMENT 2: removing random number of ipsi vs contra edges, effect on paths
# 

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

# iterations for random edge removal as control
n_init = 8

# excise edges and generate graphs
random_ipsi500, random_contra500 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined, 500, n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
random_ipsi1000, random_contra1000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined, 1000, n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
random_ipsi2000, random_contra2000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined, 2000, n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))
random_ipsi4000, random_contra4000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined, 4000, n_init, seed=0, exclude_nodes=(all_sensories+dVNC_pair_ids))

# %%
# generate and save paths
cutoff=5

# generate and save paths for controls
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-500-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi500[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-500-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra500[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-1000-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi1000[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-1000-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra1000[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-2000-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi2000[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-2000-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra2000[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-4000-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi4000[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-4000-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra4000[i].G, all_sensories, dVNC_pair_ids, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

# %%
# analyze paths: total and count per # hops
def process_paths_ipsi_contra(ipsi_paths, contra_paths, count_removed):
    ipsi_counts = [len(x) for x in ipsi_paths]
    contra_counts = [len(x) for x in contra_paths]

    path_counts_data = []
    for row in zip(ipsi_counts, [f'ipsi-{count_removed}']*len(ipsi_counts)):
        path_counts_data.append(row)
    
    for row in zip(contra_counts, [f'contra-{count_removed}']*len(contra_counts)):
        path_counts_data.append(row)

    path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition'])
    path_counts_data.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges/processed/excised_graph_random-ipsi-contra_{count_removed}-removed.csv')

    # count per # hops
    ipsi_path_counts = [[len(x) for x in path] for path in ipsi_paths]
    contra_path_counts = [[len(x) for x in path] for path in contra_paths]

    path_counts_length_data = []
    for i, path_length in enumerate(ipsi_path_counts):
        for row in zip(path_length, [f'ipsi-{count_removed}']*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    for i, path_length in enumerate(contra_path_counts):
        for row in zip(path_length, [f'contra-{count_removed}']*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'N'])

    path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
    path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'path_length']).count()
    path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges/processed/excised_graph_random-ipsi-contra_{count_removed}-removed_path-lengths.csv')


cutoff=5
n_init = 8

count_removed = 500
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_paths, random_contra_paths, count_removed)

count_removed = 1000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_paths, random_contra_paths, count_removed)

count_removed = 2000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_paths, random_contra_paths, count_removed)

count_removed = 4000
random_ipsi_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_paths, random_contra_paths, count_removed)

# %%
##########
# EXPERIMENT 3: removing random number of ipsi vs contra edges, effect on paths on just one side of brain
# 

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

all_edges_combined_split = []
for i in range(len(all_edges_combined.index)):
    row = all_edges_combined.iloc[i]
    if((row.upstream_status=='paired') & (row.downstream_status=='paired')):
        if(row.type=='ipsilateral'):
            all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.type=='contralateral'):
            all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.left, row.type, row.upstream_status, row.downstream_status])
            all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])

    if((row.upstream_status=='nonpaired') & (row.downstream_status=='paired')):
        if(row.upstream_pair_id in left):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.upstream_pair_id in right):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])

    if((row.upstream_status=='paired') & (row.downstream_status=='nonpaired')):
        if(row.downstream_pair_id in left):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.downstream_pair_id in right):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'right', row.left, row.type, row.upstream_status, row.downstream_status])

    #if((row.upstream_status=='nonpaired') & (row.downstream_status=='nonpaired')):

all_edges_combined_split = pd.DataFrame(all_edges_combined_split, columns = ['upstream_pair_id', 'downstream_pair_id', 'upstream_side', 'downstream_side', 'edge_weight', 'type', 'upstream_status', 'downstream_status'])

# iterations for random edge removal as control
n_init = 8

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dVNC_left = list(np.intersect1d(dVNC, left))
dVNC_right = list(np.intersect1d(dVNC, right))

sensories_pair_ids = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs').name]
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]
all_sensories_left = list(np.intersect1d(all_sensories, left))
all_sensories_right = list(np.intersect1d(all_sensories, right))

# generate wildtype graph
split_graph = pg.Analyze_Nx_G(all_edges_combined_split, graph_type='directed', split_pairs=True)

# excise edges and generate graphs
random_ipsi500_left, random_ipsi500_right, random_contra500 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 500, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi1000_left, random_ipsi1000_right, random_contra1000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 1000, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi2000_left, random_ipsi2000_right, random_contra2000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 2000, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi4000_left, random_ipsi4000_right, random_contra4000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 4000, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi8000_left, random_ipsi8000_right, random_contra8000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 7943, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)

# %%
# generate and save paths
cutoff=5

# generate wildtype paths
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_wildtype'
pg.Prograph.generate_save_simple_paths(split_graph.G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=save_path)

# generate and save paths
count = 500
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi500_left[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi500_right[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra500[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 1000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi1000_left[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi1000_right[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra1000[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 2000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi2000_left[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi2000_right[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra2000[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 4000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi4000_left[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi4000_right[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra4000[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 8000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi8000_left[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi8000_right[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra8000[i].G, all_sensories_left, dVNC_left, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

# %%
# analyze paths: total and count per # hops
def process_paths_ipsi_contra(ipsi_left_paths, ipsi_right_paths, contra_paths, count_removed):
    ipsi_left_counts = [len(x) for x in ipsi_left_paths]
    ipsi_right_counts = [len(x) for x in ipsi_right_paths]
    contra_counts = [len(x) for x in contra_paths]

    path_counts_data = []
    for row in zip(ipsi_left_counts, [f'ipsi-left']*len(ipsi_left_counts), [count_removed]*len(ipsi_left_counts)):
        path_counts_data.append(row)

    for row in zip(ipsi_right_counts, [f'ipsi-right']*len(ipsi_right_counts), [count_removed]*len(ipsi_right_counts)):
        path_counts_data.append(row)
    
    for row in zip(contra_counts, [f'contra']*len(contra_counts), [count_removed]*len(contra_counts)):
        path_counts_data.append(row)

    path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition', 'edges_removed'])
    path_counts_data.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_{count_removed}-removed.csv')

    # count per # hops
    ipsi_left_path_counts = [[len(x) for x in path] for path in ipsi_left_paths]
    ipsi_right_path_counts = [[len(x) for x in path] for path in ipsi_right_paths]
    contra_path_counts = [[len(x) for x in path] for path in contra_paths]

    path_counts_length_data = []
    for i, path_length in enumerate(ipsi_left_path_counts):
        for row in zip(path_length, [f'ipsi-left']*len(path_length), [count_removed]*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    for i, path_length in enumerate(ipsi_right_path_counts):
        for row in zip(path_length, [f'ipsi-right']*len(path_length), [count_removed]*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    for i, path_length in enumerate(contra_path_counts):
        for row in zip(path_length, [f'contra']*len(path_length), [count_removed]*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'edges_removed', 'N'])

    path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
    path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'edges_removed', 'path_length']).count()
    path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_{count_removed}-removed_path-lengths.csv')

cutoff=5
n_init = 8

count_removed = 500
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 1000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 2000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 4000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 8000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

# wildtype paths
graph_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC_cutoff{cutoff}_wildtype.csv.gz')
wt_count = len(graph_paths)
path_counts_data = []
path_counts_data.append([wt_count, f'wildtype', 0])
path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition', 'edges_removed'])
path_counts_data.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype.csv')

path_counts_length_data = []
path_counts = [len(x) for x in graph_paths]
for row in zip(path_counts, [f'wildtype']*len(path_counts), [0]*len(path_counts), [0]*len(path_counts)):
    path_counts_length_data.append(row)

path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'edges_removed', 'N'])
path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'path_length']).count()
path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype_path_lengths.csv')

# %%
# plot total paths per condition from left -> left paths

total_paths = pd.concat([pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_500-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_1000-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_2000-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_4000-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_8000-removed.csv', index_col=0)], axis=0)

wildtype = pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype.csv', index_col=0)

total_paths = pd.concat([total_paths, pd.DataFrame([[wildtype['count'].values[0], 'contra', 0]], columns = total_paths.columns), 
                                        pd.DataFrame([[wildtype['count'].values[0], 'ipsi-left', 0]], columns = total_paths.columns), 
                                        pd.DataFrame([[wildtype['count'].values[0], 'ipsi-right', 0]], columns = total_paths.columns)], axis=0)

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(data = total_paths, x='edges_removed', y='count', hue='condition', err_style='bars', linewidth=0.75, err_kws={'elinewidth':0.75}, ax=ax)
ax.set(ylim=(0, 1100000))
plt.savefig('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/path-counts_left-to-left_removing-edge-types.pdf', format='pdf', bbox_inches='tight')

# plot total paths per path length from left -> left paths
total_paths = pd.concat([pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_500-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_1000-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_2000-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_4000-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph_random-ipsi-contra_8000-removed_path-lengths.csv')], axis=0)

wildtype = pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype_path_lengths.csv')

total_paths_normalized = []
for i in range(len(total_paths.index)):
    length = total_paths.iloc[i].path_length
    row = [total_paths.iloc[i].condition, total_paths.iloc[i].N, 
        total_paths.iloc[i].edges_removed, total_paths.iloc[i].path_length, 
        total_paths.iloc[i].value/wildtype[wildtype.path_length==length].value.values[0]] # normalized path counts by wildtype

    total_paths_normalized.append(row)

total_paths_normalized = pd.DataFrame(total_paths_normalized, columns = total_paths.columns)

for removed in [500, 1000, 2000, 4000, 8000]:
    fig, ax = plt.subplots(1,1, figsize=(2,2))
    sns.lineplot(data=total_paths_normalized[total_paths_normalized.edges_removed==removed], x='path_length', y='value', hue='condition', err_style='bars', linewidth=0.75, err_kws={'elinewidth':0.75}, ax=ax)
    ax.set(ylim=(0, 1.1))
    plt.savefig(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/path-length-counts_left-to-left_removing-{removed}-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
##########
# EXPERIMENT 4: removing random number of ipsi vs contra edges, effect on paths on just one side of brain to opposite side
# 

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

all_edges_combined_split = []
for i in range(len(all_edges_combined.index)):
    row = all_edges_combined.iloc[i]
    if((row.upstream_status=='paired') & (row.downstream_status=='paired')):
        if(row.type=='ipsilateral'):
            all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.type=='contralateral'):
            all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.left, row.type, row.upstream_status, row.downstream_status])
            all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])

    if((row.upstream_status=='nonpaired') & (row.downstream_status=='paired')):
        if(row.upstream_pair_id in left):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.upstream_pair_id in right):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, pm.Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])

    if((row.upstream_status=='paired') & (row.downstream_status=='nonpaired')):
        if(row.downstream_pair_id in left):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.right, row.type, row.upstream_status, row.downstream_status])
        if(row.downstream_pair_id in right):
            if(row.type=='ipsilateral'):
                all_edges_combined_split.append([pm.Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
            if(row.type=='contralateral'):
                all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'right', row.left, row.type, row.upstream_status, row.downstream_status])

    #if((row.upstream_status=='nonpaired') & (row.downstream_status=='nonpaired')):

all_edges_combined_split = pd.DataFrame(all_edges_combined_split, columns = ['upstream_pair_id', 'downstream_pair_id', 'upstream_side', 'downstream_side', 'edge_weight', 'type', 'upstream_status', 'downstream_status'])

# iterations for random edge removal as control
n_init = 8

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dVNC_left = list(np.intersect1d(dVNC, left))
dVNC_right = list(np.intersect1d(dVNC, right))

sensories_pair_ids = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs').name]
all_sensories = [x for sublist in sensories_pair_ids for x in sublist]
all_sensories_left = list(np.intersect1d(all_sensories, left))
all_sensories_right = list(np.intersect1d(all_sensories, right))

# generate wildtype graph
split_graph = pg.Analyze_Nx_G(all_edges_combined_split, graph_type='directed', split_pairs=True)

# excise edges and generate graphs
random_ipsi500_left, random_ipsi500_right, random_contra500 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 500, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi1000_left, random_ipsi1000_right, random_contra1000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 1000, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi2000_left, random_ipsi2000_right, random_contra2000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 2000, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi4000_left, random_ipsi4000_right, random_contra4000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 4000, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)
random_ipsi8000_left, random_ipsi8000_right, random_contra8000 = pg.Prograph.excise_ipsi_contra_edge_experiment(all_edges_combined_split, 7943, n_init, 0, left, right, exclude_nodes=(all_sensories+dVNC), split_pairs=True)

# %%
# generate and save paths
cutoff=5

# generate wildtype paths
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_wildtype'
pg.Prograph.generate_save_simple_paths(split_graph.G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=save_path)

# generate and save paths
count = 500
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi500_left[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi500_right[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra500[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 1000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi1000_left[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi1000_right[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra1000[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 2000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi2000_left[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi2000_right[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra2000[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 4000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi4000_left[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi4000_right[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra4000[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

count = 8000
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi8000_left[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_ipsi8000_right[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))
save_path = f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count}-N'
Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(random_contra8000[i].G, all_sensories_left, dVNC_right, cutoff=cutoff, save_path=f'{save_path}{i}') for i in tqdm((range(n_init))))

# %%
# analyze paths: total and count per # hops
def process_paths_ipsi_contra(ipsi_left_paths, ipsi_right_paths, contra_paths, count_removed):
    ipsi_left_counts = [len(x) for x in ipsi_left_paths]
    ipsi_right_counts = [len(x) for x in ipsi_right_paths]
    contra_counts = [len(x) for x in contra_paths]

    path_counts_data = []
    for row in zip(ipsi_left_counts, [f'ipsi-left']*len(ipsi_left_counts), [count_removed]*len(ipsi_left_counts)):
        path_counts_data.append(row)

    for row in zip(ipsi_right_counts, [f'ipsi-right']*len(ipsi_right_counts), [count_removed]*len(ipsi_right_counts)):
        path_counts_data.append(row)
    
    for row in zip(contra_counts, [f'contra']*len(contra_counts), [count_removed]*len(contra_counts)):
        path_counts_data.append(row)

    path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition', 'edges_removed'])
    path_counts_data.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_{count_removed}-removed.csv')

    # count per # hops
    ipsi_left_path_counts = [[len(x) for x in path] for path in ipsi_left_paths]
    ipsi_right_path_counts = [[len(x) for x in path] for path in ipsi_right_paths]
    contra_path_counts = [[len(x) for x in path] for path in contra_paths]

    path_counts_length_data = []
    for i, path_length in enumerate(ipsi_left_path_counts):
        for row in zip(path_length, [f'ipsi-left']*len(path_length), [count_removed]*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    for i, path_length in enumerate(ipsi_right_path_counts):
        for row in zip(path_length, [f'ipsi-right']*len(path_length), [count_removed]*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    for i, path_length in enumerate(contra_path_counts):
        for row in zip(path_length, [f'contra']*len(path_length), [count_removed]*len(path_length), [i]*len(path_length)):
            path_counts_length_data.append(row)

    path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'edges_removed', 'N'])

    path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
    path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'edges_removed', 'path_length']).count()
    path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_{count_removed}-removed_path-lengths.csv')

cutoff=5
n_init = 8

count_removed = 500
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 1000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 2000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 4000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

count_removed = 8000
random_ipsi_left_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-left-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_ipsi_right_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-ipsi-right-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
random_contra_paths = Parallel(n_jobs=-1)(delayed(pg.Prograph.open_simple_paths)(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_minus-edges-contra-{count_removed}-N{i}.csv.gz') for i in tqdm(range(n_init)))
process_paths_ipsi_contra(random_ipsi_left_paths, random_ipsi_right_paths, random_contra_paths, count_removed)

# wildtype paths
graph_paths = pg.Prograph.open_simple_paths(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/all-paths-sens-to-dVNC-right_cutoff{cutoff}_wildtype.csv.gz')
wt_count = len(graph_paths)
path_counts_data = []
path_counts_data.append([wt_count, f'wildtype', 0])
path_counts_data = pd.DataFrame(path_counts_data, columns=['count', 'condition', 'edges_removed'])
path_counts_data.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype_to-dVNC-right.csv')

path_counts_length_data = []
path_counts = [len(x) for x in graph_paths]
for row in zip(path_counts, [f'wildtype']*len(path_counts), [0]*len(path_counts), [0]*len(path_counts)):
    path_counts_length_data.append(row)

path_counts_length_data = pd.DataFrame(path_counts_length_data, columns=['path_length', 'condition', 'edges_removed', 'N'])
path_counts_length_data['value'] = [1]*len(path_counts_length_data) # just adding [1] so that groupby has something to count
path_counts_length_data_counts = path_counts_length_data.groupby(['condition', 'N', 'path_length']).count()
path_counts_length_data_counts.to_csv(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype_path_lengths_to-dVNC-right.csv')

# %%
# plot total paths per condition from left -> right paths

total_paths = pd.concat([pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_500-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_1000-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_2000-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_4000-removed.csv', index_col=0),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_8000-removed.csv', index_col=0)], axis=0)

wildtype = pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype_to-dVNC-right.csv', index_col=0)

total_paths = pd.concat([total_paths, pd.DataFrame([[wildtype['count'].values[0], 'contra', 0]], columns = total_paths.columns), 
                                        pd.DataFrame([[wildtype['count'].values[0], 'ipsi-left', 0]], columns = total_paths.columns), 
                                        pd.DataFrame([[wildtype['count'].values[0], 'ipsi-right', 0]], columns = total_paths.columns)], axis=0)

fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
sns.lineplot(data = total_paths, x='edges_removed', y='count', hue='condition', err_style='bars', linewidth=0.5, err_kws={'elinewidth':0.5}, ax=ax)
ax.set(ylim=(0, 1100000))
plt.savefig('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/path-counts_left-to-right_removing-edge-types.pdf', format='pdf', bbox_inches='tight')

# plot total paths per path length from left -> left paths
total_paths = pd.concat([pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_500-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_1000-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_2000-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_4000-removed_path-lengths.csv'),
                            pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/excised_graph-to-dVNC-right_random-ipsi-contra_8000-removed_path-lengths.csv')], axis=0)

wildtype = pd.read_csv('interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/wildtype_path_lengths_to-dVNC-right.csv')

total_paths_normalized = []
for i in range(len(total_paths.index)):
    length = total_paths.iloc[i].path_length
    row = [total_paths.iloc[i].condition, total_paths.iloc[i].N, 
        total_paths.iloc[i].edges_removed, total_paths.iloc[i].path_length, 
        total_paths.iloc[i].value/wildtype[wildtype.path_length==length].value.values[0]] # normalized path counts by wildtype

    total_paths_normalized.append(row)

total_paths_normalized = pd.DataFrame(total_paths_normalized, columns = total_paths.columns)

for removed in [500, 1000, 2000, 4000, 8000]:
    fig, ax = plt.subplots(1,1, figsize=(2,2))
    sns.lineplot(data=total_paths_normalized[total_paths_normalized.edges_removed==removed], x='path_length', y='value', hue='condition', err_style='bars', linewidth=0.75, err_kws={'elinewidth':0.75}, ax=ax)
    ax.set(ylim=(0, 1.1))
    plt.savefig(f'interhemisphere/csv/paths/random-ipsi-contra-edges_left-paths/processed/path-length-counts_left-to-right_removing-{removed}-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
