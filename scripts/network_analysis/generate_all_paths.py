# %%
#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm


rm = pymaid.CatmaidInstance(url, token, name, password)

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

graph = pg.Analyze_Nx_G(all_edges_combined, graph_type='directed')
pairs = pm.Promat.get_pairs()

# %%
# load neuron types

all_sensories = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain sensories')
all_outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

sensory_pair_ids = pm.Promat.load_pairs_from_annotation('sensory', pairs, return_type='all_pair_ids', skids=all_sensories, use_skids=True)
outputs_pair_ids = pm.Promat.load_pairs_from_annotation('output', pairs, return_type='all_pair_ids', skids=all_outputs, use_skids=True)
dVNC_pair_ids = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')


# %%
# generate and save paths all sensory to outputs

cutoff = 6
outputs = [dVNC_pair_ids, dSEZ_pair_ids, RGN_pair_ids, outputs_pair_ids]
output_types = ['dVNC', 'dSEZ', 'RGN', 'output']
save_paths = [f'data/paths/all_paths_sens-to-{output_type}_cutoff{cutoff}' for output_type in output_types]

Parallel(n_jobs=-1)(delayed(pg.Prograph.generate_save_simple_paths)(G=graph.G, source_list=sensory_pair_ids, targets=outputs[i], cutoff=cutoff, save_path = save_paths[i]) for i in tqdm(range(len(save_paths))))

'''
save_path = f'data/paths/all_paths_sens-to-dVNC_cutoff{cutoff}'
pg.Prograph.generate_save_simple_paths(graph.G, sensory_pair_ids, dVNC_pair_ids, cutoff=cutoff, save_path=save_path)

save_path = f'data/paths/all_paths_sens-to-dSEZ_cutoff{cutoff}'
pg.Prograph.generate_save_simple_paths(graph.G, sensory_pair_ids, dSEZ_pair_ids, cutoff=cutoff, save_path=save_path)

save_path = f'data/paths/all_paths_sens-to-RGN_cutoff{cutoff}'
pg.Prograph.generate_save_simple_paths(graph.G, sensory_pair_ids, RGN_pair_ids, cutoff=cutoff, save_path=save_path)

save_path = f'data/paths/all_paths_sens-to-output_cutoff{cutoff}'
pg.Prograph.generate_save_simple_paths(graph.G, sensory_pair_ids, outputs_pair_ids, cutoff=cutoff, save_path=save_path)
'''

# %%
# 

dVNC_paths = pg.Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-dVNC_cutoff{cutoff}.csv.gz')
dSEZ_paths = pg.Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-dSEZ_cutoff{cutoff}.csv.gz')
RGN_paths = pg.Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-RGN_cutoff{cutoff}.csv.gz')
output_paths = pg.Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-output_cutoff{cutoff}.csv.gz')

# %%
