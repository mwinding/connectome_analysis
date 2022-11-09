# %%
#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

from contools import Analyze_Nx_G, Prograph, Promat, Celltype_Analyzer

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

# load previously generated paths
all_edges_combined = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=True)

graph = Analyze_Nx_G(all_edges_combined, graph_type='directed')
pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
# load neuron types

all_sensories = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain sensories')
all_outputs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

sensory_pair_ids = Promat.load_pairs_from_annotation('sensory', pairs, return_type='all_pair_ids', skids=all_sensories, use_skids=True)
outputs_pair_ids = Promat.load_pairs_from_annotation('output', pairs, return_type='all_pair_ids', skids=all_outputs, use_skids=True)
dVNC_pair_ids = Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_ids')
dSEZ_pair_ids = Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids')
RGN_pair_ids = Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids')


# %%
# generate and save paths all sensory to outputs

cutoff = 6
outputs = [dVNC_pair_ids, dSEZ_pair_ids, RGN_pair_ids, outputs_pair_ids]
output_types = ['dVNC', 'dSEZ', 'RGN', 'output']
save_paths = [f'data/paths/all_paths_sens-to-{output_type}_cutoff{cutoff}' for output_type in output_types]

Parallel(n_jobs=-1)(delayed(Prograph.generate_save_simple_paths)(G=graph.G, source_list=sensory_pair_ids, targets=outputs[i], cutoff=cutoff, save_path = save_paths[i]) for i in tqdm(range(len(save_paths))))

# %%
# 

dVNC_paths = Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-dVNC_cutoff{cutoff}.csv.gz')
dSEZ_paths = Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-dSEZ_cutoff{cutoff}.csv.gz')
RGN_paths = Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-RGN_cutoff{cutoff}.csv.gz')
output_paths = Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-output_cutoff{cutoff}.csv.gz')

# %%
