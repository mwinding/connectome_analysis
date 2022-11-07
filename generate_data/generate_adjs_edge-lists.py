# %%

import pymaid
import contools
from contools import generate_adjs
import numpy as np
from datetime import datetime

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
data_date = datetime.today().strftime('%Y-%m-%d')
rm = pymaid.CatmaidInstance(url, token, name, password)

# %%
# select neurons to include in adjacency matrices
all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# specify split tags
split_tag = 'mw axon split'
special_split_tags = ['mw axon start', 'mw axon end']
not_split_skids = pymaid.get_skids_by_annotation(['mw unsplittable'])

generate_adjs.adj_split_axons_dendrites(all_neurons, split_tag, special_split_tags, not_split_skids)

# %%
# generate edge list with average pairwise threshold = 3
threshold = 2
pairs = contools.Promat.get_pairs(pairs_path=pairs_path)
generate_adjs.edge_thresholds(path='data/adj', threshold=threshold, left_annot='mw left', right_annot='mw right', pairs = pairs, fraction_input=False, date=data_date)

# %%
# generate edge list with %input threshold = 0.01
threshold = 0.01
pairs = contools.Promat.get_pairs(pairs_path=pairs_path, remove_notes=False)
generate_adjs.edge_thresholds(path='data/adj', threshold=threshold, left_annot='mw left', right_annot='mw right', pairs = pairs, fraction_input=True, date=data_date)

# %%
# same with A1 neurons and brain
# select neurons to include in adjacency matrices

data_date = datetime.today().strftime('%Y-%m-%d') + 'b'

all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons', 'mw A1 neurons paired'])
all_neurons = all_neurons + contools.Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories') # add a few nonpaired sensories to the list
all_neurons = list(np.unique(all_neurons)) # remove duplicates between 'mw A1 neurons paired' and 'mw A1 sensories'
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are so incomplete, they have no split point

# specify split tags
split_tag = 'mw axon split'
special_split_tags = ['mw axon start', 'mw axon end']
not_split_skids = pymaid.get_skids_by_annotation(['mw unsplittable'])

generate_adjs.adj_split_axons_dendrites(all_neurons, split_tag, special_split_tags, not_split_skids, data_date=data_date)

# %%
# generate edge list with %input threshold = 0.01 for brain + A1 dataset
data_date = datetime.today().strftime('%Y-%m-%d') + 'b'

threshold = 0.01
pairs = contools.Promat.get_pairs(pairs_path=pairs_path, remove_notes=False)
generate_adjs.edge_thresholds(path='data/adj', threshold=threshold, left_annot='mw left', right_annot='mw right', pairs = pairs, fraction_input=True, date=data_date)

# %%
