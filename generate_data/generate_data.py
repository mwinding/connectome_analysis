# %%

import pymaid
import contools
from contools import generate_adjs
import numpy as np

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
data_date = '2022-03-15'
rm = pymaid.CatmaidInstance(url, token, name, password)

# %%

# select neurons to include in adjacency matrices
all_neurons = pymaid.get_skids_by_annotation(['mw brain paper clustered neurons', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation('mw brain very incomplete')
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are so incomplete, they have no split point

# specify split tags
split_tag = 'mw axon split'
special_split_tags = ['mw axon start', 'mw axon end']
not_split_skids = pymaid.get_skids_by_annotation(['mw unsplittable'])

generate_adjs.adj_split_axons_dendrites(all_neurons, split_tag, special_split_tags, not_split_skids)

# %%

# generate edge list with average pairwise threshold = 3
threshold = 3
pairs = contools.Promat.get_pairs(pairs_path=pairs_path)
generate_adjs.edge_thresholds(path='data/adj', threshold=threshold, left_annot='mw left', right_annot='mw right', pairs = pairs, fraction_input=False, date=data_date)

# %%

# generate edge list with %input threshold = 0.01
threshold = 0.01
pairs = contools.Promat.get_pairs(pairs_path=pairs_path, remove_notes=False)
generate_adjs.edge_thresholds(path='data/adj', threshold=threshold, left_annot='mw left', right_annot='mw right', pairs = pairs, fraction_input=True, date=data_date)

# %%