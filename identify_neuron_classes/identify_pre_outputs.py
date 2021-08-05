#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import connectome_tools.process_skeletons as skel

edge_ad = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)
pairs = pm.Promat.get_pairs()

# %%
#

dVNCs = pymaid.get_skids_by_annotation('mw dVNC')
dVNCs = [x if x!=21790197 else 15672263 for x in dVNCs] # a single descending neuron was incorrectly merged and split, so skid is different...
dSEZs = pymaid.get_skids_by_annotation('mw dSEZ')
RGNs = pymaid.get_skids_by_annotation('mw RGN')

edge_ad.set_index('downstream_skid', inplace=True)
pre_dVNCs = np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, dVNCs), :].upstream_skid)
pre_dSEZs = np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, dSEZs), :].upstream_skid)
pre_RGNs = np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, RGNs), :].upstream_skid)

#pymaid.add_annotations(pre_dVNCs, 'mw pre-dVNC')
#pymaid.add_annotations(pre_dSEZs, 'mw pre-dSEZ')
#pymaid.add_annotations(pre_RGNs, 'mw pre-RGN')

# %%
# identify dVNCs-to-A1 and dVNCs-not-to-A1

A1s = pymaid.get_skids_by_annotation('mw A1 neurons paired')
dVNCs_to_A1 = edge_ad.loc[np.intersect1d(A1s, edge_ad.index), 'upstream_skid']
dVNCs_to_A1 = list(np.intersect1d(np.unique(dVNCs_to_A1), dVNCs))
dVNCs_not_to_A1 = list(np.setdiff1d(dVNCs, dVNCs_to_A1))

dVNCs_to_A1 = [x if x!=15672263 else 21790197 for x in dVNCs_to_A1]
pymaid.add_annotations(dVNCs_to_A1, 'mw dVNC to A1')
pymaid.add_annotations(dVNCs_not_to_A1, 'mw dVNC not to A1')

# %%