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

pymaid.add_annotations(pre_dVNCs, 'mw pre-dVNC')
pymaid.add_annotations(pre_dSEZs, 'mw pre-dSEZ')
pymaid.add_annotations(pre_RGNs, 'mw pre-RGN')

# %%
# 