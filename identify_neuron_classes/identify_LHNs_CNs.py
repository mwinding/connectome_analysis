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
edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
edge_ad = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)
pairs = pm.Promat.get_pairs()
edge_ad.set_index('upstream_skid', inplace=True)

# %%
# 

uPN = pymaid.get_skids_by_annotation('mw uPN')
tPN = pymaid.get_skids_by_annotation('mw tPN')
vPN = pymaid.get_skids_by_annotation('mw vPN')
KC = pymaid.get_skids_by_annotation('mw KC')
mPN = pymaid.get_skids_by_annotation('mw mPN')
LON = pymaid.get_skids_by_annotation('mw LON')

MBON = pymaid.get_skids_by_annotation('mw MBON')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')

exclude = uPN + tPN + vPN + KC + mPN + LON + MBON + MBIN
# %%
# identify LHNs

ds_uPNs = np.setdiff1d(np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, uPN), :].downstream_skid), exclude)
ds_vPNs = np.setdiff1d(np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, vPN), :].downstream_skid), exclude)
ds_tPNs = np.setdiff1d(np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, tPN), :].downstream_skid), exclude)
LHN_pure = np.setdiff1d(np.unique(list(ds_uPNs)), exclude)
LHN = np.setdiff1d(np.unique(list(ds_uPNs) + list(ds_vPNs) + list(ds_tPNs)), exclude)

pymaid.add_annotations(LHN_pure, 'mw LHN-from-uPN')
#pymaid.add_annotations(LHN, 'mw LHN-from-uPN-vPN-tPN')
pymaid.add_annotations(LHN, 'mw LHN')
pymaid.add_annotations(ds_uPNs, 'mw ds-uPNs')
pymaid.add_annotations(ds_vPNs, 'mw ds-vPNs')
pymaid.add_annotations(ds_tPNs, 'mw ds-tPNs')

annot = 'mw CN'
pymaid.remove_annotations(pymaid.get_skids_by_annotation(annot), annot)
# %%
# identify CNs

LHN = pymaid.get_skids_by_annotation('mw LHN')

ds_MBONs = np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, MBON), :].downstream_skid)
ds_LHNs = np.unique(edge_ad.loc[np.intersect1d(edge_ad.index, LHN), :].downstream_skid)

CN1 = list(np.intersect1d(ds_MBONs, ds_LHNs))
CN2 = list(np.intersect1d(ds_MBONs, LHN))

exclude = uPN + tPN + vPN + KC + mPN + MBIN + LON
CN = np.setdiff1d(np.unique(CN1 + CN2), exclude)

pymaid.add_annotations(CN, 'mw CN')

# %%
