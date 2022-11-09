#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from contools import Celltype, Celltype_Analyzer, Promat, Prograph

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=True)
edge_ad = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=False)
pairs = Promat.get_pairs(pairs_path=pairs_path)
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
LHN_pure = np.setdiff1d(np.unique(ds_uPNs), exclude)
LHN = np.setdiff1d(np.unique(np.concatenate([ds_uPNs, ds_vPNs, ds_tPNs])), exclude)

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
