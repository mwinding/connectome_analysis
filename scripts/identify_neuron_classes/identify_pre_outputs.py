#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date_A1_brain, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat


edges_ad = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date_A1_brain, pairs_combined=False)
pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
#

dVNCs = pymaid.get_skids_by_annotation('mw dVNC')
dSEZs = pymaid.get_skids_by_annotation('mw dSEZ')
RGNs = pymaid.get_skids_by_annotation('mw RGN')

edges_ad.set_index('downstream_skid', inplace=True)
pre_dVNCs = np.unique(edges_ad.loc[np.intersect1d(edges_ad.index, dVNCs), :].upstream_skid)
pre_dSEZs = np.unique(edges_ad.loc[np.intersect1d(edges_ad.index, dSEZs), :].upstream_skid)
pre_RGNs = np.unique(edges_ad.loc[np.intersect1d(edges_ad.index, RGNs), :].upstream_skid)

#pymaid.add_annotations(pre_dVNCs, 'mw pre-dVNC')
#pymaid.add_annotations(pre_dSEZs, 'mw pre-dSEZ')
#pymaid.add_annotations(pre_RGNs, 'mw pre-RGN')

# %%
# identify dVNCs-to-A1 and dVNCs-not-to-A1

A1s = pymaid.get_skids_by_annotation('mw A1 neurons paired')
dVNCs_to_A1 = edges_ad.loc[np.intersect1d(A1s, edges_ad.index), 'upstream_skid']
dVNCs_to_A1 = list(np.intersect1d(np.unique(dVNCs_to_A1), dVNCs))
dVNCs_not_to_A1 = list(np.setdiff1d(dVNCs, dVNCs_to_A1))

pymaid.add_annotations(dVNCs_to_A1, 'mw dVNC to A1')
pymaid.add_annotations(dVNCs_not_to_A1, 'mw dVNC not to A1')

# %%