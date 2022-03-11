#%%

from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

import pickle

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')

dVNC_FB_casc = pymaid.get_skids_by_annotation('mw dVNC-feedback-casc 2022-03-10')
dVNC_EC_casc = pymaid.get_skids_by_annotation('mw dVNC-efference-casc 2022-03-10')
dSEZ_FB_casc = pymaid.get_skids_by_annotation('mw dSEZ-feedback-casc 2022-03-10')
dSEZ_EC_casc = pymaid.get_skids_by_annotation('mw dSEZ-efference-casc 2022-03-10')

dVNC_FB_1hop = pymaid.get_skids_by_annotation('mw dVNC-feedback-1hop 2022-03-10')
dVNC_EC_1hop = pymaid.get_skids_by_annotation('mw dVNC-efference-1hop 2022-03-10')
dSEZ_FB_1hop = pymaid.get_skids_by_annotation('mw dSEZ-feedback-1hop 2022-03-10')
dSEZ_EC_1hop = pymaid.get_skids_by_annotation('mw dSEZ-efference-1hop 2022-03-10')

# %%
# what fraction of each cell type is in FB or EC pathway downstream of dVNC

from contools import Celltype_Analyzer

FBNs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MB-FBNs')
DANs = pymaid.get_skids_by_annotation('mw MBIN subclass_DAN')
MBINs = pymaid.get_skids_by_annotation('mw MBIN')

fraction_FBN_dVNC_FB = len(np.intersect1d(FBNs, dVNC_FB_casc))/len(FBNs)
fraction_DANs_dVNC_FB = len(np.intersect1d(DANs, dVNC_FB_casc))/len(DANs)
fraction_MBINs_dVNC_FB = len(np.intersect1d(MBINs, dVNC_FB_casc))/len(MBINs)

fraction_FBN_dVNC_EC = len(np.intersect1d(FBNs, dVNC_EC_casc))/len(FBNs)
fraction_DANs_dVNC_EC = len(np.intersect1d(DANs, dVNC_EC_casc))/len(DANs)
fraction_MBINs_dVNC_EC = len(np.intersect1d(MBINs, dVNC_EC_casc))/len(MBINs)

fraction_FBN_dSEZ_FB = len(np.intersect1d(FBNs, dSEZ_FB_casc))/len(FBNs)
fraction_DANs_dSEZ_FB = len(np.intersect1d(DANs, dSEZ_FB_casc))/len(DANs)
fraction_MBINs_dSEZ_FB = len(np.intersect1d(MBINs, dSEZ_FB_casc))/len(MBINs)

fraction_FBN_dSEZ_EC = len(np.intersect1d(FBNs, dSEZ_EC_casc))/len(FBNs)
fraction_DANs_dSEZ_EC = len(np.intersect1d(DANs, dSEZ_EC_casc))/len(DANs)
fraction_MBINs_dSEZ_EC = len(np.intersect1d(MBINs, dSEZ_EC_casc))/len(MBINs)

df = [['MBIN', fraction_MBINs_dVNC_FB, 'dVNC', 'feedback', ],
        ['MBIN', fraction_MBINs_dVNC_EC, 'dVNC', 'efference_copy'],
        ['DAN', fraction_DANs_dVNC_FB, 'dVNC', 'feedback'],
        ['DAN', fraction_DANs_dVNC_EC, 'dVNC', 'efference_copy'],
        ['MB_FBN', fraction_FBN_dVNC_FB, 'dVNC', 'feedback'],
        ['MB_FBN', fraction_FBN_dVNC_EC, 'dVNC', 'efference_copy'],
        ['MBIN', fraction_MBINs_dSEZ_FB, 'dSEZ', 'feedback', ],
        ['MBIN', fraction_MBINs_dSEZ_EC, 'dSEZ', 'efference_copy'],
        ['DAN', fraction_DANs_dSEZ_FB, 'dSEZ', 'feedback'],
        ['DAN', fraction_DANs_dSEZ_EC, 'dSEZ', 'efference_copy'],
        ['MB_FBN', fraction_FBN_dSEZ_FB, 'dSEZ', 'feedback'],
        ['MB_FBN', fraction_FBN_dSEZ_EC, 'dSEZ', 'efference_copy']]

df = pd.DataFrame(df, columns = ['celltype', 'fraction', 'upstream_partner', 'connection_type'])
df
# %%
