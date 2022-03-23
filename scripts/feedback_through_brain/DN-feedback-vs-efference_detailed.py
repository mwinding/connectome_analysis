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

from contools import Celltype, Celltype_Analyzer, Promat, Cascade_Analyzer
import pickle

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')

dVNC_FB_casc = pymaid.get_skids_by_annotation('mw dVNC-feedback-casc 2022-03-15')
dVNC_EC_casc = pymaid.get_skids_by_annotation('mw dVNC-efference-casc 2022-03-15')
dSEZ_FB_casc = pymaid.get_skids_by_annotation('mw dSEZ-feedback-casc 2022-03-15')
dSEZ_EC_casc = pymaid.get_skids_by_annotation('mw dSEZ-efference-casc 2022-03-15')

dVNC_FB_1hop = pymaid.get_skids_by_annotation('mw dVNC-feedback-1hop 2022-03-15')
dVNC_EC_1hop = pymaid.get_skids_by_annotation('mw dVNC-efference-1hop 2022-03-15')
dSEZ_FB_1hop = pymaid.get_skids_by_annotation('mw dSEZ-feedback-1hop 2022-03-15')
dSEZ_EC_1hop = pymaid.get_skids_by_annotation('mw dSEZ-efference-1hop 2022-03-15')

dVNC_FB_2hop = pymaid.get_skids_by_annotation('mw dVNC-feedback-2hop 2022-03-15')
dVNC_EC_2hop = pymaid.get_skids_by_annotation('mw dVNC-efference-2hop 2022-03-15')
dSEZ_FB_2hop = pymaid.get_skids_by_annotation('mw dSEZ-feedback-2hop 2022-03-15')
dSEZ_EC_2hop = pymaid.get_skids_by_annotation('mw dSEZ-efference-2hop 2022-03-15')

# %%
# what fraction of each cell type is in FB or EC pathway downstream of dVNC

def fraction_in(skids_FB, skids_EC, us_partner, distance):
        FBNs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MB-FBNs')
        DANs = pymaid.get_skids_by_annotation('mw MBIN subclass_DAN')
        MBINs = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw MBIN'), pymaid.get_skids_by_annotation('mw MBIN subclass_OAN')))

        count_FBN_FB = len(np.intersect1d(FBNs, skids_FB))
        count_DAN_FB = len(np.intersect1d(DANs, skids_FB))
        count_MBIN_FB = len(np.intersect1d(MBINs, skids_FB))

        count_FBN_EC = len(np.intersect1d(FBNs, skids_EC))
        count_DAN_EC = len(np.intersect1d(DANs, skids_EC))
        count_MBIN_EC = len(np.intersect1d(MBINs, skids_EC))

        total_FBN = len(FBNs)
        total_DAN = len(DANs)
        total_MBIN = len(MBINs)

        fraction_FBN_FB = len(np.intersect1d(FBNs, skids_FB))/len(FBNs)
        fraction_DAN_FB = len(np.intersect1d(DANs, skids_FB))/len(DANs)
        fraction_MBIN_FB = len(np.intersect1d(MBINs, skids_FB))/len(MBINs)

        fraction_FBN_EC = len(np.intersect1d(FBNs, skids_EC))/len(FBNs)
        fraction_DAN_EC = len(np.intersect1d(DANs, skids_EC))/len(DANs)
        fraction_MBIN_EC = len(np.intersect1d(MBINs, skids_EC))/len(MBINs)

        df = [['MBIN', count_MBIN_FB, total_MBIN, fraction_MBIN_FB, us_partner, 'feedback', distance],
                ['MBIN', count_MBIN_EC, total_MBIN, fraction_MBIN_EC, us_partner, 'efference_copy', distance],
                ['DAN', count_DAN_FB, total_DAN, fraction_DAN_FB, us_partner, 'feedback', distance],
                ['DAN', count_DAN_EC, total_DAN, fraction_DAN_EC, us_partner, 'efference_copy', distance],
                ['MB_FBN', count_FBN_FB, total_FBN, fraction_FBN_FB, us_partner, 'feedback', distance],
                ['MB_FBN', count_FBN_EC, total_FBN, fraction_FBN_EC, us_partner, 'efference_copy', distance]]

        df = pd.DataFrame(df, columns = ['celltype', 'count', 'total', 'fraction', 'upstream_partner', 'connection_type', 'distance'])
        return(df)

dfs = [
        fraction_in(dVNC_FB_casc, dVNC_EC_casc, 'dVNC', 'cascade_8hop'),
        fraction_in(dVNC_FB_1hop, dVNC_EC_1hop, 'dVNC', '1-hop'),
        fraction_in(dVNC_FB_2hop, dVNC_EC_2hop, 'dVNC', '2-hop'),
        fraction_in(dSEZ_FB_casc, dSEZ_EC_casc, 'dSEZ', 'cascade_8hop'),
        fraction_in(dSEZ_FB_1hop, dSEZ_EC_1hop, 'dSEZ', '1-hop'),
        fraction_in(dSEZ_FB_2hop, dSEZ_EC_2hop, 'dSEZ', '2-hop')
        ]

dfs = pd.concat(dfs, axis=0)

# %%
# check celltypes downstream of dSEZs (1-hop, 2-hop, cascade)

_, celltypes = Celltype_Analyzer.default_celltypes()

cts = [Celltype('dVNC_FB_1hop', dVNC_FB_1hop), Celltype('dVNC_EC_1hop', dVNC_EC_1hop),
        Celltype('dVNC_FB_2hop', dVNC_FB_2hop), Celltype('dVNC_EC_2hop', dVNC_EC_2hop),
        Celltype('dVNC_FB_casc', dVNC_FB_casc), Celltype('dVNC_EC_casc', dVNC_EC_casc),
        Celltype('dSEZ_FB_1hop', dSEZ_FB_1hop), Celltype('dSEZ_EC_1hop', dSEZ_EC_1hop),
        Celltype('dSEZ_FB_2hop', dSEZ_FB_2hop), Celltype('dSEZ_EC_2hop', dSEZ_EC_2hop),
        Celltype('dSEZ_FB_casc', dSEZ_FB_casc), Celltype('dSEZ_EC_casc', dSEZ_EC_casc)]

cts = [Celltype('dVNC_1hop', list(np.unique(dVNC_FB_1hop + dVNC_EC_1hop))),
        Celltype('dVNC_2hop', list(np.unique(dVNC_FB_2hop + dVNC_EC_2hop))), 
        Celltype('dVNC_casc', list(np.unique(dVNC_FB_casc + dVNC_EC_casc))),
        Celltype('dSEZ_1hop',list(np.unique(dSEZ_FB_1hop + dSEZ_EC_1hop))),
        Celltype('dSEZ_2hop', list(np.unique(dSEZ_FB_2hop + dSEZ_EC_2hop))),
        Celltype('dSEZ_casc', list(np.unique(dSEZ_FB_casc + dSEZ_EC_casc)))]

cts = Celltype_Analyzer(cts)
cts.set_known_types(celltypes)

cts.memberships()
cts.memberships(raw_num=True)
cts.memberships(by_celltype=False)

# classic PNs
uPN = pymaid.get_skids_by_annotation('mw uPN')
mPN = pymaid.get_skids_by_annotation('mw mPN')
tPN = pymaid.get_skids_by_annotation('mw tPN')
vPN = pymaid.get_skids_by_annotation('mw vPN')

PN_celltypes = [Celltype('uPN', uPN), Celltype('mPN', mPN), Celltype('vPN', vPN), Celltype('tPN', tPN)]
cts.set_known_types(PN_celltypes)
cts.memberships(raw_num=True)

# new PNs
PN_types = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 2nd_order PN', split=True, return_celltypes=True)
cts.set_known_types(PN_types)
cts.memberships(raw_num=True)
# %%
