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

dVNC_FB_3hop = pymaid.get_skids_by_annotation('mw dVNC-feedback-3hop 2022-03-15')
dVNC_EC_3hop = pymaid.get_skids_by_annotation('mw dVNC-efference-3hop 2022-03-15')
dSEZ_FB_3hop = pymaid.get_skids_by_annotation('mw dSEZ-feedback-3hop 2022-03-15')
dSEZ_EC_3hop = pymaid.get_skids_by_annotation('mw dSEZ-efference-3hop 2022-03-15')

dVNC_FB_4hop = pymaid.get_skids_by_annotation('mw dVNC-feedback-4hop 2022-03-15')
dVNC_EC_4hop = pymaid.get_skids_by_annotation('mw dVNC-efference-4hop 2022-03-15')
dSEZ_FB_4hop = pymaid.get_skids_by_annotation('mw dSEZ-feedback-4hop 2022-03-15')
dSEZ_EC_4hop = pymaid.get_skids_by_annotation('mw dSEZ-efference-4hop 2022-03-15')

dVNC_FB_5hop = pymaid.get_skids_by_annotation('mw dVNC-feedback-5hop 2022-03-15')
dVNC_EC_5hop = pymaid.get_skids_by_annotation('mw dVNC-efference-5hop 2022-03-15')
dSEZ_FB_5hop = pymaid.get_skids_by_annotation('mw dSEZ-feedback-5hop 2022-03-15')
dSEZ_EC_5hop = pymaid.get_skids_by_annotation('mw dSEZ-efference-5hop 2022-03-15')

# %%
# what fraction of each cell type is in FB or EC pathway downstream of dVNC

def fraction_in(skids_FB, skids_EC, us_partner, distance):
        FBNs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MB-FBNs')
        DANs = pymaid.get_skids_by_annotation('mw MBIN subclass_DAN')
        MBINs = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw MBIN'), pymaid.get_skids_by_annotation('mw MBIN subclass_OAN') + pymaid.get_skids_by_annotation('mw MBIN subclass_DAN')))
        OANs = pymaid.get_skids_by_annotation('mw MBIN subclass_OAN')

        count_FBN_FB = len(np.intersect1d(FBNs, skids_FB))
        count_DAN_FB = len(np.intersect1d(DANs, skids_FB))
        count_MBIN_FB = len(np.intersect1d(MBINs, skids_FB))
        count_OAN_FB = len(np.intersect1d(OANs, skids_FB))

        count_FBN_EC = len(np.intersect1d(FBNs, skids_EC))
        count_DAN_EC = len(np.intersect1d(DANs, skids_EC))
        count_MBIN_EC = len(np.intersect1d(MBINs, skids_EC))
        count_OAN_EC = len(np.intersect1d(OANs, skids_EC))

        total_FBN = len(FBNs)
        total_DAN = len(DANs)
        total_MBIN = len(MBINs)
        total_OAN = len(OANs)

        fraction_FBN_FB = len(np.intersect1d(FBNs, skids_FB))/len(FBNs)
        fraction_DAN_FB = len(np.intersect1d(DANs, skids_FB))/len(DANs)
        fraction_MBIN_FB = len(np.intersect1d(MBINs, skids_FB))/len(MBINs)
        fraction_OAN_FB = len(np.intersect1d(OANs, skids_FB))/len(OANs)

        fraction_FBN_EC = len(np.intersect1d(FBNs, skids_EC))/len(FBNs)
        fraction_DAN_EC = len(np.intersect1d(DANs, skids_EC))/len(DANs)
        fraction_MBIN_EC = len(np.intersect1d(MBINs, skids_EC))/len(MBINs)
        fraction_OAN_EC = len(np.intersect1d(OANs, skids_EC))/len(OANs)

        df = [['MBIN', count_MBIN_FB, total_MBIN, fraction_MBIN_FB, us_partner, 'feedback', distance],
                ['MBIN', count_MBIN_EC, total_MBIN, fraction_MBIN_EC, us_partner, 'efference_copy', distance],
                ['DAN', count_DAN_FB, total_DAN, fraction_DAN_FB, us_partner, 'feedback', distance],
                ['DAN', count_DAN_EC, total_DAN, fraction_DAN_EC, us_partner, 'efference_copy', distance],
                ['OAN', count_DAN_FB, total_DAN, fraction_OAN_FB, us_partner, 'feedback', distance],
                ['OAN', count_DAN_EC, total_DAN, fraction_OAN_EC, us_partner, 'efference_copy', distance],
                ['MB_FBN', count_FBN_FB, total_FBN, fraction_FBN_FB, us_partner, 'feedback', distance],
                ['MB_FBN', count_FBN_EC, total_FBN, fraction_FBN_EC, us_partner, 'efference_copy', distance]]

        df = pd.DataFrame(df, columns = ['celltype', 'count', 'total', 'fraction', 'upstream_partner', 'connection_type', 'distance'])
        return(df)

dfs = [
        fraction_in(dVNC_FB_casc, dVNC_EC_casc, 'dVNC', 'cascade_8hop'),
        fraction_in(dVNC_FB_1hop, dVNC_EC_1hop, 'dVNC', '1-hop'),
        fraction_in(dVNC_FB_2hop, dVNC_EC_2hop, 'dVNC', '2-hop'),
        fraction_in(dVNC_FB_3hop, dVNC_EC_3hop, 'dVNC', '3-hop'),
        fraction_in(dVNC_FB_4hop, dVNC_EC_4hop, 'dVNC', '4-hop'),
        fraction_in(dVNC_FB_5hop, dVNC_EC_5hop, 'dVNC', '5-hop'),
        fraction_in(dSEZ_FB_casc, dSEZ_EC_casc, 'dSEZ', 'cascade_8hop'),
        fraction_in(dSEZ_FB_1hop, dSEZ_EC_1hop, 'dSEZ', '1-hop'),
        fraction_in(dSEZ_FB_2hop, dSEZ_EC_2hop, 'dSEZ', '2-hop'),
        fraction_in(dSEZ_FB_3hop, dSEZ_EC_3hop, 'dSEZ', '3-hop'),
        fraction_in(dSEZ_FB_4hop, dSEZ_EC_4hop, 'dSEZ', '4-hop'),
        fraction_in(dSEZ_FB_5hop, dSEZ_EC_5hop, 'dSEZ', '5-hop')
        ]

dfs = pd.concat(dfs, axis=0)

# %%
# plot fraction MBIN, DAN, MB-FBN downstream of dVNCs

ylim = (0,1)

# dVNCs
# feedback
fig, ax = plt.subplots(1,1, figsize=(1.5, .75))
sns.barplot(x=dfs[(dfs.upstream_partner=='dVNC') & (dfs.connection_type=='feedback') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].distance, 
                y=dfs[(dfs.upstream_partner=='dVNC') & (dfs.connection_type=='feedback') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].fraction, 
                hue=dfs[(dfs.upstream_partner=='dVNC') & (dfs.connection_type=='feedback') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].celltype, 
                order=['1-hop', '2-hop', '3-hop', '4-hop', '5-hop', 'cascade_8hop'],
                ax=ax)
ax.set(ylim=ylim)
plt.savefig('plots/feedback_MBINs-DANs-FBNs_ds-dVNC.pdf', format='pdf', bbox_inches='tight')

# efference copy
fig, ax = plt.subplots(1,1, figsize=(1.5, .75))
sns.barplot(x=dfs[(dfs.upstream_partner=='dVNC') & (dfs.connection_type=='efference_copy') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].distance, 
                y=dfs[(dfs.upstream_partner=='dVNC') & (dfs.connection_type=='efference_copy') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].fraction, 
                hue=dfs[(dfs.upstream_partner=='dVNC') & (dfs.connection_type=='efference_copy') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].celltype, 
                order=['1-hop', '2-hop', '3-hop', '4-hop', '5-hop', 'cascade_8hop'],
                ax=ax)
ax.set(ylim=ylim)
plt.savefig('plots/efference_MBINs-DANs-FBNs_ds-dVNC.pdf', format='pdf', bbox_inches='tight')

# dSEZs
# feedback
fig, ax = plt.subplots(1,1, figsize=(1.5, .75))
sns.barplot(x=dfs[(dfs.upstream_partner=='dSEZ') & (dfs.connection_type=='feedback') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].distance, 
                y=dfs[(dfs.upstream_partner=='dSEZ') & (dfs.connection_type=='feedback') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].fraction, 
                hue=dfs[(dfs.upstream_partner=='dSEZ') & (dfs.connection_type=='feedback') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].celltype, 
                order=['1-hop', '2-hop', '3-hop', '4-hop', '5-hop', 'cascade_8hop'],
                ax=ax)
ax.set(ylim=ylim)
plt.savefig('plots/feedback_MBINs-DANs-FBNs_ds-dSEZ.pdf', format='pdf', bbox_inches='tight')

# efference copy
fig, ax = plt.subplots(1,1, figsize=(1.5, .75))
sns.barplot(x=dfs[(dfs.upstream_partner=='dSEZ') & (dfs.connection_type=='efference_copy') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].distance, 
                y=dfs[(dfs.upstream_partner=='dSEZ') & (dfs.connection_type=='efference_copy') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].fraction, 
                hue=dfs[(dfs.upstream_partner=='dSEZ') & (dfs.connection_type=='efference_copy') & ((dfs.celltype=='DAN') | (dfs.celltype=='MB_FBN'))].celltype, 
                order=['1-hop', '2-hop', '3-hop', '4-hop', '5-hop', 'cascade_8hop'],
                ax=ax)
ax.set(ylim=ylim)
plt.savefig('plots/efference_MBINs-DANs-FBNs_ds-dSEZ.pdf', format='pdf', bbox_inches='tight')

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
# hop matrix from DN to MBIN and MB-FBN

# make a hop matrix plot for ds partners of dVNCs
# copy-pasted old code below, modify as needed

pairs = Promat.get_pairs(pairs_path=pairs_path)
edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=False)

# VNC layering with respect to sensories or motorneurons
threshold = 0.01
hops = 5
dVNC_pairs = Promat.load_pairs_from_annotation('mw dVNC', pairs)
ds_dVNCs = [[list(dVNC_pairs.loc[i, :].values)] + Promat.downstream_multihop(edges=edges, sources=list(dVNC_pairs.loc[i, :].values), hops=hops, pairs=pairs) for i in dVNC_pairs.index]
ds_dVNCs_names = [x[0][0] for x in ds_dVNCs]

MBINs_all = pymaid.get_skids_by_annotation('mw MBIN')
DANs_pairs = Promat.load_pairs_from_annotation('mw MBIN subclass_DAN', pairs, return_type='all_pair_ids_bothsides')
OANs_pairs = Promat.load_pairs_from_annotation('mw MBIN subclass_OAN', pairs, return_type='all_pair_ids_bothsides')
MBINs_pairs = Promat.extract_pairs_from_list(np.setdiff1d(MBINs_all, pymaid.get_skids_by_annotation('mw MBIN subclass_DAN') + pymaid.get_skids_by_annotation('mw MBIN subclass_OAN')), pairs)[0]

MBINs_all_pairs = pd.concat([DANs_pairs, OANs_pairs, MBINs_pairs])
MBINs_all_pairs.reset_index(drop=True)

ds_dVNC_A1 = pymaid.get_skids_by_annotation('mw A1 ds_dVNC')
ds_dVNC_layers,ds_dVNC_skids = Celltype_Analyzer.layer_id(ds_dVNCs, ds_dVNCs_names, MBINs_all)
DANs_pairs = Promat.extract_pairs_from_list(DANs, pairs)[0]
ds_dVNC_mat, ds_dVNC_mat_plotting = Promat.hop_matrix(ds_dVNC_skids.T, ds_dVNCs_names, MBINs_all_pairs.leftid, include_start=True)

fig, ax = plt.subplots(1,1,figsize=(5,5))
annotations = ds_dVNC_mat.loc[ds_dVNC_mat_plotting.sum(axis=1)!=0].astype(int).astype(str)
annotations[annotations=='0']=''
sns.heatmap(ds_dVNC_mat_plotting.loc[ds_dVNC_mat_plotting.sum(axis=1)!=0], annot = annotations, fmt = 's', square=True, cmap='Blues', ax=ax)
plt.savefig('plots/dVNC-MBIN_hop-matrix.pdf', format='pdf', bbox_inches='tight')