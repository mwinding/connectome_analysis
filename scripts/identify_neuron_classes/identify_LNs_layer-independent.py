#%%
# script didn't give reasonable LNs in my opinion; will use the other version

import pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from data_settings import data_date, pairs_path, data_date_projectome
from contools import Promat, Prograph, Celltype, Celltype_Analyzer, Adjacency_matrix

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

edges_ad = Promat.pull_edges('ad', threshold=0.01, data_date=data_date, pairs_combined=True)
adj_all = Promat.pull_adj('summed')
adj_all = Adjacency_matrix(adj_all, [], 'all-all', pairs, fraction_input=False)
adj_all_pairwise = adj_all.adj_pairwise

outputs = pd.read_csv(f'data/adj/outputs_{data_date}.csv', index_col=0)

# %%
pairs = Promat.get_pairs(pairs_path)
LN_pairids = Promat.load_pairs_from_annotation('mw LN', pairs, return_type='all_pair_ids')
brain_pairids = Promat.load_pairs_from_annotation('mw brain neurons', pairs, return_type='all_pair_ids')
brain_pairids = list(np.intersect1d(brain_pairids, [x[1] for x in adj_all_pairwise.index]))

brain_partners_ad = Promat.find_all_partners(brain_pairids, edges_ad, pairs_path)
brain_partners_ad = brain_partners_ad.set_index('source_pairid')
brain_partners_summed = Promat.find_all_partners(brain_pairids, edges_summed, pairs_path)
brain_partners_summed = brain_partners_summed.set_index('source_pairid')

LN_metric = []
for pairid in brain_pairids:
    us = brain_partners_ad.loc[pairid, 'upstream']
    ds_of_us = list(brain_partners_ad.loc[np.intersect1d(us, brain_partners_ad.index)].downstream.values)
    ds_of_us = [x for sublist in ds_of_us for x in sublist]

    # fraction of synapses
    cols = [x[1] for x in adj_all_pairwise.columns]
    LN_us = adj_all_pairwise.loc[(slice(None), pairid), (slice(None), np.intersect1d(us, cols))].sum(axis=1)[0]
    LN_ds = adj_all_pairwise.loc[(slice(None), pairid), (slice(None), np.intersect1d(ds_of_us, cols))].sum(axis=1)[0]

    pairids = Promat.get_paired_skids(pairid, pairs)
    pairid_outputs = outputs.loc[pairids, :].sum(axis=0).sum()/len(pairids) # take average so divide by len(pairids); takes into account nonpaired neuorns (len=1)

    if(pairid_outputs>0):
        LN_us_metric = LN_us/pairid_outputs
        LN_ds_metric = LN_ds/pairid_outputs
        LN_combo_metric = (LN_ds+LN_us)/pairid_outputs
    else:
        LN_us_metric = 0
        LN_ds_metric = 0
        LN_combo_metric = 0

    LN_metric.append([pairid, LN_us_metric, LN_ds_metric, LN_combo_metric])

    '''
    # simply using fraction of downstream neurons; doesn't work well for descending neurons for example...
    # also picks up all KCs
    pairid_ds = brain_partners_summed.loc[pairid, 'downstream']

    if(len(pairid_ds)>0):
        LN_us_metric = len(np.intersect1d(pairid_ds, us))/len(pairid_ds)
        LN_ds_metric = len(np.intersect1d(pairid_ds, ds_of_us))/len(pairid_ds)
        LN_combo_metric = LN_us_metric + LN_ds_metric
    else:
        LN_us_metric = 0
        LN_ds_metric = 0
        LN_combo_metric = 0
    '''

LN_metric_df = pd.DataFrame(LN_metric, columns=['pairid', 'LN_us_metric', 'LN_ds_metric', 'LN_combo_metric'])

# %%
