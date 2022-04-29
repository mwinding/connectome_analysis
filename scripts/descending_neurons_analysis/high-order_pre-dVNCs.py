# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# load high-order (5th-order) pre-dVNCs
pdVNC_ho = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 5th_order')

# load pairs and celltypes
pairs = Promat.get_pairs(pairs_path=pairs_path)
_, celltypes = Celltype_Analyzer.default_celltypes()

# load edges
ad_edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=True)

# %%
# plot upstream and downstream partners
pdVNC_ho_pairids = Promat.extract_pairs_from_list(pdVNC_ho, pairs, return_pair_ids=True)

# plot partners (chromosome plots)
partners = Promat.find_all_partners(pdVNC_ho_pairids, ad_edges, pairs_path)
Celltype_Analyzer.chromosome_plot(partners, 'plots/pdVNC-ho_chromosome-plots', celltypes, simple=True)

# annotate partners in CATMAID
'''
annots = []
for i in partners.index:
    annot = f'mw {partners.loc[i].source_pairid} upstream partners'
    pymaid.add_annotations(partners.loc[i, 'upstream'], annot)
    annots.append(annot)

    annot = f'mw {partners.loc[i].source_pairid} pair'
    pymaid.add_annotations(partners.loc[i, 'source_pair'], annot)
    annots.append(annot)
    
    annot = f'mw {partners.loc[i].source_pairid} downstream partners'
    pymaid.add_annotations(partners.loc[i, 'downstream'], annot)
    annots.append(annot)

pymaid.add_meta_annotations([annot for annot in annots], 'mw pdVNC-ho partners')
'''
# %%
# plot projectome data for associated dVNCs


# %%
# plot neurons morphologies together and by pair

# organize pairs
pdVNC_ho_pairs = Promat.extract_pairs_from_list(pdVNC_ho, pairs)[0].values

# plot pairs
neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .05)
#color = '#dab6b3'
color = 'tab:gray'

n_rows = 1
n_cols = 6
alpha = 1
cn_size = 0.5

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, skids in enumerate(pdVNC_ho_pairs):
    neurons = pymaid.get_neurons(skids)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    #navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=color, alpha=alpha, ax=ax)
    navis.plot2d(x=[neurons, neuropil], connectors=True, cn_size=cn_size, color=color, alpha=alpha, ax=ax, method='3d_complex')
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'plots/morpho_5th-order_pre-dVNCs.png', format='png', dpi=300, transparent=True)

# plot all
all_neurons = [x for sublist in pdVNC_ho_pairs for x in sublist]

n_rows = 1
n_cols = 1
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

neurons = pymaid.get_neurons(all_neurons)

ax = fig.add_subplot(projection="3d")
navis.plot2d(x=[neurons, neuropil], connectors=True, cn_size=cn_size, color=color, alpha=alpha, ax=ax, method='3d_complex')
ax.azim = -90
ax.elev = -90
ax.dist = 6
ax.set_xlim3d((-4500, 110000))
ax.set_ylim3d((-4500, 110000))
fig.savefig(f'plots/morpho_5th-order_pre-dVNCs_all.png', format='png', dpi=300, transparent=True)

# %%
# connectivity between these pre-dVNCs
pymaid.clear_cache()
pdVNC_ho_partners = Celltype_Analyzer.get_skids_from_meta_annotation('mw pdVNC-ho partners', split=True, return_celltypes=True)
pdVNC_ho_partners_cta = Celltype_Analyzer(pdVNC_ho_partners)

fig, ax = plt.subplots(1,1)
adj = Promat.pull_adj(type_adj='ad', data_date=data_date)
pdVNC_ho_connect = pdVNC_ho_partners_cta.connectivity(adj=adj, normalize_pre_num=True)
sns.heatmap(pdVNC_ho_connect, square=True, cmap='Blues', ax=ax)
fig.savefig(f'plots/pdVNC-ho_partner_connectivity.pdf', format='pdf')

# only pairs connectivity
fig, ax = plt.subplots(1,1)
pair_bool = [True if string.find('pair')!=-1 else False for string in pdVNC_ho_connect.index]
pdVNC_ho_connect_raw = pdVNC_ho_partners_cta.connectivity(adj=adj)
sns.heatmap(pdVNC_ho_connect_raw.iloc[pair_bool, pair_bool], square=True, cmap='Blues', ax=ax)
fig.savefig(f'plots/pdVNC-ho_connectivity.pdf', format='pdf')

# only over-threshold connections
ad_edges_ho = ad_edges.copy()
ad_edges_ho.index = ad_edges_ho.upstream_pair_id
ad_edges_ho = ad_edges_ho.loc[pdVNC_ho_pairids]
ad_edges_ho.index = ad_edges_ho.downstream_pair_id
ad_edges_ho = ad_edges_ho.loc[np.intersect1d(pdVNC_ho_pairids, ad_edges_ho.index)]
ad_edges_ho

# similarity between upstream/downstream partners
partner_bool = [False if string.find('pair')!=-1 else True for string in pdVNC_ho_connect.index]
pdVNC_ho_partners_only = [partners for i, partners in enumerate(pdVNC_ho_partners) if partner_bool[i]]
pdVNC_ho_partners_only_cta = Celltype_Analyzer(pdVNC_ho_partners_only)

fig, ax = plt.subplots(1,1)
sns.heatmap(pdVNC_ho_partners_only_cta.compare_membership(sim_type='dice'), square=True, ax=ax)
fig.savefig(f'plots/pdVNC-ho_shared-partners.pdf', format='pdf')

# %%
