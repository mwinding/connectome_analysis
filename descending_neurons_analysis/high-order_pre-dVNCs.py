# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# load high-order (5th-order) pre-dVNCs
pdVNC_ho = list(np.unique(ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 5th_order')))


# %%
# plot upstream and downstream partners

ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
pairs = pm.Promat.get_pairs()
_, celltypes = ct.Celltype_Analyzer.default_celltypes()

# plot partners (chromosome plots)
partners = pm.Promat.find_all_partners(pdVNC_ho_pairids, ad_edges)
ct.chromosome_plot(partners, 'descending_neurons_analysis/plots/pdVNC-ho_chromosome-plots', celltypes, simple=True)

# annotate partners in CATMAID
[pymaid.add_annotations(partners.loc[i, 'downstream'], f'mw {partners.loc[i].source_pairid} downstream partners') for i in partners.index]
[pymaid.add_annotations(partners.loc[i, 'upstream'], f'mw {partners.loc[i].source_pairid} downstream partners') for i in partners.index]

# %%
# plot projectome data for associated dVNCs



# %%
# plot neurons morphologies together and by pair

# organize pairs
pairs = pm.Promat.get_pairs()
pdVNC_ho_pairs = pm.Promat.extract_pairs_from_list(pdVNC_ho, pairs)[0].values

# plot pairs
neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .05)
color = '#dab6b3'

n_rows = 1
n_cols = 6
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, skids in enumerate(pdVNC_ho_pairs):
    neurons = pymaid.get_neurons(skids)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'descending_neurons_analysis/plots/morpho_5th-order_pre-dVNCs.png', format='png', dpi=300, transparent=True)

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
navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=color, alpha=alpha, ax=ax)
ax.azim = -90
ax.elev = -90
ax.dist = 6
ax.set_xlim3d((-4500, 110000))
ax.set_ylim3d((-4500, 110000))
fig.savefig(f'descending_neurons_analysis/plots/morpho_5th-order_pre-dVNCs_all.png', format='png', dpi=300, transparent=True)

# %%
