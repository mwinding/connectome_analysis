#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import pairs_path
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

pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
# load and plot some LNs
LNs = pymaid.get_skids_by_annotation('mw LNs to plot')
LNs = Promat.extract_pairs_from_list(LNs, pairList=pairs)[0]

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
LN_color = '#5D8B90'

n_rows = 3
n_cols = 5
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, index in enumerate(LNs.index):
    neurons = pymaid.get_neurons(LNs.loc[index, :].values)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=LN_color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'plots/morpho_LNs.png', format='png', dpi=300, transparent=True)

# %%
# load and plot some PNs
PNs = pymaid.get_skids_by_annotation('mw PNs to plot')
PNs = pm.Promat.extract_pairs_from_list(PNs, pairList=pairs)[0]

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
PN_color = '#1D79B7'

n_rows = 3
n_cols = 5
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, index in enumerate(PNs.index):
    neurons = pymaid.get_neurons(LNs.loc[index, :].values)

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=PN_color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'plots/morpho_PNs.png', format='png', dpi=300, transparent=True)

# %%
# plot all LNs types
import math

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
LN = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

LN_ipsi = np.intersect1d(LN, ipsi)
LN_bilat = np.intersect1d(LN, bilateral)
LN_contra = np.intersect1d(LN, contra)

# create dataframes with left/right neuron pairs; nonpaired neurons had duplicated skids in left/right column
LN_ipsi = Promat.load_pairs_from_annotation(annot='', pairList=pairs, return_type='all_pair_ids_bothsides', skids=LN_ipsi, use_skids=True)
LN_bilat = Promat.load_pairs_from_annotation(annot='', pairList=pairs, return_type='all_pair_ids_bothsides', skids=LN_bilat, use_skids=True)
LN_contra = Promat.load_pairs_from_annotation(annot='', pairList=pairs, return_type='all_pair_ids_bothsides', skids=LN_contra, use_skids=True)
LN_color = '#5D8B90'

# sort LN_ipsi with published neurons first
pub = [7941652, 7941642, 7939979, 8311264, 7939890, 5291791, 8102935, 8877971, 8274021, 10555409, 7394271, 8273369, 17414715, 8700125, 8480418, 15571194]
pub_names = ['Broad D1', 'Broad D2', 'Broad T1', 'Broad T2', 'Broad T3', 'picky 0', 'picky 1', 'picky 2', 'picky 3', 'picky 4', 'choosy 1', 'choosy 2', 'ChalOLP', 'GlulOLP', 'OLP4', 'APL']

pub = pub + [4985759, 4620453]
pub_names = pub_names + ['']*(len(LN_ipsi.index) - len(pub_names))
LN_ipsi = LN_ipsi.set_index('leftid', drop=False)
LN_ipsi = LN_ipsi.loc[pub + list(np.setdiff1d(LN_ipsi.index, pub)), :]
LN_ipsi.index = range(len(LN_ipsi.index))

# ipsi LNs
n_cols = 8
n_rows = math.ceil(len(LN_ipsi)/n_cols) # round up to determine how many rows there should be
alpha = 1
zoom = 5.5

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, index in enumerate(LN_ipsi.index):
    neurons = pymaid.get_neurons(np.unique(LN_ipsi.loc[index, :].values))

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=LN_color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = zoom
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))
    ax.text(x=(ax.get_xlim()[0] + ax.get_xlim()[1])/2 + ax.get_xlim()[1]*0.05, y=ax.get_ylim()[1]*4/5, horizontalalignment="center", z=0, 
                    s=pub_names[i], transform=ax.transData, color=LN_color, alpha=1, fontsize=10)

fig.savefig(f'plots/morpho_ipsi_LNs.png', format='png', dpi=300, transparent=True)

# bilat LNs
n_cols = 8
n_rows = math.ceil(len(LN_bilat)/n_cols) # round up to determine how many rows there should be
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, index in enumerate(LN_bilat.index):
    neurons = pymaid.get_neurons(np.unique(LN_bilat.loc[index, :].values))

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=LN_color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = zoom
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'plots/morpho_bilat_LNs.png', format='png', dpi=300, transparent=True)

# contra LNs
n_cols = 8
n_rows = math.ceil(len(LN_contra)/n_cols) # round up to determine how many rows there should be
alpha = 1

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, index in enumerate(LN_contra.index):
    neurons = pymaid.get_neurons(np.unique(LN_contra.loc[index, :].values))

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=LN_color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = zoom
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'plots/morpho_contra_LNs.png', format='png', dpi=300, transparent=True)
# %%
# plot many 2nd-order PNs

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
LN = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
MBINs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MBINs')
MBONs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain MBONs')
KCs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain KCs')
uPNs = pymaid.get_skids_by_annotation('mw uPN')
mPNs = pymaid.get_skids_by_annotation('mw mPN')
outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')
exclude = LN + MBINs + MBONs + KCs + uPNs + mPNs + outputs + [5327961, 11184236] + [12740290, 2432564] #potential LN that didn't pass threshold and a quasi-dSEZ neuron

PN_guste = pymaid.get_skids_by_annotation('mw gustatory-external 2nd_order')
PN_guste = np.setdiff1d(PN_guste, exclude)
PN_guste, _, _nonpaired = pm.Promat.extract_pairs_from_list(PN_guste)

PN_gustp = pymaid.get_skids_by_annotation('mw gustatory-pharyngeal 2nd_order')
PN_gustp = np.setdiff1d(PN_gustp, exclude)
PN_gustp, _, _nonpaired = pm.Promat.extract_pairs_from_list(PN_gustp)

PN_ent = pymaid.get_skids_by_annotation('mw enteric 2nd_order')
PN_ent = np.setdiff1d(PN_ent, exclude)
PN_ent, _, _nonpaired = pm.Promat.extract_pairs_from_list(PN_ent)

PNs = pd.concat([PN_guste.loc[0:9, :], PN_gustp.loc[0:8, :], PN_ent.loc[0:8, :]]).reset_index(drop=True)


# contra LNs
n_cols = 4
n_rows = math.ceil(len(PNs)/n_cols) # round up to determine how many rows there should be
alpha = 1
PN_color = '#1D79B7'

fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)

for i, index in enumerate(PNs.index):
    neurons = pymaid.get_neurons(np.unique(PNs.loc[index, :].values))

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    navis.plot2d(x=[neurons, neuropil], connectors_only=False, color=PN_color, alpha=alpha, ax=ax)
    ax.azim = -90
    ax.elev = -90
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_non-uPN-mPN_PNs.png', format='png', dpi=300, transparent=True)

# %%
