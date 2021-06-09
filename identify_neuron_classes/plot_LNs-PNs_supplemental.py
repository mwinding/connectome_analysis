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

import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load and plot some LNs
LNs = pymaid.get_skids_by_annotation('mw LNs to plot')
LNs = pm.Promat.extract_pairs_from_list(LNs)[0]

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

fig.savefig(f'identify_neuron_classes/plots/morpho_LNs.png', format='png', dpi=300, transparent=True)

# %%
# load and plot some PNs
PNs = pymaid.get_skids_by_annotation('mw PNs to plot')
PNs = pm.Promat.extract_pairs_from_list(PNs)[0]

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

fig.savefig(f'identify_neuron_classes/plots/morpho_PNs.png', format='png', dpi=300, transparent=True)

# %%
# plot all LNs types
import math

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
LN = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

LN_ipsi = np.intersect1d(LN, ipsi)
LN_bilat = np.intersect1d(LN, bilateral)
LN_contra = np.intersect1d(LN, contra)

LN_ipsi, _, LN_ipsi_nonpaired = pm.Promat.extract_pairs_from_list(LN_ipsi)
LN_ipsi.loc[42]=[LN_ipsi_nonpaired.values[0][0], LN_ipsi_nonpaired.values[0][0]] # *** warning, hard-coded

LN_bilat, _, LN_bilat_nonpaired = pm.Promat.extract_pairs_from_list(LN_bilat) # no nonpaired neurons
LN_contra, _, LN_contra_nonpaired = pm.Promat.extract_pairs_from_list(LN_contra) # no nonpaired neurons
LN_color = '#5D8B90'

# ipsi LNs
n_cols = 8
n_rows = math.ceil(len(LN_ipsi)/n_cols) # round up to determine how many rows there should be
alpha = 1

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
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_ipsi_LNs.png', format='png', dpi=300, transparent=True)

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
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_bilat_LNs.png', format='png', dpi=300, transparent=True)

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
    ax.dist = 6
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

fig.savefig(f'identify_neuron_classes/plots/morpho_contra_LNs.png', format='png', dpi=300, transparent=True)
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
