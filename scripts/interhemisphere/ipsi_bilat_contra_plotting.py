#%%

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random

import cmasher as cmr

from contools import Celltype, Celltype_Analyzer, Promat
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %% 
# load axon/dendrite types
brain = pymaid.get_skids_by_annotation('mw brain neurons')
pdiff = pymaid.get_skids_by_annotation('mw partially differentiated')
incomplete = pymaid.get_skids_by_annotation('mw brain very incomplete')

brain = np.setdiff1d(brain, pdiff + incomplete)

ipsi_dendrite = np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral dendrite'), brain)
contra_dendrite = np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral dendrite'), brain)
bilateral_dendrite = np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral dendrite'), brain)

ipsi_axon = np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), brain)
contra_axon = np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), brain)
bilateral_axon = np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), brain)

# %%
# number of dendrite_axon types

ipsi_ipsi = len(np.intersect1d(ipsi_dendrite, ipsi_axon))
ipsi_bilateral = len(np.intersect1d(ipsi_dendrite, bilateral_axon))
ipsi_contra = len(np.intersect1d(ipsi_dendrite, contra_axon))
bilateral_ipsi = len(np.intersect1d(bilateral_dendrite, ipsi_axon))
bilateral_bilateral = len(np.intersect1d(bilateral_dendrite, bilateral_axon))
bilateral_contra = len(np.intersect1d(bilateral_dendrite, contra_axon))
contra_ipsi = len(np.intersect1d(contra_dendrite, ipsi_axon))
contra_bilateral = len(np.intersect1d(contra_dendrite, bilateral_axon))
contra_contra = len(np.intersect1d(contra_dendrite, contra_axon))

neuron_types = pd.DataFrame([[ipsi_ipsi, ipsi_bilateral, ipsi_contra], 
                            [bilateral_ipsi, bilateral_bilateral, bilateral_contra],
                            [contra_ipsi, contra_bilateral, contra_contra]], 
                            index = ['ipsi', 'bilateral', 'contra'], 
                            columns = ['ipsi', 'bilateral', 'contra'])

fig, ax = plt.subplots(1,1, figsize=(0.75,0.75))
sns.heatmap(neuron_types, annot=True, fmt = 'd', ax=ax, vmax=600, cmap='Blues', cbar=False)
ax.set(xticks=([]), yticks=([]))
fig.savefig('plots/cell_types_dendrite-axon.pdf', format='pdf', bbox_inches='tight')

# %%
# plotting example neurons for supplemental figure

interhemi_names = ['mw ipsi-ipsi to plot', 'mw bilat-ipsi to plot', 'mw contra-ipsi to plot', 
                    'mw ipsi-bilat to plot', 'mw bilat-bilat to plot', 'mw contra-bilat to plot', 'mw contra-contra to plot']
interhemi_types = [pymaid.get_skids_by_annotation(name) for name in interhemi_names]
interhemi_types = [ct.Celltype(interhemi_names[i].replace('mw ', '').replace(' to plot', ''), cell_type) for i, cell_type in enumerate(interhemi_types)]

neuropil = pymaid.get_volume('PS_Neuropil_manual')
neuropil.color = (250, 250, 250, .075)
colors = ['#1F77B4', '#FF7F0E', '#2ca02c', '#652d90', '#c03d3e', '#603813', '#00a69c']

n_rows = 1
n_cols = 3
alpha = 1

for i, celltype in enumerate(interhemi_types):

    fig = plt.figure(figsize=(n_cols*2, n_rows*2))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
    axs = np.empty((n_rows, n_cols), dtype=object)


    neurons = pymaid.get_neurons(celltype.get_skids())
    if(type(neurons)==pymaid.core.CatmaidNeuron):
        neurons = [neurons]

    for j, skid in enumerate(neurons):

        inds = np.unravel_index(j, shape=(n_rows, n_cols))
        ax = fig.add_subplot(gs[inds], projection="3d")
        axs[inds] = ax
        navis.plot2d(x=[skid, neuropil], connectors = True, color='tab:gray', alpha=alpha, ax=ax, method='3d_complex')
        ax.azim = -90
        ax.elev = -90
        ax.dist = 6
        ax.set_xlim3d((-4500, 110000))
        ax.set_ylim3d((-4500, 110000))

    fig.savefig(f'plots/morpho_{celltype.get_name()}.png', format='png', dpi=300, transparent=True)

# %%
