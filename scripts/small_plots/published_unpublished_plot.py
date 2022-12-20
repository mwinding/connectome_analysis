#%%

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'arial'

pymaid.CatmaidInstance(url, token, name, password)
# %%
# all major brain types and published neurons

inputs = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs and ascending').name]
inputs = inputs + [pymaid.get_skids_by_annotation('mw A1 ascending unknown')]
inputs = [x for sublist in inputs for x in sublist]

outputs = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain outputs').name]
outputs = [x for sublist in outputs for x in sublist]

interneurons = pymaid.get_skids_by_annotation('mw brain neurons')
interneurons = list(np.setdiff1d(interneurons, outputs))

published = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('papers').name]
published = [x for sublist in published for x in sublist]

# %%
# 
cns = pymaid.get_volume('cns')
neuropil = pymaid.get_volume('neuropil')
cns.color = (250, 250, 250, .05)
neuropil.color = (250, 250, 250, .05)

pub_inputs_nl = pymaid.get_neurons(np.intersect1d(inputs, published))
unpub_inputs_nl = pymaid.get_neurons(np.setdiff1d(inputs, published))

pub_outputs_nl = pymaid.get_neurons(np.intersect1d(outputs, published))
unpub_outputs_nl = pymaid.get_neurons(np.setdiff1d(outputs, published))

pub_interneurons_nl = pymaid.get_neurons(np.intersect1d(interneurons, published))
unpub_interneurons_nl = pymaid.get_neurons(np.setdiff1d(interneurons, published))

fig, ax = navis.plot2d(x=[pub_inputs_nl, cns], connectors_only=False, color='grey', alpha=0.5)
ax.azim = -90
ax.elev = -90
ax.dist = 3.5
plt.show()
fig.savefig('plots/published_inputs.pdf', format='pdf', bbox_format = 'tight')

fig, ax = navis.plot2d(x=[unpub_inputs_nl, cns], connectors_only=False, color=sns.color_palette()[1], alpha=0.5)
ax.azim = -90
ax.elev = -90
ax.dist = 3.5
plt.show()
fig.savefig('plots/unpublished_inputs.pdf', format='pdf', bbox_format = 'tight')

# %%
# plot output neurons

fig, ax = navis.plot2d(x=[pub_outputs_nl, cns], connectors_only=False, color='grey', alpha=0.5)
ax.azim = -90
ax.elev = -90
ax.dist = 3.5
plt.show()
plt.savefig('plots/published_outputs.pdf', format='pdf', bbox_format = 'tight')

fig, ax = navis.plot2d(x=[unpub_outputs_nl, cns], connectors_only=False, color=sns.color_palette()[1], alpha=0.5)
ax.azim = -90
ax.elev = -90
ax.dist = 3.5
plt.show()
plt.savefig('plots/unpublished_outputs.pdf', format='pdf', bbox_format = 'tight')

# %%
# plot interneurons
alpha = 0.025
fig, ax = navis.plot2d(x=[pub_interneurons_nl, cns], connectors_only=False, color='grey', alpha=alpha)
ax.azim = -90
ax.elev = -90
ax.dist = 3.5
plt.show()
fig.savefig(f'small_plots/plots/published_interneurons_alpha{alpha}.pdf', format='pdf', bbox_format = 'tight')

fig, ax = navis.plot2d(x=[unpub_interneurons_nl, cns], connectors_only=False, color=sns.color_palette()[1], alpha=alpha)
ax.azim = -90
ax.elev = -90
ax.dist = 3.5
plt.show()
fig.savefig(f'small_plots/plots/unpublished_interneurons_alpha{alpha}.pdf', format='pdf', bbox_format = 'tight')

# %%
