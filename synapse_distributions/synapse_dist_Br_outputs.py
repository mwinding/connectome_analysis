#%%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

rm = pymaid.CatmaidInstance(url, token, name, password)

def get_connectors_group(annotation):
    skids = pymaid.get_skids_by_annotation(annotation)
    neurons = pymaid.get_neuron(skids)

    outputs = neurons.connectors[neurons.connectors['relation']==0]
    inputs = neurons.connectors[neurons.connectors['relation']==1]

    return(outputs, inputs)

#%%

dVNC_outputs, dVNC_inputs = get_connectors_group("mw dVNC")
dSEZ_outputs, dSEZ_inputs = get_connectors_group("mw dSEZ")
RG_outputs, RG_inputs = get_connectors_group("mw RG")

#%%
fig, ax = plt.subplots(1,1,figsize=(8,2))

sns.distplot(dSEZ_outputs['z'], color = 'purple', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(dVNC_outputs['z'], color = 'black', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(RG_outputs['z'], color = 'red', ax = ax, hist = False, kde_kws = {'shade': True})


# %%
fig, ax = plt.subplots(1,1,figsize=(4,2))

sns.distplot(dVNC_outputs['y'], color = 'purple', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(dSEZ_outputs['y'], color = 'black', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(RG_outputs['y'], color = 'red', ax = ax, hist = False, kde_kws = {'shade': True})

# %%
# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(4,1))
ax.set(xlim = (0, 100000))
sns.distplot(RG_inputs['z'], color = 'lightgreen', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(RG_outputs['z'], color = 'green', ax = ax, hist = False, kde_kws = {'shade': True})

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('synapse_distributions/plots/RG_outputs.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1,1,figsize=(4,1))
ax.set(xlim = (0, 100000))
sns.distplot(dSEZ_inputs['z'], color = 'moccasin', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(dSEZ_outputs['z'], color = 'tan', ax = ax, hist = False, kde_kws = {'shade': True})

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('synapse_distributions/plots/dSEZ_outputs.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1,1,figsize=(8,1))
ax.set(xlim = (0, 200000))
sns.distplot(dVNC_inputs['z'], color = 'salmon', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(dVNC_outputs['z'], color = 'crimson', ax = ax, hist = False, kde_kws = {'shade': True})

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('synapse_distributions/plots/dVNC_outputs.pdf', format='pdf', bbox_inches='tight')

# %%
# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(4,1))
ax.set(xlim = (0, 100000))
sns.distplot(RG_inputs['y'], color = 'lightgreen', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(RG_outputs['y'], color = 'green', ax = ax, hist = False, kde_kws = {'shade': True})

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('synapse_distributions/plots/RG_outputs_y.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1,1,figsize=(4,1))
ax.set(xlim = (0, 100000))
sns.distplot(dSEZ_inputs['y'], color = 'moccasin', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(dSEZ_outputs['y'], color = 'tan', ax = ax, hist = False, kde_kws = {'shade': True})

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('synapse_distributions/plots/dSEZ_outputs_y.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1,1,figsize=(4,1))
ax.set(xlim = (0, 100000))
sns.distplot(dVNC_inputs['y'], color = 'salmon', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(dVNC_outputs['y'], color = 'crimson', ax = ax, hist = False, kde_kws = {'shade': True})

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('synapse_distributions/plots/dVNC_outputs_y.pdf', format='pdf', bbox_inches='tight')



# %%
