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

rm = pymaid.CatmaidInstance(url, name, password, token)

def get_connectors_group(annotation):
    skids = pymaid.get_skids_by_annotation(annotation)
    neurons = pymaid.get_neuron(skids)

    outputs = neurons.connectors[neurons.connectors['relation']==0]
    inputs = neurons.connectors[neurons.connectors['relation']==1]

    return(outputs, inputs)

#%%

ORN_outputs, ORN_inputs = get_connectors_group("mw ORN")
thermo_outputs, thermo_inputs = get_connectors_group("mw thermosensories")
visual_outputs, visual_inputs = get_connectors_group("mw photoreceptors")
AN_outputs, AN_inputs = get_connectors_group("mw AN sensories")
MN_outputs, MN_inputs = get_connectors_group("mw MN sensories")
noci_outputs, noci_inputs = get_connectors_group("mw A00c")

#%%
fig, ax = plt.subplots(1,1,figsize=(8,2))

sns.distplot(ORN_outputs['z'], color = 'red', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(thermo_outputs['z'], color = 'orange', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(visual_outputs['z'], color = 'green', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(AN_outputs['z'], color = 'black', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(MN_outputs['z'], color = 'grey', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(noci_outputs['z'], color = 'brown', ax = ax, hist = False, kde_kws = {'shade': True})


# %%
fig, ax = plt.subplots(1,1,figsize=(4,2))

sns.distplot(ORN_outputs['y'], color = 'red', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(thermo_outputs['y'], color = 'orange', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(visual_outputs['y'], color = 'green', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(AN_outputs['y'], color = 'black', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(MN_outputs['y'], color = 'grey', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(noci_outputs['y'], color = 'brown', ax = ax, hist = False, kde_kws = {'shade': True})


# %%
