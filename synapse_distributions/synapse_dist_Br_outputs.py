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
