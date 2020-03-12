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

def get_connectors_group(skids):
    neurons = pymaid.get_neuron(skids)

    outputs = neurons.connectors[neurons.connectors['relation']==0]
    inputs = neurons.connectors[neurons.connectors['relation']==1]

    return(outputs, inputs)

#%%
contra_skids = pymaid.get_skids_by_annotation("mw brain crosses commissure")
left_skids = pymaid.get_skids_by_annotation("mw left")
right_skids = pymaid.get_skids_by_annotation("mw right")

contra_left = np.intersect1d(contra_skids, left_skids)
contra_right = np.intersect1d(contra_skids, right_skids)

contra_left_outputs, contra_left_inputs = get_connectors_group(contra_left)
contra_right_outputs, contra_right_inputs = get_connectors_group(contra_right)
print('done')

#%%
brain_skids = pymaid.get_skids_by_annotation("mw brain neurons")
ipsi_skids = np.setdiff1d(brain_skids, contra_skids)
ipsi_left = np.intersect1d(ipsi_skids, left_skids)
ipsi_right = np.intersect1d(ipsi_skids, right_skids)

ipsi_left_outputs, ipsi_left_inputs = get_connectors_group(ipsi_left)
ipsi_right_outputs, ipsi_right_inputs = get_connectors_group(ipsi_right)
print('done')

#%%
fig, ax = plt.subplots(1,1,figsize=(4,2))

sns.distplot(contra_left_outputs['x'], color = 'red', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(contra_left_inputs['x'], color = 'blue', ax = ax, hist = False, kde_kws = {'shade': True})

#%%
fig, ax = plt.subplots(1,1,figsize=(4,2))

sns.distplot(ipsi_left_outputs['x'], color = 'red', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(ipsi_left_inputs['x'], color = 'blue', ax = ax, hist = False, kde_kws = {'shade': True})

# %%
