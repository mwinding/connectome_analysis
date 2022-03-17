#%%

import pymaid
import navis
#import pyoctree
from pymaid_creds import url, name, password, token
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import connectome_tools.process_skeletons as skel

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

def get_connectors_group(skids, output_separate=False):

    neurons = []
    for skid in skids:
        neurons.append(pymaid.get_neuron(skid))
    axons, dendrites = skel.split_axons_dendrites(neurons, 'mw axon split')

    axon_outputs = []
    dendrite_inputs = []
    for axon, dendrite in zip(axons, dendrites):
        axon_outputs.append(axon.connectors[axon.connectors['type']==0])
        dendrite_inputs.append(dendrite.connectors[dendrite.connectors['type']==1])

    if(output_separate):
        return(axon_outputs, dendrite_inputs)
    if(output_separate==False):
        return(pd.concat(axon_outputs, axis=0), pd.concat(dendrite_inputs, axis=0))

#%%
# loading neurons, splitting axon/dendrite, and identifying axon and dendrite connectors
ipsi_skids = pymaid.get_skids_by_annotation("mw ipsilateral axon")
bilateral_skids = pymaid.get_skids_by_annotation("mw bilateral axon")
contra_skids = pymaid.get_skids_by_annotation("mw contralateral axon")
contra_contra_skids = pymaid.get_skids_by_annotation('mw contralateral dendrite')
contra_skids = np.setdiff1d(contra_skids, contra_contra_skids) # remove neurons that are contralateral axon + dendrite

#left_skids = pymaid.get_skids_by_annotation("mw left")
right_skids = pymaid.get_skids_by_annotation("mw right")
brain_skids = pymaid.get_skids_by_annotation("mw brain neurons")
brain_right = np.intersect1d(right_skids, brain_skids)

ipsi_right = np.intersect1d(ipsi_skids, right_skids)
bilateral_right = np.intersect1d(bilateral_skids, right_skids)
contra_right = np.intersect1d(contra_skids, right_skids)

ipsi_right_outputs, ipsi_right_inputs = get_connectors_group(ipsi_right)
bilateral_right_outputs, bilateral_right_inputs = get_connectors_group(bilateral_right)
contra_right_outputs, contra_right_inputs = get_connectors_group(contra_right)

#%%
# plot axon output and dendrite input spatial distributions

min_x = 939 #based on 'cns' volume
max_x = 105096 #based on 'cns' volume

commissure_min_x = 51500 #based on 'Brain Commissure' volume
commissure_max_x = 56000 #based on 'Brain Commissure' volume

width = 1.15
height = 0.3

fig, ax = plt.subplots(1,1,figsize=(width,height))
sns.distplot(ipsi_right_inputs['x'], color = '#5DB2E2', ax = ax, hist = False, kde_kws = {'shade': True, 'linewidth': 0.5})
sns.distplot(ipsi_right_outputs['x'], color = '#1F77B4', ax = ax, hist = False, kde_kws = {'shade': True, 'linewidth': 0.5})
plt.axvline(commissure_min_x, color='gray', linewidth=0.5)
plt.axvline(commissure_max_x, color='gray', linewidth=0.5)
ax.set(xlim=(max_x, min_x), yticks=([]), xticks=([]), xlabel='')
plt.savefig('synapse_distributions/plots/ipsi_right.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(width,height))
sns.distplot(bilateral_right_inputs['x'], color = '#FFB176', ax = ax, hist = False, kde_kws = {'shade': True, 'linewidth': 0.5})
sns.distplot(bilateral_right_outputs['x'], color = '#FF7F0E', ax = ax, hist = False, kde_kws = {'shade': True, 'linewidth': 0.5})
plt.axvline(commissure_min_x, color='gray', linewidth=0.5)
plt.axvline(commissure_max_x, color='gray', linewidth=0.5)
ax.set(xlim=(max_x, min_x), yticks=([]), xticks=([]), xlabel='')
plt.savefig('synapse_distributions/plots/bilateral_right.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(width,height))
sns.distplot(contra_right_inputs['x'], color = '#74E274', ax = ax, hist = False, kde_kws = {'shade': True, 'linewidth': 0.5})
sns.distplot(contra_right_outputs['x'], color = '#2CA02C', ax = ax, hist = False, kde_kws = {'shade': True, 'linewidth': 0.5})
plt.axvline(commissure_min_x, color='gray', linewidth=0.5)
plt.axvline(commissure_max_x, color='gray', linewidth=0.5)
ax.set(xlim=(max_x, min_x), yticks=([]), xticks=([]), xlabel='')
plt.savefig('synapse_distributions/plots/contra_right.pdf', format='pdf', bbox_inches='tight')
#%%
