#%%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

#%%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

connectors_split_center = pd.read_csv('axon_dendrite_data/connectdists_all_centeredsplit.csv')

splittable_inputs = []
splittable_outputs = []
for i in tqdm(range(len(connectors_split_center))):
    if(connectors_split_center.iloc[i]['type']=='postsynaptic'):
        splittable_inputs.append(connectors_split_center.iloc[i]['distance'])
    if(connectors_split_center.iloc[i]['type']=='presynaptic'):
        splittable_outputs.append(connectors_split_center.iloc[i]['distance'])

#%%
connectors_unsplittable = pd.read_csv('axon_dendrite_data/unsplittable_connectordists_raw.csv')

unsplittable_inputs = []
unsplittable_outputs = []
# calculated mean distance of inputs/outputs
mean = 65728.75903161563
for i in tqdm(range(len(connectors_unsplittable))):
    if(connectors_unsplittable.iloc[i]['type']=='postsynaptic'):
        unsplittable_inputs.append(connectors_unsplittable.iloc[i]['distance_root']-mean)
    if(connectors_unsplittable.iloc[i]['type']=='presynaptic'):
        unsplittable_outputs.append(connectors_unsplittable.iloc[i]['distance_root']-mean)

#%%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(splittable_inputs, color = 'royalblue', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(splittable_outputs, color = 'crimson', ax = ax, hist = False, kde_kws = {'shade': True})

ax.set(xlim = (-125000, 200000))
plt.axvline(x=0, color = 'gray')
ax.set_yticks([])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150, 200])
ax.set_ylabel('Synapse Density')
ax.set_xlabel('Distance (in um)')    

plt.savefig('axon_dendrite_split_analysis/plots/splittable.eps', format='eps')

# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(unsplittable_inputs, color = 'royalblue', ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(unsplittable_outputs, color = 'crimson', ax = ax, hist = False, kde_kws = {'shade': True})

ax.set(xlim = (-125000, 200000))
ax.set_yticks([])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150, 200])
ax.set_ylabel('Synapse Density')    
ax.set_xlabel('Distance (in um)')    

plt.savefig('axon_dendrite_split_analysis/plots/unsplittable.eps', format='eps')


# %%
print(unsplittable_inputs)

# %%
