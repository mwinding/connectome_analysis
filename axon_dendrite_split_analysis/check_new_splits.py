# %%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

# %%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pandas as pd
import numpy as np
import connectome_tools.process_matrix as promat
import connectome_tools.process_skeletons as proskel
import networkx as nx 
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)

# getting required skids
unsplittables = pymaid.get_skids_by_annotation('mw mixed axon/dendrite')
unsplittables_2 = pymaid.get_skids_by_annotation('mw mixed axon/dendrite <0.2 segregation index')
unsplittables_2_skids = np.setdiff1d(unsplittables_2, unsplittables) # old unsplittables that have now been split

# %%
# distances of each synapse to split point, in list of pandas DataFrames for each neuron
connectdists_list = proskel.dist_from_split(unsplittables_2_skids, 'mw axon split')
connectdists = pd.concat(connectdists_list, axis=0) # all connectors combined

splittable_inputs = connectdists[connectdists['type'] == 'postsynaptic']['distance']
splittable_outputs = connectdists[connectdists['type'] == 'presynaptic']['distance']

#%%
# plotting as histogram distribution

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(splittable_inputs, ax = ax, hist = False, kde_kws = {'shade': True})
sns.distplot(splittable_outputs, ax = ax, hist = False, kde_kws = {'shade': True})
#sns.distplot(splittable_inputs, ax = ax)
#sns.distplot(splittable_outputs, ax = ax)

ax.set(xlim = (-125000, 200000))
plt.axvline(x=0, color = 'gray')
ax.set_yticks([])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150, 200])
ax.set_ylabel('Synapse Density')
ax.set_xlabel('Distance (in um)')    

plt.savefig('axon_dendrite_split_analysis/plots/new_splittables_less_than020_segIndex.pdf', format='pdf', bbox_inches = 'tight')

# %%
# plotting as raster plot
import matplotlib as ml

inputs_array = []
outputs_array = []
inputs_norm_array = []
outputs_norm_array = []

for i in range(len(connectdists_list)):
    connectdist = connectdists_list[i]
    output_index = connectdist['type']=='presynaptic'
    input_index = connectdist['type']=='postsynaptic'

    outputs = np.array(connectdist[output_index]['distance'])
    inputs = np.array(connectdist[input_index]['distance'])

    both = np.concatenate([outputs, inputs])
    min_both = min(both)
    max_both = max(both)

    norm_outputs = outputs
    norm_inputs = inputs

    for j in range(len(outputs)):
        if(outputs[j]>=0):
            norm_outputs[j] = norm_outputs[j]/max_both
        if(outputs[j]<0):
            norm_outputs[j] = -norm_outputs[j]/min_both

    for j in range(len(inputs)):
        if(inputs[j]>=0):
            norm_inputs[j] = norm_inputs[j]/max_both
        if(inputs[j]<0):
            norm_inputs[j] = -norm_inputs[j]/min_both

    inputs_array.append(inputs)
    outputs_array.append(outputs)
    inputs_norm_array.append(norm_inputs)
    outputs_norm_array.append(norm_outputs)

lineoffsets = np.arange(0, len(inputs_array), 1)


fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.set(xlim = (-1, 1))
ax.eventplot(inputs_norm_array, lineoffsets = lineoffsets, alpha = 0.5)
ax.eventplot(outputs_norm_array, lineoffsets = lineoffsets, color = 'orange', alpha = 0.5)

plt.savefig('axon_dendrite_split_analysis/plots/new_splittables_less_than020_rasterPlot.pdf', format='pdf', bbox_inches = 'tight')

# %%
