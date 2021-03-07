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

# %%
rm = pymaid.CatmaidInstance(url, token, name, password)

skids = pymaid.get_skids_by_annotation('mw brain neurons')
unsplittable = pymaid.get_skids_by_annotation('mw mixed axon/dendrite')
unsplittable = pymaid.get_skids_by_annotation('mw mixed axon/dendrite')
immature = pymaid.get_skids_by_annotation('mw brain few synapses')
brain_skids_split = np.setdiff1d(skids, unsplittable) 
brain_skids_split = np.setdiff1d(brain_skids_split, immature) # only neurons with split tags

# %%
# distances of each synapse to split point, in list of pandas DataFrames for each neuron
connectdists_list = proskel.dist_from_split(brain_skids_split, 'mw axon split')
connectdists = pd.concat(connectdists_list, axis=0) # all connectors combined

splittable_inputs = connectdists[connectdists['type'] == 'postsynaptic']['distance']
splittable_outputs = connectdists[connectdists['type'] == 'presynaptic']['distance']

connectdists.to_csv('axon_dendrite_split_analysis/plots/connectdists_wholebrain.csv')

# %%
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

plt.savefig('axon_dendrite_split_analysis/plots/all_splittables_brain_hists.pdf', format='pdf', bbox_inches = 'tight')

# %%
# preparing data for plotting
# normalizing and sorting by individual neuron
import matplotlib as ml

inputs_array = []
outputs_array = []
inputs_norm_array = []
outputs_norm_array = []

connectdists = pd.read_csv('axon_dendrite_split_analysis/plots/connectdists_wholebrain.csv')
all_skids = np.unique(connectdists['skeletonid'])

connectdists_list = []
for i in range(len(all_skids)):
    index = connectdists['skeletonid'] == all_skids[i]
    skid_connectors = connectdists[index]
    connectdists_list.append(skid_connectors)
    
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

# %%
# plot normalized synapse raster plot
fig, ax = plt.subplots(1,1,figsize=(5,10))
ax.set(xlim = (-1, 1))
ax.eventplot(inputs_norm_array, lineoffsets = lineoffsets, alpha = 0.5)
ax.eventplot(outputs_norm_array, lineoffsets = lineoffsets, color = sns.color_palette()[1], alpha = 0.5)

plt.savefig('axon_dendrite_split_analysis/plots/all_splittables_brain_rasterPlot.pdf', format='pdf', bbox_inches = 'tight')


# %%
# plot raw synapse raster plot
# sorted by neuron size
# doesn't look as good...

connectdists = pd.read_csv('axon_dendrite_split_analysis/plots/connectdists_wholebrain.csv')
all_skids = np.unique(connectdists['skeletonid'])

connectdists_list = []
min_dist = []
max_dist = []
size = []
for i in range(len(all_skids)):
    index = connectdists['skeletonid'] == all_skids[i]
    skid_connectors = connectdists[index]
    connectdists_list.append(skid_connectors)
    min_dist.append(np.min(skid_connectors['distance']))
    max_dist.append(np.max(skid_connectors['distance']))
    size.append(-np.min(skid_connectors['distance']) + np.max(skid_connectors['distance']))

inputs_array = []
outputs_array = []
for i in range(len(connectdists_list)):
    connectdist = connectdists_list[i]
    output_index = connectdist['type']=='presynaptic'
    input_index = connectdist['type']=='postsynaptic'

    outputs = np.array(connectdist[output_index]['distance'])
    inputs = np.array(connectdist[input_index]['distance'])

    inputs_array.append(inputs)
    outputs_array.append(outputs)

lineoffsets = np.arange(0, len(inputs_array), 1)

sort_input = sorted(zip(size, inputs_array))
sort_output = sorted(zip(size, outputs_array))
size_sorted, inputs_array_sorted = zip(*sort_input)
size_sorted, outputs_array_sorted = zip(*sort_output)


# plot synapse raster plot
fig, ax = plt.subplots(1,1,figsize=(5,10))
ax.eventplot(inputs_array_sorted, lineoffsets = lineoffsets, alpha = 0.5)
ax.eventplot(outputs_array_sorted, lineoffsets = lineoffsets, color = sns.color_palette()[1], alpha = 0.5)

plt.savefig('axon_dendrite_split_analysis/plots/all_splittables_brain_rasterPlot_sorted.pdf', format='pdf', bbox_inches = 'tight')

# %%
# plotting as normalized histogram distribution

all_inputs_norm = np.concatenate(inputs_norm_array)
all_outputs_norm = np.concatenate(outputs_norm_array)

fig, ax = plt.subplots(1,1,figsize=(1,.75))

# parameters like font, axis width, etc
ax.set(xticks=[-1, 0, 1])
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Density', fontname="Arial", fontsize = 6)
ax.set_xlabel('Distance from Split Point', fontname="Arial", fontsize = 6)

ax.set(xlim = (-1, 1))

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

sns.distplot(all_inputs_norm, ax = ax, hist = False, kde = True, kde_kws=dict(linewidth=0.5, shade = True), norm_hist=True)
sns.distplot(all_outputs_norm, ax = ax, hist = False, kde = True, kde_kws=dict(linewidth=0.5, shade = True), norm_hist=True)

plt.savefig('axon_dendrite_split_analysis/plots/all_splittables_brain_hist_norm.pdf', format='pdf', bbox_inches = 'tight')

# %%
