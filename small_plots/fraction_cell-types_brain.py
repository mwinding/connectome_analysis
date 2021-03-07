#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import connectome_tools.process_matrix as pm

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

brain = pymaid.get_skids_by_annotation('mw brain neurons')

# inputs
input_names = pymaid.get_annotated('mw brain inputs and ascending').name
input_names_formatted = ['ORN', 'thermo', 'photo', 'AN-sens', 'MN-sens', 'vtd', 'proprio', 'mechano', 'class II_III', 'noci', 'unknown']
inputs = [pymaid.get_skids_by_annotation(x) for x in input_names]
inputs = inputs + [pymaid.get_skids_by_annotation('mw A1 ascending unknown')]

outputs = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain outputs').name]
outputs_names_formatted = ['dSEZs', 'dVNCs', 'RGNs']

outputs_names_formatted = [outputs_names_formatted[i] for i in [2,0,1]]
outputs = [outputs[i] for i in [2,0,1]]

outputs_all = [x for sublist in outputs for x in sublist]

brain = list(np.setdiff1d(brain, outputs_all))

all_celltypes, celltype_names = pm.Promat.celltypes()
# %%
# plot number brain inputs, interneurons, outputs

col_width = 0.15
plot_height = 0.75
# general cell types
colors = ['#1D79B7', '#5D8C90', '#D4E29E', '#FF8734', '#E55560', '#F9EB4D', '#C144BC']
fig, ax = plt.subplots(1,1,figsize=(col_width*len(celltype_names[1:8]), plot_height))
graph = sns.barplot(x=celltype_names[1:8], y=[len(skids) for skids in all_celltypes[1:8]], ax=ax, palette = colors)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5,
        [len(skids) for skids in all_celltypes[1:8]][i], ha="center", color=colors[i])
    i += 1
ax.set(ylim=(0, 230))

plt.savefig('small_plots/plots/general-celltype-counts.pdf', format='pdf', bbox_inches='tight')

# inputs
fig, ax = plt.subplots(1,1,figsize=(col_width*len(inputs),plot_height))
graph = sns.barplot(x=input_names_formatted, y=[len(skids) for skids in inputs], ax=ax)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5,
        [len(skids) for skids in inputs][i],ha="center")
    i += 1
ax.set(ylim=(0, 230))
plt.savefig('small_plots/plots/input-counts.pdf', format='pdf', bbox_inches='tight')

# outputs
colors = ['#9467BD','#D88052', '#A52A2A']
fig, ax = plt.subplots(1,1,figsize=(col_width*len(outputs),plot_height))
graph = sns.barplot(x=outputs_names_formatted, y=[len(skids) for skids in outputs], ax=ax, palette=colors)
plt.xticks(rotation=45, ha='right')
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5,
        [len(skids) for skids in outputs][i],ha="center", color=colors[i])
    i += 1
ax.set(ylim=(0, 230))
plt.savefig('small_plots/plots/output-counts.pdf', format='pdf', bbox_inches='tight')

# %%
