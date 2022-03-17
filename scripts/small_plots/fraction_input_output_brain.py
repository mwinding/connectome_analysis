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


# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

brain = pymaid.get_skids_by_annotation('mw brain neurons')
sensories = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain inputs and ascending').name]
sensories = [x for sublist in sensories for x in sublist]
sensories = sensories + pymaid.get_skids_by_annotation('mw A1 ascending unknown')

outputs = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw brain outputs').name]
outputs = [x for sublist in outputs for x in sublist]

brain = list(np.setdiff1d(brain, outputs))
# %%
# plot number brain inputs, interneurons, outputs

fig, ax = plt.subplots(1,1,figsize=(2.5,5))
sns.barplot(x=['Inputs', 'Interneurons', 'Outputs'], y=[len(sensories), len(brain), len(outputs)], ax=ax)
plt.xticks(rotation=45, ha='right')
plt.savefig('small_plots/plots/general-neuron-counts.pdf', format='pdf', bbox_inches='tight')
# %%
# plot number brain inputs, interneurons, outputs

published = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('papers').name]
published = [x for sublist in published for x in sublist]

published_types = [len(np.intersect1d(sensories, published)), len(np.intersect1d(brain, published)), len(np.intersect1d(outputs, published))]
unpublished_types = [len(np.setdiff1d(sensories, published)), len(np.setdiff1d(brain, published)), len(np.setdiff1d(outputs, published))]

fractions = list(np.array(published_types)/(np.array(published_types)+np.array(unpublished_types))) + list(np.array(unpublished_types)/(np.array(published_types)+np.array(unpublished_types)))
col_width = 0.25

fig, ax = plt.subplots(1,1,figsize=(col_width*len(published_types),1.25))
sns.barplot(x=['Inputs', 'Interneurons', 'Outputs'], y=published_types, ax=ax, color='grey')
graph = sns.barplot(x=['Inputs', 'Interneurons', 'Outputs'], y=unpublished_types, bottom = published_types, ax=ax, color = sns.color_palette()[1])
plt.xticks(rotation=45, ha='right')

i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 5, round(fractions[i], 2), ha="center")
    i += 1

plt.savefig('small_plots/plots/percent_published.pdf', format='pdf', bbox_inches='tight')

# %%
