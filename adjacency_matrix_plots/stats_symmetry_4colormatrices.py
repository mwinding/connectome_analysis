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

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

aa = pd.read_csv('data/axon-axon.csv', header = 0, index_col = 0)
ad = pd.read_csv('data/axon-dendrite.csv', header = 0, index_col = 0)
dd = pd.read_csv('data/dendrite-dendrite.csv', header = 0, index_col = 0)
da = pd.read_csv('data/dendrite-axon.csv', header = 0, index_col = 0)

#%%
threshold_aa = pd.read_csv('data/threshold_sweep_csvs/threshold-synapses-Gaa-min-dend0-rem-pdiffTrue-subsetFalse.csv', header = 0, index_col = 0)
threshold_ad = pd.read_csv('data/threshold_sweep_csvs/threshold-synapses-Gad-min-dend0-rem-pdiffTrue-subsetFalse.csv', header = 0, index_col = 0)
threshold_dd = pd.read_csv('data/threshold_sweep_csvs/threshold-synapses-Gdd-min-dend0-rem-pdiffTrue-subsetFalse.csv', header = 0, index_col = 0)
threshold_da = pd.read_csv('data/threshold_sweep_csvs/threshold-synapses-Gda-min-dend0-rem-pdiffTrue-subsetFalse.csv', header = 0, index_col = 0)
threshold_G = pd.read_csv('data/threshold_sweep_csvs/threshold-synapses-G-min-dend0-rem-pdiffTrue-subsetFalse.csv', header = 0, index_col = 0)

# %%
# symmetry between paired edges and % edges / synapses left for each threshold
marker_size = 5

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(2.75, 1.4))

sns.scatterplot(x = threshold_G.index+1 , y = threshold_G['Prop. edges left'], ax = ax, color = sns.color_palette()[0], markers=True, linewidth = 1, s = marker_size, alpha = 0.5, edgecolor="none")
sns.lineplot(x = threshold_G.index+1 , y = threshold_G['Prop. edges left'], ax = ax, color = sns.color_palette()[0], linewidth = 1, alpha = 0.3)

sns.scatterplot(x = threshold_G.index+1 , y = threshold_G['Prop. synapses left'], ax = ax, color = sns.color_palette()[4], markers=True, linewidth = 1, s = marker_size, alpha = 0.8, edgecolor="none")
sns.lineplot(x = threshold_G.index+1 , y = threshold_G['Prop. synapses left'], ax = ax, color = sns.color_palette()[4], linewidth = 1, alpha = 0.5)

sns.scatterplot(x = threshold_G.index+1 , y = threshold_G['Prop. paired edges symmetric'], ax = ax, color = sns.color_palette()[2], markers=True, linewidth = 1, s = marker_size, alpha = 0.8, edgecolor="none")
sns.lineplot(x = threshold_G.index+1 , y = threshold_G['Prop. paired edges symmetric'], ax = ax, color = sns.color_palette()[2], linewidth = 1, alpha = 0.5)

ax.set(xticks=[1, 5, 10, 15, 20])

ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('', fontname="Arial", fontsize = 6)
ax.set_xlabel('Synapse Threshold', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")


plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/symmetric_edges_summedgraph.pdf', format='pdf', bbox_inches='tight')
# %%
# symmetry between pairs by 4-color graphs

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(1,1.25))

sns.lineplot(x = threshold_dd.index+1 , y = threshold_dd['Prop. paired edges symmetric'], ax = ax, color = sns.color_palette()[2], linewidth = 0.5)
sns.lineplot(x = threshold_da.index+1 , y = threshold_da['Prop. paired edges symmetric'], ax = ax, color = sns.color_palette()[3], linewidth = 0.5)
sns.lineplot(x = threshold_aa.index+1 , y = threshold_aa['Prop. paired edges symmetric'], ax = ax, color = sns.color_palette()[1], linewidth = 0.5)
sns.lineplot(x = threshold_ad.index+1 , y = threshold_ad['Prop. paired edges symmetric'], ax = ax, color = sns.color_palette()[0], linewidth = 0.5)

#plt.axvline(x= np.where(threshold_ad['Prop. paired edges symmetric']>= 0.9)[0][0], color = sns.color_palette()[0], linewidth = 0.5)
#plt.axvline(x= np.where(threshold_aa['Prop. paired edges symmetric']>= 0.9)[0][0], color = sns.color_palette()[1], linewidth = 0.5)
#plt.axvline(x= np.where(threshold_dd['Prop. paired edges symmetric']>= 0.9)[0][0], color = sns.color_palette()[2], linewidth = 0.5)
#plt.axvline(x= np.where(threshold_da['Prop. paired edges symmetric']>= 0.9)[0][0], color = sns.color_palette()[3], linewidth = 0.5)

ax.set(xticks=[1, 5, 10, 15, 20])

ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Fraction of Symmetric Edges', fontname="Arial", fontsize = 6)
ax.set_xlabel('Synapse Threshold', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/symmetric_edges.pdf', format='pdf', bbox_inches='tight')
# %%
# cumulative distribution of connections by number of synapses
fig, ax = plt.subplots(1,1,figsize=(1,1))

sns.lineplot(x = threshold_ad.index , y = 1-threshold_ad['Prop. synapses left'], ax = ax, color = sns.color_palette()[0], linewidth=0.5)
sns.lineplot(x = threshold_aa.index , y = 1-threshold_aa['Prop. synapses left'], ax = ax, color = sns.color_palette()[1], linewidth=0.5)
sns.lineplot(x = threshold_dd.index , y = 1-threshold_dd['Prop. synapses left'], ax = ax, color = sns.color_palette()[2], linewidth=0.5)
sns.lineplot(x = threshold_da.index , y = 1-threshold_da['Prop. synapses left'], ax = ax, color = sns.color_palette()[3], linewidth=0.5)

ax.set(xticks=np.arange(0,21,5))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Cumulative Fraction of Matrix', fontname="Arial", fontsize = 6)
ax.set_xlabel('Connection Strength\n(Number of Synapses)', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/cumuldist_synapses.pdf', format='pdf', bbox_inches='tight')

# %%
# cumulative distribution of edges by number of synapses
fig, ax = plt.subplots(1,1,figsize=(1,1))

sns.lineplot(x = threshold_ad.index , y = 1-threshold_ad['Prop. edges left'], ax = ax, color = sns.color_palette()[0], linewidth = 0.5)
sns.lineplot(x = threshold_aa.index , y = 1-threshold_aa['Prop. edges left'], ax = ax, color = sns.color_palette()[1], linewidth = 0.5)
sns.lineplot(x = threshold_dd.index , y = 1-threshold_dd['Prop. edges left'], ax = ax, color = sns.color_palette()[2], linewidth = 0.5)
sns.lineplot(x = threshold_da.index , y = 1-threshold_da['Prop. edges left'], ax = ax, color = sns.color_palette()[3], linewidth = 0.5)

ax.set(xticks=np.arange(0,21,5))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Cumulative Fraction of Matrix', fontname="Arial", fontsize = 6)
ax.set_xlabel('Connection Strength\n(Number of Synapses)', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/cumuldist_edges.pdf', format='pdf', bbox_inches='tight')

# %%
from tqdm import tqdm

def syn_contrib (matrix):
    mat = np.matrix(matrix)
    total_syn = mat.sum(axis = None) # total synapses

    syn_contrib = []
    for i in tqdm(range(0, 100)):
        contrib = mat[mat==i].sum(axis = None)/total_syn
        syn_contrib.append(contrib)

    return(syn_contrib)


ad_syn_contrib = syn_contrib(ad)
aa_syn_contrib = syn_contrib(aa)
dd_syn_contrib = syn_contrib(dd)
da_syn_contrib = syn_contrib(da)


# %%
# contribution of connections by number of synapses
fig, ax = plt.subplots(1,1,figsize=(1,1))
max_x = 6

sns.scatterplot(x = np.arange(1, max_x, 1), y = dd_syn_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[2], s = 5, alpha = 0.8)
sns.scatterplot(x = np.arange(1, max_x, 1), y = da_syn_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[3], s = 5, alpha = 0.8)
sns.scatterplot(x = np.arange(1, max_x, 1), y = aa_syn_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[1], s = 5, alpha = 0.8)
sns.scatterplot(x = np.arange(1, max_x, 1), y = ad_syn_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[0], s = 5, alpha = 0.8)


ax.set(xticks=[1, 2, 3, 4, 5])
ax.set(yticks=[0, 0.1, 0.2, 0.3, 0.4])
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Fraction of Synapses', fontname="Arial", fontsize = 6)
ax.set_xlabel('Connection Strength\n(Number of Synapses)', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/contrib_synapses.pdf', format='pdf', bbox_inches='tight')

# %%
from tqdm import tqdm

def edge_contrib (matrix):
    total_edge = np.matrix(matrix > 0).sum(axis = None) # total edges

    mat = np.matrix(matrix)

    edge_contrib = []
    edge_contrib.append(0.0)
    for i in tqdm(range(1, 100)):
        contrib = np.matrix(mat==i).sum(axis=None)/total_edge
        edge_contrib.append(contrib)

    return(edge_contrib)

ad_edge_contrib = edge_contrib(ad)
aa_edge_contrib = edge_contrib(aa)
dd_edge_contrib = edge_contrib(dd)
da_edge_contrib = edge_contrib(da)
# %%
# contribution of edges by number of synapses
fig, ax = plt.subplots(1,1,figsize=(1,1))
max_x = 6

sns.scatterplot(x = np.arange(1, max_x, 1), y = dd_edge_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[2], s = 5, alpha = 0.8)
sns.scatterplot(x = np.arange(1, max_x, 1), y = da_edge_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[3], s = 5, alpha = 0.8)
sns.scatterplot(x = np.arange(1, max_x, 1), y = aa_edge_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[1], s = 5, alpha = 0.8)
sns.scatterplot(x = np.arange(1, max_x, 1), y = ad_edge_contrib[1:max_x], ax = ax, linewidth = 0, color = sns.color_palette()[0], s = 5, alpha = 0.8)


ax.set(xticks=np.arange(1, max_x, 1))
ax.set(yticks=np.arange(0, 1, .2))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Fraction of Edges', fontname="Arial", fontsize = 6)
ax.set_xlabel('Connection Strength\n(Number of Synapses)', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/contrib_edges.pdf', format='pdf', bbox_inches='tight')

# %%
