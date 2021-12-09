#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm


#%%
# cascades from each sensory modality
# load previously generated cascades

import pickle

p = 0.05
hops = 8
n_init = 1000

input_hit_hist_list = pickle.load(open('data/cascades/sensory-modality-cascades_1000-n_init.p', 'rb'))

# %%
# sort by dVNCs

pairs = pm.Promat.get_pairs()
dVNC_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs)
dSEZ_pairs = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs)
RGN_pairs = pm.Promat.load_pairs_from_annotation('mw RGN', pairs)

# build hit_hists per dVNC pair (averaged hits)
dVNC_inputs_list = []
for hit_hist in input_hit_hist_list:
    dVNC_inputs = []
    df = hit_hist.skid_hit_hist
    for ind in dVNC_pairs.index:
        pair = list(dVNC_pairs.loc[ind].values)
        if((pair[0] in df.index)&(pair[1] in df.index)):
            pair_data = (df.loc[pair].sum(axis=0)/2).values
            dVNC_inputs.append([pair[0]] + list(pair_data))

    dVNC_inputs_list.append(dVNC_inputs)

# build list of DataFrames
dVNC_inputs_df_list = []
for dVNC_inputs in dVNC_inputs_list:
    df = pd.DataFrame(dVNC_inputs, columns=['pair_id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df.set_index('pair_id', drop=True, inplace=True)
    dVNC_inputs_df_list.append(df)

input_hit_hist_list[9].name='mechano-II_III' # fix issue with generating path with "/"

# plot data
for sens_i, hit_hist_df in enumerate(dVNC_inputs_df_list):
    data = []
    for j in hit_hist_df.columns:
        for i in hit_hist_df.index:
            data.append([j, hit_hist_df.loc[i, j]])

    data = pd.DataFrame(data, columns = ['hops', 'hits'])

    # plot
    fig, axs = plt.subplots(2,1,figsize=(1, 1.5))
    ax = axs[0]
    ax.set(ylim=(0,1000))
    ax.get_xaxis().set_visible(False)
    sns.stripplot(x = data.hops, y=data.hits, s=0.75, alpha=0.5, ax=ax)
    
    ax = axs[1]
    sort_threshold = n_init/10
    hit_hist_df_sort = hit_hist_df.copy()
    hit_hist_df_sort[hit_hist_df_sort<sort_threshold]=0
    index = hit_hist_df_sort.sort_values(by=[1,2,3,4,5,6,7,8,9,10], ascending=False).index
    sns.heatmap(hit_hist_df.loc[index], cmap='Blues', ax=ax, cbar=False, vmax=500)
    plt.savefig(f'cascades/plots/dVNC-hits-per-hops_{input_hit_hist_list[sens_i].name}.pdf', format='pdf', bbox_inches='tight')

# %%
# number of hop layers from sensory modalities

hit_thres = n_init/10

multilayered_df = []
for i, dVNC_inputs_df in enumerate(dVNC_inputs_df_list):
    number_layers = (dVNC_inputs_df>hit_thres).sum(axis=1)
    add = list(zip(dVNC_inputs_df.index, (dVNC_inputs_df>hit_thres).sum(axis=1), [input_hit_hist_list[i].name]*len(dVNC_inputs_df.index)))
    multilayered_df.append(add)

multilayered_df = [x for sublist in multilayered_df for x in sublist]
multilayered_df = pd.DataFrame(multilayered_df, columns = ['skid', 'layers', 'sens'])

binwidth = 1
bins = np.arange(min(multilayered_df.layers), max(multilayered_df.layers) + binwidth*1.5) - binwidth*0.5

figsize = (.75, 0.35)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.histplot(x=multilayered_df.layers, bins=bins, stat='density')
plt.xticks(rotation=45, ha='right')
ax.set(xlim=(-0.75, 6.75), ylim=(0,0.55))
plt.savefig(f'cascades/plots/dVNC-single-cell-layers.pdf', format='pdf', bbox_inches='tight')


# %%
# same plot for dSEZs and RGNs

# dSEZs
# build hit_hists per dSEZ pair (averaged hits)
dSEZ_inputs_list = []
for hit_hist in input_hit_hist_list:
    dSEZ_inputs = []
    df = hit_hist.skid_hit_hist
    for ind in dSEZ_pairs.index:
        pair = list(dSEZ_pairs.loc[ind].values)
        if((pair[0] in df.index)&(pair[1] in df.index)):
            pair_data = (df.loc[pair].sum(axis=0)/2).values
            dSEZ_inputs.append([pair[0]] + list(pair_data))

    dSEZ_inputs_list.append(dSEZ_inputs)

# build list of DataFrames
dSEZ_inputs_df_list = []
for dSEZ_inputs in dSEZ_inputs_list:
    df = pd.DataFrame(dSEZ_inputs, columns=['pair_id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df.set_index('pair_id', drop=True, inplace=True)
    dSEZ_inputs_df_list.append(df)

# RGNs
# build hit_hists per dSEZ pair (averaged hits)
RGN_inputs_list = []
for hit_hist in input_hit_hist_list:
    RGN_inputs = []
    df = hit_hist.skid_hit_hist
    for ind in RGN_pairs.index:
        pair = list(RGN_pairs.loc[ind].values)
        if((pair[0] in df.index)&(pair[1] in df.index)):
            pair_data = (df.loc[pair].sum(axis=0)/2).values
            RGN_inputs.append([pair[0]] + list(pair_data))

    RGN_inputs_list.append(RGN_inputs)

# build list of DataFrames
RGN_inputs_df_list = []
for RGN_inputs in RGN_inputs_list:
    df = pd.DataFrame(RGN_inputs, columns=['pair_id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df.set_index('pair_id', drop=True, inplace=True)
    RGN_inputs_df_list.append(df)

# number of hop layers from sensory modalities

hit_thres = n_init/10

# dSEZ
multilayered_df = []
for i, dSEZ_inputs_df in enumerate(dSEZ_inputs_df_list):
    number_layers = (dSEZ_inputs_df>hit_thres).sum(axis=1)
    add = list(zip(dSEZ_inputs_df.index, (dSEZ_inputs_df>hit_thres).sum(axis=1), [input_hit_hist_list[i].name]*len(dSEZ_inputs_df.index)))
    multilayered_df.append(add)

multilayered_df = [x for sublist in multilayered_df for x in sublist]
multilayered_df = pd.DataFrame(multilayered_df, columns = ['skid', 'layers', 'sens'])

binwidth = 1
bins = np.arange(min(multilayered_df.layers), max(multilayered_df.layers) + binwidth*1.5) - binwidth*0.5

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.histplot(x=multilayered_df.layers, bins=bins, stat='probability')
plt.xticks(rotation=45, ha='right')
ax.set(xlim=(-0.75, 6.75), ylim=(0,0.55))
plt.savefig(f'cascades/plots/dSEZ-single-cell-layers.pdf', format='pdf', bbox_inches='tight')


# RGN
multilayered_df = []
for i, RGN_inputs_df in enumerate(RGN_inputs_df_list):
    number_layers = (RGN_inputs_df>hit_thres).sum(axis=1)
    add = list(zip(RGN_inputs_df.index, (RGN_inputs_df>hit_thres).sum(axis=1), [input_hit_hist_list[i].name]*len(RGN_inputs_df.index)))
    multilayered_df.append(add)

multilayered_df = [x for sublist in multilayered_df for x in sublist]
multilayered_df = pd.DataFrame(multilayered_df, columns = ['skid', 'layers', 'sens'])

binwidth = 1
bins = np.arange(min(multilayered_df.layers), max(multilayered_df.layers) + binwidth*1.5) - binwidth*0.5

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.histplot(x=multilayered_df.layers, bins=bins, stat='probability')
plt.xticks(rotation=45, ha='right')
ax.set(xlim=(-0.75, 6.75), ylim=(0,0.55))
plt.savefig(f'cascades/plots/RGN-single-cell-layers.pdf', format='pdf', bbox_inches='tight')

# %%
