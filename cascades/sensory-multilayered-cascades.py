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
# pull sensory annotations and then pull associated skids
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [ct.Celltype(name, ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

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

# plot data
for sens_i, hit_hist_df in enumerate(dVNC_inputs_df_list):
    data = []
    for j in hit_hist_df.columns:
        for i in hit_hist_df.index:
            data.append([j, hit_hist_df.loc[i, j]])

    data = pd.DataFrame(data, columns = ['hops', 'hits'])

    # plot
    fig, axs = plt.subplots(2,1,figsize=(2,2))
    ax = axs[0]
    ax.set(ylim=(0,1000))
    sns.stripplot(x = data.hops, y=data.hits, s=1, alpha=0.5, ax=ax)
    
    ax = axs[1]
    sns.heatmap(hit_hist_df, cmap='Blues', ax=ax, cbar=False)
    plt.savefig(f'cascades/plots/hits-per-hops_{input_hit_hist_list[sens_i].name}.pdf', format='pdf', bbox_inches='tight')

