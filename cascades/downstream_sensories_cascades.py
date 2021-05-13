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

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')

#%%
# pull sensory annotations and then pull associated skids
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
sens = [ct.Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

#%%
# cascades from each sensory modality

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
adj=adj_ad

input_hit_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = order, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

# %%
# plot sensory cascades raw

fig, axs = plt.subplots(len(input_hit_hist_list), 1, figsize=(10, 20))
fig.tight_layout(pad=2.0)

for i, hit_hist in enumerate(input_hit_hist_list):
    ax = axs[i]
    sns.heatmap(hit_hist.skid_hit_hist, ax=ax)
    ax.set_xlabel(hit_hist.get_name())

plt.savefig('cascades/plots/sensory_modality_signals.pdf', format='pdf', bbox_inches='tight')
os.system('say "code executed"')

# %%
# how close are descending neurons to sensory?

# load output types
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RGN')

# generate Cascade_Analyzer objects containing name of pathway and the hit_hist to each output type
dVNC_hits = [casc.Cascade_Analyzer(f'{hit_hist.get_name()}-dVNC', hit_hist.skid_hit_hist.loc[dVNC, :]) for hit_hist in input_hit_hist_list]
dSEZ_hits = [casc.Cascade_Analyzer(f'{hit_hist.get_name()}-dSEZ', hit_hist.skid_hit_hist.loc[dSEZ, :]) for hit_hist in input_hit_hist_list]
RGN_hits = [casc.Cascade_Analyzer(f'{hit_hist.get_name()}-RGN', hit_hist.skid_hit_hist.loc[RGN, :]) for hit_hist in input_hit_hist_list]

dVNC_hits = [casc.Cascade_Analyzer([hit_hist.get_name(), 'dVNC'], hit_hist.skid_hit_hist.loc[dVNC, :]) for hit_hist in input_hit_hist_list]
dSEZ_hits = [casc.Cascade_Analyzer([hit_hist.get_name(), 'dSEZ'], hit_hist.skid_hit_hist.loc[dSEZ, :]) for hit_hist in input_hit_hist_list]
RGN_hits = [casc.Cascade_Analyzer([hit_hist.get_name(), 'RGN'], hit_hist.skid_hit_hist.loc[RGN, :]) for hit_hist in input_hit_hist_list]

# max possible hits that all output neuron types could receive 
max_dVNC_hits = len(dVNC_hits[0].skid_hit_hist.index)*n_init
max_dSEZ_hits = len(dVNC_hits[0].skid_hit_hist.index)*n_init
max_RGN_hits = len(dVNC_hits[0].skid_hit_hist.index)*n_init

# organize data so that each sens -> dVNC, dSEZ, RGN is intercalated
sens_output_data = list(zip(dVNC_hits, dSEZ_hits, RGN_hits))
sens_output_data = [x for sublist in sens_output_data for x in sublist]
sens_output_df = pd.DataFrame([x.skid_hit_hist.sum(axis=0) for x in sens_output_data])

# set up multiindex
sens_output_df['source']=[x.get_name()[0] for x in sens_output_data]
sens_output_df['target']=[x.get_name()[1] for x in sens_output_data]
sens_output_df = sens_output_df.set_index(['source', 'target'])

# normalize by max possible input to each output type (num neurons * n_init)
sens_output_df_plot = sens_output_df.copy()
sens_output_df_plot.loc[(slice(None), 'dVNC'), :] = sens_output_df_plot.loc[(slice(None), 'dVNC'), :]/max_dVNC_hits
sens_output_df_plot.loc[(slice(None), 'dSEZ'), :] = sens_output_df_plot.loc[(slice(None), 'dSEZ'), :]/max_dSEZ_hits
sens_output_df_plot.loc[(slice(None), 'RGN'), :] = sens_output_df_plot.loc[(slice(None), 'RGN'), :]/max_RGN_hits

import cmasher as cmr
fig, ax = plt.subplots(1, 1, figsize=(1.5, 2))
fig.tight_layout(pad=3.0)
vmax = 0.35
cmap = cmr.torch

sns.heatmap(sens_output_df_plot, ax = ax, cmap = cmap, vmax=vmax)
ax.set_title('Signal to brain outputs')
ax.set(xlim = (0, 11))
plt.savefig('cascades/plots/sensory_modality_signals_to_output.pdf', format='pdf', bbox_inches='tight')

# determine mean/median hop distance from sens -> output
def counts_to_list(count_list):
    expanded_counts = []
    for i, count in enumerate(count_list):
        expanded = np.repeat(i, count)
        expanded_counts.append(expanded)
    
    return([x for sublist in expanded_counts for x in sublist])

all_sens_output_dist = []
for row in sens_output_df.iterrows():
    list_hits = counts_to_list(row[1])
    all_sens_output_dist.append([row[0][0], row[0][1], np.mean(list_hits), np.median(list_hits)])

all_sens_output_dist = pd.DataFrame(all_sens_output_dist, columns = ['source', 'target', 'mean_hop', 'median_hop'])
# %%
# plotting visits by modality to each descending to VNC neuron pair 
# supplemental figure
# *** CONTINUE WORKING HERE

dVNC_hits_summed = [pd.DataFrame(x.skid_hit_hist.iloc[:, 0:8].sum(axis=1), columns=[x.get_name()[0]]) for x in dVNC_hits]
dVNC_hits_summed = pd.concat(dVNC_hits_summed, axis=1)
dVNC_hits_pairwise = pm.Promat.convert_df_to_pairwise(dVNC_hits_summed)

fig, axs = plt.subplots(
    3, 1, figsize=(8, 8)
)

fig.tight_layout(pad=3.0)

ax = axs[0]
ax.get_xaxis().set_visible(False)
ax.set_title('Signal to Individual VNC Descending Neurons')
sns.heatmap(dVNC_hits.iloc[:, 1:len(dVNC_hits)].sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False]).T, ax = ax, rasterized = True)

ax = axs[1]
ax.get_xaxis().set_visible(False)
ax.set_title('Signal to Individual SEZ Descending Neurons')
sns.heatmap(dSEZ_hits.iloc[:, 1:len(dSEZ_hits)].sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False]).T, ax = ax, rasterized = True)

ax = axs[2]
ax.set_xlabel('Individual Ring Gland Neurons')
ax.get_xaxis().set_visible(False)
ax.set_title('Signal to Individual Ring Gland Neurons')
sns.heatmap(RG_hits.iloc[:, 1:len(RG_hits)].sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False]).T, ax = ax, rasterized = True)

plt.savefig('cascades/plots/signal_to_individual_outputs.pdf', format='pdf', bbox_inches='tight')

#%%
# alternative clustermap plot of descending neurons
# supplemental figure plot

fig = sns.clustermap(dVNC_hits.iloc[:, 1:len(dVNC_hits)].T, row_cluster = False, figsize = (8, 4), rasterized = True)
ax = fig.ax_heatmap
ax.set_xlabel('Individual dVNCs')
ax.set_xticks([])
fig.savefig('cascades/plots/signal_to_individual_dVNCs.pdf')

fig = sns.clustermap(dSEZ_hits.iloc[:, 1:len(dVNC_hits)].T, row_cluster = False, figsize = (8, 4), rasterized = True)
ax = fig.ax_heatmap
ax.set_xlabel('Individual dSEZs')
ax.set_xticks([])
fig.savefig('cascades/plots/signal_to_individual_dSEZs.pdf')

fig = sns.clustermap(RG_hits.iloc[:, 1:len(dVNC_hits)].T, row_cluster = False, figsize = (8, 4), rasterized = True)
ax = fig.ax_heatmap
ax.set_xlabel('Individual RG neurons')
ax.set_xticks([])
fig.savefig('cascades/plots/signal_to_individual_RGs.pdf')

# %%
##########
# **** Note Well: REALLY old code below, deprecated or never used in paper ****
##########

# %%
# num of descendings at each level
# this assumes that thresholding per node is useful; it might not be

threshold = 50
num_dVNC_dsSens = pd.DataFrame(([np.array(dVNC_ORN_hit>threshold).sum(axis = 0),
                                np.array(dVNC_AN_hit>threshold).sum(axis = 0),
                                np.array(dVNC_MN_hit>threshold).sum(axis = 0),
                                np.array(dVNC_A00c_hit>threshold).sum(axis = 0),
                                np.array(dVNC_vtd_hit>threshold).sum(axis = 0),
                                np.array(dVNC_thermo_hit>threshold).sum(axis = 0),
                                np.array(dVNC_photo_hit>threshold).sum(axis = 0)]),
                                index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

num_dSEZ_dsSens = pd.DataFrame(([np.array(dSEZ_ORN_hit>threshold).sum(axis = 0),
                                np.array(dSEZ_AN_hit>threshold).sum(axis = 0),
                                np.array(dSEZ_MN_hit>threshold).sum(axis = 0),
                                np.array(dSEZ_A00c_hit>threshold).sum(axis = 0),
                                np.array(dSEZ_vtd_hit>threshold).sum(axis = 0),
                                np.array(dSEZ_thermo_hit>threshold).sum(axis = 0),
                                np.array(dSEZ_photo_hit>threshold).sum(axis = 0)]),
                                index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

num_RG_dsSens = pd.DataFrame(([np.array(RG_ORN_hit>threshold).sum(axis = 0),
                                np.array(RG_AN_hit>threshold).sum(axis = 0),
                                np.array(RG_MN_hit>threshold).sum(axis = 0),
                                np.array(RG_A00c_hit>threshold).sum(axis = 0),
                                np.array(RG_vtd_hit>threshold).sum(axis = 0),
                                np.array(RG_thermo_hit>threshold).sum(axis = 0),
                                np.array(RG_photo_hit>threshold).sum(axis = 0)]),
                                index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

fig, axs = plt.subplots(
    3, 1, figsize=(8, 8)
)

fig.tight_layout(pad=3.0)
vmax = 50
cmap = cmr.heat

ax = axs[0]
ax.set_title('Number of VNC Descending Neurons downstream of Sensory Signal')
sns.heatmap(num_dVNC_dsSens, ax = ax, vmax = vmax, rasterized=True, cmap = cmap)
ax.set(xlim = (0, 13))

ax = axs[1]
ax.set_title('Number of SEZ Descending Neurons downstream of Sensory Signal')
sns.heatmap(num_dSEZ_dsSens, ax = ax, vmax = vmax, rasterized=True, cmap = cmap)
ax.set(xlim = (0, 13))

ax = axs[2]
ax.set_title('Number of Ring Gland Neurons downstream of Sensory Signal')
sns.heatmap(num_RG_dsSens, ax = ax, vmax = vmax, rasterized=True, cmap = cmap)
ax.set_xlabel('Hops from sensory')
ax.set(xlim = (0, 13))

plt.savefig('cascades/plots/number_outputs_ds_each_sensory_modality.pdf', format='pdf', bbox_inches='tight')

# %%
# When modality are each outputs associated with?
dVNC_hits = pd.DataFrame(([ dVNC_skids, 
                            dVNC_ORN_hit.sum(axis = 1),
                            dVNC_AN_hit.sum(axis = 1),
                            dVNC_MN_hit.sum(axis = 1),
                            dVNC_thermo_hit.sum(axis = 1),
                            dVNC_photo_hit.sum(axis = 1),
                            dVNC_A00c_hit.sum(axis = 1),
                            dVNC_vtd_hit.sum(axis = 1)]),
                            index = ['dVNC_skid', 'ORN', 'AN', 'MN', 'thermo', 'photo', 'A00c', 'vtd'])
dVNC_hits = dVNC_hits.T

dSEZ_hits = pd.DataFrame(([ dSEZ_skids, 
                            dSEZ_ORN_hit.sum(axis = 1),
                            dSEZ_AN_hit.sum(axis = 1),
                            dSEZ_MN_hit.sum(axis = 1),
                            dSEZ_thermo_hit.sum(axis = 1),
                            dSEZ_photo_hit.sum(axis = 1),
                            dSEZ_A00c_hit.sum(axis = 1),
                            dSEZ_vtd_hit.sum(axis = 1)]),
                            index = ['dSEZ_skid', 'ORN', 'AN', 'MN', 'thermo', 'photo', 'A00c', 'vtd'])
dSEZ_hits = dSEZ_hits.T

RG_hits = pd.DataFrame(([ RG_skids, 
                            RG_ORN_hit.sum(axis = 1),
                            RG_AN_hit.sum(axis = 1),
                            RG_MN_hit.sum(axis = 1),
                            RG_thermo_hit.sum(axis = 1),
                            RG_photo_hit.sum(axis = 1),
                            RG_A00c_hit.sum(axis = 1),
                            RG_vtd_hit.sum(axis = 1)]),
                            index = ['RG_skid', 'ORN', 'AN', 'MN', 'thermo', 'photo', 'A00c', 'vtd'])
RG_hits = RG_hits.T

# %%
# sensory characterization of each layer of each sensory modality
import plotly.express as px
from pandas.plotting import parallel_coordinates

# replacement if I want to use this later
#sensory_profiles = [hit_hist.skid_hit_hist.sum(axis=1).values for hit_hist in input_hit_hist_list]
#sensory_profiles = pd.DataFrame(sensory_profiles, index=[hit_hist.get_name() for hit_hist in input_hit_hist_list], columns = input_hit_hist_list[0].skid_hit_hist.index)

sensory_profile = pd.DataFrame(([ORN_hit_hist.sum(axis = 1), 
                    AN_hit_hist.sum(axis = 1),
                    MN_hit_hist.sum(axis = 1), 
                    A00c_hit_hist.sum(axis = 1),
                    vtd_hit_hist.sum(axis = 1), 
                    thermo_hit_hist.sum(axis = 1),
                    photo_hit_hist.sum(axis = 1)]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile0 = pd.DataFrame(([ORN_hit_hist[:, 0], 
                    AN_hit_hist[:, 0],
                    MN_hit_hist[:, 0], 
                    A00c_hit_hist[:, 0],
                    vtd_hit_hist[:, 0], 
                    thermo_hit_hist[:, 0],
                    photo_hit_hist[:, 0]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile1 = pd.DataFrame(([ORN_hit_hist[:, 1], 
                    AN_hit_hist[:, 1],
                    MN_hit_hist[:, 1], 
                    A00c_hit_hist[:, 1],
                    vtd_hit_hist[:, 1], 
                    thermo_hit_hist[:, 1],
                    photo_hit_hist[:, 1]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile2 = pd.DataFrame(([ORN_hit_hist[:, 2], 
                    AN_hit_hist[:, 2],
                    MN_hit_hist[:, 2], 
                    A00c_hit_hist[:, 2],
                    vtd_hit_hist[:, 2], 
                    thermo_hit_hist[:, 2],
                    photo_hit_hist[:, 2]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile3 = pd.DataFrame(([ORN_hit_hist[:, 3], 
                    AN_hit_hist[:, 3],
                    MN_hit_hist[:, 3], 
                    A00c_hit_hist[:, 3],
                    vtd_hit_hist[:, 3], 
                    thermo_hit_hist[:, 3],
                    photo_hit_hist[:, 3]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile4 = pd.DataFrame(([ORN_hit_hist[:, 4], 
                    AN_hit_hist[:, 4],
                    MN_hit_hist[:, 4], 
                    A00c_hit_hist[:, 4],
                    vtd_hit_hist[:, 4], 
                    thermo_hit_hist[:, 4],
                    photo_hit_hist[:, 4]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile5 = pd.DataFrame(([ORN_hit_hist[:, 5], 
                    AN_hit_hist[:, 5],
                    MN_hit_hist[:, 5], 
                    A00c_hit_hist[:, 5],
                    vtd_hit_hist[:, 5], 
                    thermo_hit_hist[:, 5],
                    photo_hit_hist[:, 5]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile6 = pd.DataFrame(([ORN_hit_hist[:, 6], 
                    AN_hit_hist[:, 6],
                    MN_hit_hist[:, 6], 
                    A00c_hit_hist[:, 6],
                    vtd_hit_hist[:, 6], 
                    thermo_hit_hist[:, 6],
                    photo_hit_hist[:, 6]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile7 = pd.DataFrame(([ORN_hit_hist[:, 7], 
                    AN_hit_hist[:, 7],
                    MN_hit_hist[:, 7], 
                    A00c_hit_hist[:, 7],
                    vtd_hit_hist[:, 7], 
                    thermo_hit_hist[:, 7],
                    photo_hit_hist[:, 7]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile8 = pd.DataFrame(([ORN_hit_hist[:, 8], 
                    AN_hit_hist[:, 8],
                    MN_hit_hist[:, 8], 
                    A00c_hit_hist[:, 8],
                    vtd_hit_hist[:, 8], 
                    thermo_hit_hist[:, 8],
                    photo_hit_hist[:, 8]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile = sensory_profile.T
sensory_profile0 = sensory_profile0.T
sensory_profile1 = sensory_profile1.T
sensory_profile2 = sensory_profile2.T
sensory_profile3 = sensory_profile3.T
sensory_profile4 = sensory_profile4.T
sensory_profile5 = sensory_profile5.T
sensory_profile6 = sensory_profile6.T
sensory_profile7 = sensory_profile7.T
sensory_profile8 = sensory_profile8.T


#%%
# multisensory elements per layer (apples to apples)

threshold = 25

ORN0_indices = np.where(ORN_hit_hist[:, 0]>threshold)[0]
ORN1_indices = np.where(ORN_hit_hist[:, 1]>threshold)[0]
ORN2_indices = np.where(ORN_hit_hist[:, 2]>threshold)[0]
ORN3_indices = np.where(ORN_hit_hist[:, 3]>threshold)[0]
ORN4_indices = np.where(ORN_hit_hist[:, 4]>threshold)[0]
ORN5_indices = np.where(ORN_hit_hist[:, 5]>threshold)[0]
ORN6_indices = np.where(ORN_hit_hist[:, 6]>threshold)[0]
ORN7_indices = np.where(ORN_hit_hist[:, 7]>threshold)[0]
ORN8_indices = np.where(ORN_hit_hist[:, 8]>threshold)[0]


AN0_indices = np.where(AN_hit_hist[:, 0]>threshold)[0]
AN1_indices = np.where(AN_hit_hist[:, 1]>threshold)[0]
AN2_indices = np.where(AN_hit_hist[:, 2]>threshold)[0]
AN3_indices = np.where(AN_hit_hist[:, 3]>threshold)[0]
AN4_indices = np.where(AN_hit_hist[:, 4]>threshold)[0]
AN5_indices = np.where(AN_hit_hist[:, 5]>threshold)[0]
AN6_indices = np.where(AN_hit_hist[:, 6]>threshold)[0]
AN7_indices = np.where(AN_hit_hist[:, 7]>threshold)[0]
AN8_indices = np.where(AN_hit_hist[:, 8]>threshold)[0]


MN0_indices = np.where(MN_hit_hist[:, 0]>threshold)[0]
MN1_indices = np.where(MN_hit_hist[:, 1]>threshold)[0]
MN2_indices = np.where(MN_hit_hist[:, 2]>threshold)[0]
MN3_indices = np.where(MN_hit_hist[:, 3]>threshold)[0]
MN4_indices = np.where(MN_hit_hist[:, 4]>threshold)[0]
MN5_indices = np.where(MN_hit_hist[:, 5]>threshold)[0]
MN6_indices = np.where(MN_hit_hist[:, 6]>threshold)[0]
MN7_indices = np.where(MN_hit_hist[:, 7]>threshold)[0]
MN8_indices = np.where(MN_hit_hist[:, 8]>threshold)[0]


A00c0_indices = np.where(A00c_hit_hist[:, 0]>threshold)[0]
A00c1_indices = np.where(A00c_hit_hist[:, 1]>threshold)[0]
A00c2_indices = np.where(A00c_hit_hist[:, 2]>threshold)[0]
A00c3_indices = np.where(A00c_hit_hist[:, 3]>threshold)[0]
A00c4_indices = np.where(A00c_hit_hist[:, 4]>threshold)[0]
A00c5_indices = np.where(A00c_hit_hist[:, 5]>threshold)[0]
A00c6_indices = np.where(A00c_hit_hist[:, 6]>threshold)[0]
A00c7_indices = np.where(A00c_hit_hist[:, 7]>threshold)[0]
A00c8_indices = np.where(A00c_hit_hist[:, 8]>threshold)[0]

vtd0_indices = np.where(vtd_hit_hist[:, 0]>threshold)[0]
vtd1_indices = np.where(vtd_hit_hist[:, 1]>threshold)[0]
vtd2_indices = np.where(vtd_hit_hist[:, 2]>threshold)[0]
vtd3_indices = np.where(vtd_hit_hist[:, 3]>threshold)[0]
vtd4_indices = np.where(vtd_hit_hist[:, 4]>threshold)[0]
vtd5_indices = np.where(vtd_hit_hist[:, 5]>threshold)[0]
vtd6_indices = np.where(vtd_hit_hist[:, 6]>threshold)[0]
vtd7_indices = np.where(vtd_hit_hist[:, 7]>threshold)[0]
vtd8_indices = np.where(vtd_hit_hist[:, 8]>threshold)[0]

thermo0_indices = np.where(thermo_hit_hist[:, 0]>threshold)[0]
thermo1_indices = np.where(thermo_hit_hist[:, 1]>threshold)[0]
thermo2_indices = np.where(thermo_hit_hist[:, 2]>threshold)[0]
thermo3_indices = np.where(thermo_hit_hist[:, 3]>threshold)[0]
thermo4_indices = np.where(thermo_hit_hist[:, 4]>threshold)[0]
thermo5_indices = np.where(thermo_hit_hist[:, 5]>threshold)[0]
thermo6_indices = np.where(thermo_hit_hist[:, 6]>threshold)[0]
thermo7_indices = np.where(thermo_hit_hist[:, 7]>threshold)[0]
thermo8_indices = np.where(thermo_hit_hist[:, 8]>threshold)[0]

photo0_indices = np.where(photo_hit_hist[:, 0]>threshold)[0]
photo1_indices = np.where(photo_hit_hist[:, 1]>threshold)[0]
photo2_indices = np.where(photo_hit_hist[:, 2]>threshold)[0]
photo3_indices = np.where(photo_hit_hist[:, 3]>threshold)[0]
photo4_indices = np.where(photo_hit_hist[:, 4]>threshold)[0]
photo5_indices = np.where(photo_hit_hist[:, 5]>threshold)[0]
photo6_indices = np.where(photo_hit_hist[:, 6]>threshold)[0]
photo7_indices = np.where(photo_hit_hist[:, 7]>threshold)[0]
photo8_indices = np.where(photo_hit_hist[:, 8]>threshold)[0]

ORN_profile = pd.DataFrame([np.array(sensory_profile0.iloc[ORN0_indices, :].sum(axis=0)/len(ORN0_indices)), 
                    np.array(sensory_profile1.iloc[ORN1_indices, :].sum(axis=0)/len(ORN1_indices)),
                    np.array(sensory_profile2.iloc[ORN2_indices, :].sum(axis=0)/len(ORN2_indices)),
                    np.array(sensory_profile3.iloc[ORN3_indices, :].sum(axis=0)/len(ORN3_indices)), 
                    np.array(sensory_profile4.iloc[ORN4_indices, :].sum(axis=0)/len(ORN4_indices)), 
                    np.array(sensory_profile5.iloc[ORN5_indices, :].sum(axis=0)/len(ORN5_indices)),
                    np.array(sensory_profile6.iloc[ORN6_indices, :].sum(axis=0)/len(ORN6_indices)),
                    np.array(sensory_profile7.iloc[ORN7_indices, :].sum(axis=0)/len(ORN7_indices)),
                    np.array(sensory_profile8.iloc[ORN8_indices, :].sum(axis=0)/len(ORN8_indices))],
                    index = ['ORN0', 'ORN1', 'ORN2', 'ORN3', 'ORN4', 'ORN5', 'ORN6', 'ORN7', 'ORN8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

AN_profile = pd.DataFrame([np.array(sensory_profile0.iloc[AN0_indices, :].sum(axis=0)/len(AN0_indices)), 
                    np.array(sensory_profile1.iloc[AN1_indices, :].sum(axis=0)/len(AN1_indices)),
                    np.array(sensory_profile2.iloc[AN2_indices, :].sum(axis=0)/len(AN2_indices)),
                    np.array(sensory_profile3.iloc[AN3_indices, :].sum(axis=0)/len(AN3_indices)), 
                    np.array(sensory_profile4.iloc[AN4_indices, :].sum(axis=0)/len(AN4_indices)), 
                    np.array(sensory_profile5.iloc[AN5_indices, :].sum(axis=0)/len(AN5_indices)),
                    np.array(sensory_profile6.iloc[AN6_indices, :].sum(axis=0)/len(AN6_indices)), 
                    np.array(sensory_profile7.iloc[AN7_indices, :].sum(axis=0)/len(AN7_indices)), 
                    np.array(sensory_profile8.iloc[AN8_indices, :].sum(axis=0)/len(AN8_indices))],
                    index = ['AN0', 'AN1', 'AN2', 'AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

MN_profile = pd.DataFrame([np.array(sensory_profile0.iloc[MN0_indices, :].sum(axis=0)/len(MN0_indices)), 
                    np.array(sensory_profile1.iloc[MN1_indices, :].sum(axis=0)/len(MN1_indices)),
                    np.array(sensory_profile2.iloc[MN2_indices, :].sum(axis=0)/len(MN2_indices)),
                    np.array(sensory_profile3.iloc[MN3_indices, :].sum(axis=0)/len(MN3_indices)), 
                    np.array(sensory_profile4.iloc[MN4_indices, :].sum(axis=0)/len(MN4_indices)), 
                    np.array(sensory_profile5.iloc[MN5_indices, :].sum(axis=0)/len(MN5_indices)),
                    np.array(sensory_profile6.iloc[MN6_indices, :].sum(axis=0)/len(MN6_indices)), 
                    np.array(sensory_profile7.iloc[MN7_indices, :].sum(axis=0)/len(MN7_indices)), 
                    np.array(sensory_profile8.iloc[MN8_indices, :].sum(axis=0)/len(MN8_indices))],
                    index = ['MN0', 'MN1', 'MN2', 'MN3', 'MN4', 'MN5', 'MN6', 'MN7', 'MN8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

A00c_profile = pd.DataFrame([np.array(sensory_profile0.iloc[A00c0_indices, :].sum(axis=0)/len(A00c0_indices)), 
                    np.array(sensory_profile1.iloc[A00c1_indices, :].sum(axis=0)/len(A00c1_indices)),
                    np.array(sensory_profile2.iloc[A00c2_indices, :].sum(axis=0)/len(A00c2_indices)),
                    np.array(sensory_profile3.iloc[A00c3_indices, :].sum(axis=0)/len(A00c3_indices)), 
                    np.array(sensory_profile4.iloc[A00c4_indices, :].sum(axis=0)/len(A00c4_indices)), 
                    np.array(sensory_profile5.iloc[A00c5_indices, :].sum(axis=0)/len(A00c5_indices)),
                    np.array(sensory_profile6.iloc[A00c6_indices, :].sum(axis=0)/len(A00c6_indices)), 
                    np.array(sensory_profile7.iloc[A00c7_indices, :].sum(axis=0)/len(A00c7_indices)), 
                    np.array(sensory_profile8.iloc[A00c8_indices, :].sum(axis=0)/len(A00c8_indices))],
                    index = ['A00c0', 'A00c1', 'A00c2', 'A00c3', 'A00c4', 'A00c5', 'A00c6', 'A00c7', 'A00c8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

vtd_profile = pd.DataFrame([np.array(sensory_profile0.iloc[vtd0_indices, :].sum(axis=0)/len(vtd0_indices)), 
                    np.array(sensory_profile1.iloc[vtd1_indices, :].sum(axis=0)/len(vtd1_indices)),
                    np.array(sensory_profile2.iloc[vtd2_indices, :].sum(axis=0)/len(vtd2_indices)),
                    np.array(sensory_profile3.iloc[vtd3_indices, :].sum(axis=0)/len(vtd3_indices)), 
                    np.array(sensory_profile4.iloc[vtd4_indices, :].sum(axis=0)/len(vtd4_indices)), 
                    np.array(sensory_profile5.iloc[vtd5_indices, :].sum(axis=0)/len(vtd5_indices)),
                    np.array(sensory_profile6.iloc[vtd6_indices, :].sum(axis=0)/len(vtd6_indices)), 
                    np.array(sensory_profile7.iloc[vtd7_indices, :].sum(axis=0)/len(vtd7_indices)), 
                    np.array(sensory_profile8.iloc[vtd8_indices, :].sum(axis=0)/len(vtd8_indices))],
                    index = ['vtd0', 'vtd1', 'vtd2', 'vtd3', 'vtd4', 'vtd5', 'vtd6', 'vtd7', 'vtd8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

thermo_profile = pd.DataFrame([np.array(sensory_profile0.iloc[thermo0_indices, :].sum(axis=0)/len(thermo0_indices)), 
                    np.array(sensory_profile1.iloc[thermo1_indices, :].sum(axis=0)/len(thermo1_indices)),
                    np.array(sensory_profile2.iloc[thermo2_indices, :].sum(axis=0)/len(thermo2_indices)),
                    np.array(sensory_profile3.iloc[thermo3_indices, :].sum(axis=0)/len(thermo3_indices)), 
                    np.array(sensory_profile4.iloc[thermo4_indices, :].sum(axis=0)/len(thermo4_indices)), 
                    np.array(sensory_profile5.iloc[thermo5_indices, :].sum(axis=0)/len(thermo5_indices)),
                    np.array(sensory_profile6.iloc[thermo6_indices, :].sum(axis=0)/len(thermo6_indices)), 
                    np.array(sensory_profile7.iloc[thermo7_indices, :].sum(axis=0)/len(thermo7_indices)), 
                    np.array(sensory_profile8.iloc[thermo8_indices, :].sum(axis=0)/len(thermo8_indices))],
                    index = ['thermo0', 'thermo1', 'thermo2', 'thermo3', 'thermo4', 'thermo5', 'thermo6', 'thermo7', 'thermo8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

photo_profile = pd.DataFrame([np.array(sensory_profile0.iloc[photo0_indices, :].sum(axis=0)/len(photo0_indices)), 
                    np.array(sensory_profile1.iloc[photo1_indices, :].sum(axis=0)/len(photo1_indices)),
                    np.array(sensory_profile2.iloc[photo2_indices, :].sum(axis=0)/len(photo2_indices)),
                    np.array(sensory_profile3.iloc[photo3_indices, :].sum(axis=0)/len(photo3_indices)), 
                    np.array(sensory_profile4.iloc[photo4_indices, :].sum(axis=0)/len(photo4_indices)), 
                    np.array(sensory_profile5.iloc[photo5_indices, :].sum(axis=0)/len(photo5_indices)),
                    np.array(sensory_profile6.iloc[photo6_indices, :].sum(axis=0)/len(photo3_indices)), 
                    np.array(sensory_profile7.iloc[photo7_indices, :].sum(axis=0)/len(photo4_indices)), 
                    np.array(sensory_profile8.iloc[photo8_indices, :].sum(axis=0)/len(photo5_indices))],
                    index = ['photo0', 'photo1', 'photo2', 'photo3', 'photo4', 'photo5', 'photo6', 'photo7', 'photo8'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])
# %%
# plotting multisensory elements per layer

x_axis_labels = [0,1,2,3,4,5,6]
x_label = 'Hops from Sensory'

fig, axs = plt.subplots(
    4, 2, figsize=(5, 8)
)

fig.tight_layout(pad=2.5)
#cbar_ax = axs.add_axes([3, 7, .1, .75])

ax = axs[0, 0] 
ax.set_title('Signal from ORN')
ax.set(xticks=[0, 1, 2, 3, 4, 5])
sns.heatmap(ORN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[1, 0]
ax.set_title('Signal from AN')
sns.heatmap(AN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[2, 0]
ax.set_title('Signal from MN')
sns.heatmap(MN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[3, 0]
ax.set_title('Signal from A00c')
sns.heatmap(A00c_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)
ax.set_xlabel(x_label)

ax = axs[0, 1]
ax.set_title('Signal from vtd')
sns.heatmap(vtd_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[1, 1]
ax.set_title('Signal from thermo')
sns.heatmap(thermo_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[2, 1]
ax.set_title('Signal from photo')
sns.heatmap(photo_profile.T.iloc[:,0:7], ax = ax, cbar_ax = axs[3, 1], xticklabels = x_axis_labels, rasterized=True)
ax.set_xlabel(x_label)

ax = axs[3, 1]
ax.set_xlabel('Number of Visits\nfrom Sensory Signal')
#ax.axis("off")

plt.savefig('cascades/plots/sensory_integration_per_hop.pdf', format='pdf', bbox_inches='tight')


# %%
# parallel coordinate plot of different sensory layer integration
fig, axs = plt.subplots(
    6, 7, figsize=(30, 30), sharey = True
)

fig.tight_layout(pad=2.5)

threshold = 25
alpha = 0.10

#fig.tight_layout(pad=3.0)
sensory_profile0_parallel = sensory_profile0
sensory_profile0_parallel['class'] = np.zeros(len(sensory_profile0_parallel))
sensory_profile1_parallel = sensory_profile1
sensory_profile1_parallel['class'] = np.zeros(len(sensory_profile1_parallel))
sensory_profile2_parallel = sensory_profile2
sensory_profile2_parallel['class'] = np.zeros(len(sensory_profile2_parallel))
sensory_profile3_parallel = sensory_profile3
sensory_profile3_parallel['class'] = np.zeros(len(sensory_profile3_parallel))
sensory_profile4_parallel = sensory_profile4
sensory_profile4_parallel['class'] = np.zeros(len(sensory_profile4_parallel))
sensory_profile5_parallel = sensory_profile5
sensory_profile5_parallel['class'] = np.zeros(len(sensory_profile5_parallel))

column = 0
color = 'blue'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(ORN_hit_hist[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(ORN_hit_hist[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(ORN_hit_hist[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(ORN_hit_hist[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(ORN_hit_hist[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(ORN_hit_hist[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)


modality_list = AN_hit_hist
column = 1
color = 'orange'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(modality_list[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(modality_list[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(modality_list[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(modality_list[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(modality_list[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(modality_list[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)


modality_list = MN_hit_hist
column = 2
color = 'green'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(modality_list[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(modality_list[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(modality_list[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(modality_list[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(modality_list[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(modality_list[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)


modality_list = A00c_hit_hist
column = 3
color = 'maroon'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(modality_list[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(modality_list[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(modality_list[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(modality_list[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(modality_list[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(modality_list[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)



modality_list = vtd_hit_hist
column = 4
color = 'purple'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(modality_list[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(modality_list[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(modality_list[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(modality_list[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(modality_list[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(modality_list[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)


modality_list = thermo_hit_hist
column = 5
color = 'navy'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(modality_list[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(modality_list[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(modality_list[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(modality_list[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(modality_list[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(modality_list[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)


modality_list = photo_hit_hist
column = 6
color = 'black'
ax = axs[0, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile0_parallel.iloc[np.where(modality_list[:, 0]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[1, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile1_parallel.iloc[np.where(modality_list[:, 1]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[2, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile2_parallel.iloc[np.where(modality_list[:, 2]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[3, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile3_parallel.iloc[np.where(modality_list[:, 3]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[4, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile4_parallel.iloc[np.where(modality_list[:, 4]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

ax = axs[5, column]
ax.set(ylim = (0, 100))
parallel_coordinates(sensory_profile5_parallel.iloc[np.where(modality_list[:, 5]>threshold)[0], :], class_column = 'class', ax = ax, alpha = alpha, color = color)

plt.savefig('cascades/plots/sensory_integration_per_hop_parallel_coords_plot.pdf', format='pdf', bbox_inches='tight')
