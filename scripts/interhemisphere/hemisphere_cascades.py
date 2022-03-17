#%%
import sys
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cmasher as cmr

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

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

# %%
# pull sensory annotations and then pull associated skids
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [ct.Celltype(name, ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

# load left / right annotations
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

# need to switch several ascending neurons because they ascending contralateral and some contra-contra neurons

# contralateral input neurons
neurons_to_flip = pymaid.get_skids_by_annotation('mw contralateral axon')
neurons_to_flip_left = [skid for skid in neurons_to_flip if ((skid in left) & (skid in input_skids))]
neurons_to_flip_right = [skid for skid in neurons_to_flip if ((skid in right) & (skid in input_skids))]

# add neurons with contralateral axons and dendrites so they are displayed on the correct side
contra_contra = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))
neurons_to_flip_left = neurons_to_flip_left + list(np.intersect1d(contra_contra, left))
neurons_to_flip_right = neurons_to_flip_right + list(np.intersect1d(contra_contra, right))

# removing neurons_to_flip and adding to the other side
left = list(np.setdiff1d(left, neurons_to_flip_left)) + neurons_to_flip_right
right = list(np.setdiff1d(right, neurons_to_flip_right)) + neurons_to_flip_left

input_skids_left = list(np.intersect1d(input_skids, left))
input_skids_right = list(np.intersect1d(input_skids, right))

# remove bilateral axon input neurons to see how the mixing happens at the interneuron level
bilat_axon = pymaid.get_skids_by_annotation('mw bilateral axon')
bilat_axon = bilat_axon + [3795424, 11291344] # remove the ambiguous v'td neurons (project to middle of SEZ)

input_skids_left = list(np.setdiff1d(input_skids_left, bilat_axon))
input_skids_right = list(np.setdiff1d(input_skids_right, bilat_axon))

input_skids_list = [input_skids_left, input_skids_right]

#%%
# cascades from each sensory modality
# save as pickle to use later because cascades are stochastic; prevents the need to remake plots everytime
import pickle

p = 0.05
max_hops = 10
n_init = 1000
simultaneous = True
adj=adj_ad
'''
input_hit_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = ['left_inputs', 'right_inputs'], stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(input_hit_hist_list, open(f'data/cascades/hemisphere-cascades_{n_init}-n_init.p', 'wb'))
'''
input_hit_hist_list = pickle.load(open(f'data/cascades/hemisphere-cascades_{n_init}-n_init.p', 'rb'))

# %%
# plot heatmaps of number of neurons over-threshold per hop

def intersect_stats(hit_hist1, hit_hist2, threshold, hops):
    intersect_hops = []
    total_hops = []

    for i in np.arange(0, hops):
        intersect = list(np.logical_and(hit_hist1.loc[:,i]>threshold, hit_hist2.loc[:,i]>threshold))
        total = list(np.logical_or(hit_hist1.loc[:,i]>threshold, hit_hist2.loc[:,i]>threshold))
        intersect_hops.append(intersect)
        total_hops.append(total)

    intersect_hops = pd.DataFrame(intersect_hops, index=range(0, hops), columns = hit_hist1.index).T
    total_hops = pd.DataFrame(total_hops, index=range(0, hops), columns = hit_hist1.index).T

    percent = []
    for i in np.arange(0, hops):
        if(sum(total_hops[i])>0):
            percent.append(sum(intersect_hops[i])/sum(total_hops[i]))
        if(sum(total_hops[i])==0):
            percent.append(0)

    return(intersect_hops, total_hops, percent)

all_inputs_hit_hist_left = input_hit_hist_list[0].skid_hit_hist
all_inputs_hit_hist_right = input_hit_hist_list[1].skid_hit_hist

threshold = n_init/2
hops = 10
all_inputs_intersect, all_inputs_total, all_inputs_percent = intersect_stats(all_inputs_hit_hist_left, all_inputs_hit_hist_right, threshold, hops)

# identify left/right ipsi, bilateral, contralaterals
# majority types
ipsi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
ipsi = ipsi + list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))
bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
contralateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))

# add ipsilateral sensory to each
ipsi = ipsi + input_skids_left + input_skids_right

ipsi_left = list(np.intersect1d(ipsi, left))
ipsi_right = list(np.intersect1d(ipsi, right))
bilateral_left = list(np.intersect1d(bilateral, left))
bilateral_right = list(np.intersect1d(bilateral, right))
contra_left = list(np.intersect1d(contralateral, left))
contra_right = list(np.intersect1d(contralateral, right))

ipsi_left = list(np.intersect1d(ipsi_left, all_inputs_hit_hist_left.index))
ipsi_right = list(np.intersect1d(ipsi_right, all_inputs_hit_hist_right.index))
bilateral_left = list(np.intersect1d(bilateral_left, all_inputs_hit_hist_left.index))
bilateral_right = list(np.intersect1d(bilateral_right, all_inputs_hit_hist_right.index))
contra_left = list(np.intersect1d(contra_left, all_inputs_hit_hist_left.index))
contra_right = list(np.intersect1d(contra_right, all_inputs_hit_hist_right.index))

# plot results
fig, axs = plt.subplots(
    3, 1, figsize=(1, 1.75), sharex=True
)
fig.tight_layout(pad=0.05)

ax = axs[0]
i_left = (all_inputs_hit_hist_left.loc[ipsi_left]>threshold).sum(axis=0)
b_left = (all_inputs_hit_hist_left.loc[bilateral_left]>threshold).sum(axis=0)
c_left = (all_inputs_hit_hist_left.loc[contra_left]>threshold).sum(axis=0)
c_right = (all_inputs_hit_hist_left.loc[contra_right]>threshold).sum(axis=0)
b_right = (all_inputs_hit_hist_left.loc[bilateral_right]>threshold).sum(axis=0)
i_right = (all_inputs_hit_hist_left.loc[ipsi_right]>threshold).sum(axis=0)

data_left = pd.DataFrame([i_left, b_left, c_left, c_right, b_right, i_right], index = ['Ipsi(L)', 'Bilateral(L)', 'Contra(L)', 'Contra(R)', 'Bilateral(R)', 'Ipsi(R)'])
sns.heatmap(data_left.iloc[:, 0:5], ax = ax, annot=True, fmt="d", cbar = False)
ax.tick_params(left=False, bottom=False)

ax = axs[1]
i_left = (all_inputs_hit_hist_right.loc[ipsi_left]>threshold).sum(axis=0)
b_left = (all_inputs_hit_hist_right.loc[bilateral_left]>threshold).sum(axis=0)
c_left = (all_inputs_hit_hist_right.loc[contra_left]>threshold).sum(axis=0)
c_rightc_right = (all_inputs_hit_hist_right.loc[contra_right]>threshold).sum(axis=0)
b_right = (all_inputs_hit_hist_right.loc[bilateral_right]>threshold).sum(axis=0)
i_right = (all_inputs_hit_hist_right.loc[ipsi_right]>threshold).sum(axis=0)

data_right = pd.DataFrame([i_left, b_left, c_left, c_right, b_right, i_right], index = ['Ipsi(L)', 'Bilateral(L)', 'Contra(L)', 'Contra(R)', 'Bilateral(R)', 'Ipsi(R)'])
sns.heatmap(data_right.iloc[:, 0:5], ax = ax, annot=True, fmt="d", cbar = False)
ax.tick_params(left=False, bottom=False)

ax = axs[2]
i_left = all_inputs_intersect.loc[ipsi_left].sum(axis=0)/all_inputs_total.loc[ipsi_left].sum(axis=0)
b_left = all_inputs_intersect.loc[bilateral_left].sum(axis=0)/all_inputs_total.loc[bilateral_left].sum(axis=0)
c_left = all_inputs_intersect.loc[contra_left].sum(axis=0)/all_inputs_total.loc[contra_left].sum(axis=0)
c_right = all_inputs_intersect.loc[contra_right].sum(axis=0)/all_inputs_total.loc[contra_right].sum(axis=0)
b_right = all_inputs_intersect.loc[bilateral_right].sum(axis=0)/all_inputs_total.loc[bilateral_right].sum(axis=0)
i_right = all_inputs_intersect.loc[ipsi_right].sum(axis=0)/all_inputs_total.loc[ipsi_right].sum(axis=0)

data = pd.DataFrame([i_left, b_left, c_left, c_right, b_right, i_right], index = ['Ipsi(L)', 'Bilateral(L)', 'Contra(L)', 'Contra(R)', 'Bilateral(R)', 'Ipsi(R)'])
data = data.fillna(0)
sns.heatmap(data.iloc[:, 0:5], ax = ax, annot=True, fmt=".0%", cbar = False, cmap = cmr.lavender)
ax.tick_params(left=False, bottom=False)
fig.savefig('interhemisphere/plots/summary_intersect-plot.pdf', format='pdf', bbox_inches='tight')

# plot results
fig, axs = plt.subplots(
    3, 1, figsize=(1, 1.75), sharex=True
)
fig.tight_layout(pad=0.05)

ax = axs[0]
i_left = (all_inputs_hit_hist_left.loc[ipsi_left]>threshold).sum(axis=0)
b_left = (all_inputs_hit_hist_left.loc[bilateral_left]>threshold).sum(axis=0)
c_left = (all_inputs_hit_hist_left.loc[contra_left]>threshold).sum(axis=0)
c_right = (all_inputs_hit_hist_left.loc[contra_right]>threshold).sum(axis=0)
b_right = (all_inputs_hit_hist_left.loc[bilateral_right]>threshold).sum(axis=0)
i_right = (all_inputs_hit_hist_left.loc[ipsi_right]>threshold).sum(axis=0)

data_left = pd.DataFrame([i_left, b_left, c_left, c_right, b_right, i_right], index = ['Ipsi(L)', 'Bilateral(L)', 'Contra(L)', 'Contra(R)', 'Bilateral(R)', 'Ipsi(R)'])
sns.heatmap(data_left.iloc[:, 0:5], ax = ax, annot=True, fmt="d", cbar = False)
ax.tick_params(left=False, bottom=False)

ax = axs[1]
i_left = (all_inputs_hit_hist_right.loc[ipsi_left]>threshold).sum(axis=0)
b_left = (all_inputs_hit_hist_right.loc[bilateral_left]>threshold).sum(axis=0)
c_left = (all_inputs_hit_hist_right.loc[contra_left]>threshold).sum(axis=0)
c_rightc_right = (all_inputs_hit_hist_right.loc[contra_right]>threshold).sum(axis=0)
b_right = (all_inputs_hit_hist_right.loc[bilateral_right]>threshold).sum(axis=0)
i_right = (all_inputs_hit_hist_right.loc[ipsi_right]>threshold).sum(axis=0)

data_right = pd.DataFrame([i_left, b_left, c_left, c_right, b_right, i_right], index = ['Ipsi(L)', 'Bilateral(L)', 'Contra(L)', 'Contra(R)', 'Bilateral(R)', 'Ipsi(R)'])
sns.heatmap(data_right.iloc[:, 0:5], ax = ax, annot=True, fmt="d", cbar = False)
ax.tick_params(left=False, bottom=False)

ax = axs[2]
i_left = all_inputs_intersect.loc[ipsi_left].sum(axis=0)
b_left = all_inputs_intersect.loc[bilateral_left].sum(axis=0)
c_left = all_inputs_intersect.loc[contra_left].sum(axis=0)
c_right = all_inputs_intersect.loc[contra_right].sum(axis=0)
b_right = all_inputs_intersect.loc[bilateral_right].sum(axis=0)
i_right = all_inputs_intersect.loc[ipsi_right].sum(axis=0)

data = pd.DataFrame([i_left, b_left, c_left, c_right, b_right, i_right], index = ['Ipsi(L)', 'Bilateral(L)', 'Contra(L)', 'Contra(R)', 'Bilateral(R)', 'Ipsi(R)'])
data = data.fillna(0)
sns.heatmap(data.iloc[:, 0:5], ax = ax, annot=True, cbar = False, cmap = cmr.lavender)
ax.tick_params(left=False, bottom=False)
fig.savefig('interhemisphere/plots/summary_intersect-plot_raw-counts.pdf', format='pdf', bbox_inches='tight')

# %%
# identify integration center neurons
# what types of neurons are they?

data_mat = pd.DataFrame(all_inputs_intersect)
data_ipsi = data_mat.loc[np.intersect1d(ipsi, data_mat.index), :]
data_bilat = data_mat.loc[np.intersect1d(bilateral, data_mat.index), :]
data_contra = data_mat.loc[np.intersect1d(contralateral, data_mat.index), :]

all_cats = []
for i in range(len(data_mat.columns)):
    cats_hop = []
    cats_hop.append(ct.Celltype(f'hop{i}_ipsi_integrators', list(data_ipsi[data_ipsi.iloc[:, i]].index)))
    cats_hop.append(ct.Celltype(f'hop{i}_bilateral_integrators', list(data_bilat[data_bilat.iloc[:, i]].index)))
    cats_hop.append(ct.Celltype(f'hop{i}_contra_integrators', list(data_contra[data_contra.iloc[:, i]].index)))

    all_cats.append(cats_hop)

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

all_cat_memberships=[]
for i in range(len(all_cats)):
    all_cats_analyzer = ct.Celltype_Analyzer(all_cats[i])
    all_cats_analyzer.set_known_types(celltypes)
    cats_memberships = all_cats_analyzer.memberships(raw_num=True) #switch to False for percent neurons
    all_cat_memberships.append(cats_memberships)

integrator2hop = [skid for subset in [x.skids for x in all_cats[2]] for skid in subset]
integrator3hop = [skid for subset in [x.skids for x in all_cats[3]] for skid in subset]
integrator4hop = [skid for subset in [x.skids for x in all_cats[4]] for skid in subset]

pymaid.add_annotations(integrator2hop, 'mw interhemispheric integration 2-hop')
pymaid.add_annotations(integrator3hop, 'mw interhemispheric integration 3-hop')
pymaid.add_annotations(integrator4hop, 'mw interhemispheric integration 4-hop')

colors = [x.get_color() for x in celltypes] + ['tab:gray']
fraction_types_names = all_cat_memberships[1].index
#plt.bar(x=fraction_types_names,height=[1]*len(colors),color=colors)

for i in range(1, 5):
    plts=[]
    fig, ax = plt.subplots(figsize=(0.55,.6))
    plt1 = plt.bar(all_cat_memberships[i].columns, all_cat_memberships[i].iloc[0, :], color=colors[0])
    bottom = all_cat_memberships[i].iloc[0, :]
    plt.xticks(rotation=45, ha='right')
    ax.set(ylim=(0,100))
    plts.append(plt1)

    for j in range(1, len(all_cat_memberships[i].iloc[:, 0])):
        plt_next = plt.bar(all_cat_memberships[i].columns, all_cat_memberships[i].iloc[j, :], bottom = bottom, color = colors[j])
        bottom = bottom + all_cat_memberships[i].iloc[j, :]
        plts.append(plt_next)
        ax.set(ylim=(0,100))
        plt.xticks(rotation=45, ha='right')

    plt.savefig(f'interhemisphere/plots/ipsi_bi_contra_identities/integrators_hop{i}.pdf', format='pdf', bbox_inches='tight')

# %%
# cascades to descendings; L/R bias of descending input

pairs = pm.Promat.get_pairs()
dVNC = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs)
dSEZ = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs)
RGN = pm.Promat.load_pairs_from_annotation('mw RGN', pairs)

dVNC_left = list(dVNC.leftid)
dVNC_right = list(dVNC.rightid)
dSEZ_left = list(dSEZ.leftid)
dSEZ_right = list(dSEZ.rightid)
RGN_left = list(RGN.leftid)
RGN_right = list(RGN.rightid)

left_signal = all_inputs_hit_hist_left/n_init
left_signal = left_signal.sum(axis=1)

right_signal = all_inputs_hit_hist_right/n_init
right_signal = -(right_signal.sum(axis=1))

integration = (left_signal + right_signal)
integration_df = pd.DataFrame(list(zip(left_signal, right_signal, integration)), index = adj.index)


df_left = integration_df.loc[dVNC_left, :]
df_left = df_left.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
df_left = df_left.append(integration_df.loc[dSEZ_left, :])
df_left = df_left.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
df_left = df_left.append(integration_df.loc[RGN_left, :])

df_right = integration_df.loc[dVNC_right, :]
df_right = df_right.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
df_right = df_right.append(integration_df.loc[dSEZ_right, :])
df_right = df_right.append([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
df_right = df_right.append(integration_df.loc[RGN_right, :])

fig, axs = plt.subplots(1,2, figsize=(1.5,1.5), sharey=True)
fig.tight_layout(pad=0.05)
ax=axs[0]
sns.heatmap(df_left, cmap=cmr.iceburn, ax=ax, cbar=False)
ax.tick_params(left=False, bottom=False)
ax.set(yticks=([]))

ax=axs[1]
sns.heatmap(df_right, cmap=cmr.iceburn, ax=ax, cbar=False)
ax.tick_params(left=False, bottom=False)
ax.set(yticks=([]))
fig.savefig('interhemisphere/plots/left-right-visits_brain_outputs.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(1.5,1.5), sharey=True)
sns.heatmap(df_left, cmap=cmr.iceburn, ax=ax)
fig.savefig('interhemisphere/plots/left-right-visits_brain_outputs_cbar.pdf', format='pdf', bbox_inches='tight')
# %%
# lateralization metric to determine how much left/right mixing happens per neuron

left_signal = all_inputs_hit_hist_left/n_init
left_signal = left_signal.sum(axis=1)

right_signal = all_inputs_hit_hist_right/n_init
right_signal = -(right_signal.sum(axis=1))

integration = (left_signal + right_signal)
integration_df = pd.DataFrame(list(zip(left_signal, right_signal, integration)), index = adj.index)


pairs = pm.Promat.get_pairs()
dVNC = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs, return_type='all_pair_sorted')
dSEZ = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_sorted')
RGN = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_sorted')

left_signal = all_inputs_hit_hist_left/n_init
left_signal = left_signal.sum(axis=1)

right_signal = all_inputs_hit_hist_right/n_init
right_signal = -(right_signal.sum(axis=1))

integration = (left_signal + right_signal)
integration_df = pd.DataFrame(list(zip(left_signal, right_signal, integration)), index = adj.index, columns = ['left_signal', 'right_signal', 'left_right_signal'])

left_int = []
right_int = []
left_right_int = []
for i in dVNC.index:
    leftid = dVNC.loc[i, 'leftid']
    rightid = dVNC.loc[i, 'rightid']

    int_left = integration_df.loc[leftid, 'left_right_signal']
    int_right = integration_df.loc[rightid, 'left_right_signal']
    left_int.append(int_left)
    right_int.append(int_right)
    left_right_int.append(((int_left)+-(int_right))/2)

dVNC['left_integration'] = left_int
dVNC['right_integration'] = right_int
dVNC['lateralization'] = left_right_int

left_int = []
right_int = []
left_right_int = []
for i in dSEZ.index:
    leftid = dSEZ.loc[i, 'leftid']
    rightid = dSEZ.loc[i, 'rightid']

    int_left = integration_df.loc[leftid, 'left_right_signal']
    int_right = integration_df.loc[rightid, 'left_right_signal']
    left_int.append(int_left)
    right_int.append(int_right)
    left_right_int.append(((int_left)+-(int_right))/2)

dSEZ['left_integration'] = left_int
dSEZ['right_integration'] = right_int
dSEZ['lateralization'] = left_right_int

left_int = []
right_int = []
left_right_int = []
for i in RGN.index:
    leftid = RGN.loc[i, 'leftid']
    rightid = RGN.loc[i, 'rightid']

    int_left = integration_df.loc[leftid, 'left_right_signal']
    int_right = integration_df.loc[rightid, 'left_right_signal']
    left_int.append(int_left)
    right_int.append(int_right)
    left_right_int.append(((int_left)+-(int_right))/2)

RGN['left_integration'] = left_int
RGN['right_integration'] = right_int
RGN['lateralization'] = left_right_int

s = 2
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.scatterplot(x=[x for x in range(0, len(dVNC))], y=dVNC.lateralization.sort_values(), color='#A52A2A', ax=ax, s=s)
sns.scatterplot(x=[x+len(dVNC) for x in range(0, len(dSEZ))], y=dSEZ.lateralization.sort_values(), color='#C47451', ax=ax, s=s)
sns.scatterplot(x=[x+len(dVNC)+len(dSEZ) for x in range(0, len(RGN))], y=RGN.lateralization.sort_values(), color='#9467BD', ax=ax, s=s)
ax.set(ylim=(-1,1))
plt.savefig('interhemisphere/plots/signal-lateralization.pdf', format='pdf', bbox_inches='tight')

s = 2
fig, ax = plt.subplots(1,1,figsize=(1,2))
sns.scatterplot(x=[x for x in range(0, len(dVNC))], y=dVNC.lateralization.sort_values(), color='#A52A2A', ax=ax, s=s)
sns.scatterplot(x=[x for x in range(0, len(dSEZ))], y=dSEZ.lateralization.sort_values(), color='#C47451', ax=ax, s=s)
sns.scatterplot(x=[x for x in range(0, len(RGN))], y=RGN.lateralization.sort_values(), color='#9467BD', ax=ax, s=s)
ax.set(ylim=(-1,1))
plt.savefig('interhemisphere/plots/signal-lateralization_overlapping.pdf', format='pdf', bbox_inches='tight')

'''
# rasterplot
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.set(ylim = (-0.05, 1))
ax.eventplot([[x] for x in dVNC.lateralization.sort_values()] + [[x] for x in dSEZ.lateralization.sort_values()] + [[x] for x in RGN.lateralization.sort_values()], \
            lineoffsets = np.arange(1, len(dVNC)+len(dSEZ)+len(RGN)+1), linewidths = 0.75, orientation='vertical', color=['#A52A2A']*len(dVNC) + ['#C47451']*len(dSEZ) + ['#9467BD']*len(RGN))
plt.savefig('interhemisphere/plots/signal-lateralization_raster.pdf', format='pdf', bbox_inches='tight')

dVNC['type'] =['dVNC']*len(dVNC)
dSEZ['type'] =['dSEZ']*len(dSEZ)
RGN['type'] =['RGN']*len(RGN)

outputs_df = pd.concat([dVNC, dSEZ, RGN], axis=0)

# catplot
fig, ax = plt.subplots(1,1,figsize=(2,6))
sns.boxenplot(data=outputs_df, y='lateralization', x='type')
plt.savefig('interhemisphere/plots/signal-lateralization_catplot.pdf', format='pdf', bbox_inches='tight')
'''

# percent >0.25
dVNC_lateralized = dVNC[abs(dVNC.lateralization)>0.25].leftid
dSEZ_lateralized = dSEZ[abs(dSEZ.lateralization)>0.25].leftid
RGN_lateralized = RGN[abs(RGN.lateralization)>0.25].leftid

lateralized = pd.DataFrame([[len(dVNC_lateralized)/len(dVNC), 'lateralized', 'dVNC'],
                            [1-len(dVNC_lateralized)/len(dVNC), 'mixed', 'dVNC'],
                            [len(dSEZ_lateralized)/len(dSEZ), 'lateralized', 'dSEZ'],
                            [1-len(dSEZ_lateralized)/len(dSEZ), 'mixed', 'dSEZ'],
                            [len(RGN_lateralized)/len(RGN), 'lateralized', 'RGN'],
                            [1-len(RGN_lateralized)/len(RGN), 'mixed', 'RGN']], columns = ['fraction', 'lateralization', 'type']) 

lateralized = pd.DataFrame([[len(dVNC_lateralized)/len(dVNC), len(dSEZ_lateralized)/len(dSEZ), len(RGN_lateralized)/len(RGN)], 
                [1-len(dVNC_lateralized)/len(dVNC), 1-len(dSEZ_lateralized)/len(dSEZ), 1-len(RGN_lateralized)/len(RGN)]], index = ['lateralized', 'mixed'], columns = ['dVNC', 'dSEZ', 'RGN'])
fig, ax = plt.subplots(1,1, figsize=(2,2))
ax.bar(x = lateralized.columns, height = lateralized.loc['lateralized', :])
ax.bar(x = lateralized.columns, height = lateralized.loc['mixed', :], bottom = lateralized.loc['lateralized', :])
plt.savefig('interhemisphere/plots/signal-lateralization_summary.pdf', format='pdf', bbox_inches='tight')

# %%
# lateralization of all brain neurons
brain_skids = np.setdiff1d(pymaid.get_skids_by_annotation('mw brain paper clustered neurons') + pymaid.get_skids_by_annotation('mw brain accessory neurons'), input_skids + ascending_unknown)
brain_skids = np.intersect1d(integration_df.index, brain_skids)
brain = pm.Promat.load_pairs_from_annotation(annot='', pairList=pairs, return_type='all_pair_sorted', skids=brain_skids, use_skids=True)

left_int = []
right_int = []
left_right_int = []
for i in brain.index:
    leftid = brain.loc[i, 'leftid']
    rightid = brain.loc[i, 'rightid']
    
    int_left = integration_df.loc[leftid, 'left_right_signal']
    int_right = integration_df.loc[rightid, 'left_right_signal']
    left_int.append(int_left)
    right_int.append(int_right)
    left_right_int.append(((int_left)+-(int_right))/2)

brain['left_integration'] = left_int
brain['right_integration'] = right_int
brain['lateralization'] = left_right_int

# flip value of FFN-18 (unannotated contra axon, contra/bilateral dendrite)
brain.loc[brain[brain.leftid==3622234].index, 'lateralization'] = -brain[brain.leftid==3622234].lateralization

# plot all brain lateralization 
brain_sort_subthres = brain.lateralization.sort_values()[brain.lateralization.sort_values()<=threshold]
brain_sort_thres = brain.lateralization.sort_values()[brain.lateralization.sort_values()>threshold]

s=6
alpha = 0.25
fig, ax = plt.subplots(1,1,figsize=(1,2))
plt.scatter(x=[x for x in range(0, len(brain_sort_subthres))], y=brain_sort_subthres, color='none', edgecolor=sns.color_palette()[0], linewidths=0.2, alpha=alpha, s=s)
plt.scatter(x=[x for x in range(len(brain_sort_subthres), len(brain_sort_subthres)+len(brain_sort_thres))], y=brain_sort_thres, color='none', edgecolor=sns.color_palette()[1], linewidths=0.2, alpha=alpha, s=s)
ax.set(ylim=(-1.05,1.05))
plt.savefig('interhemisphere/plots/signal-lateralization_whole-brain.pdf', format='pdf', bbox_inches='tight')

# identify neurons with >0.25 lateralization
threshold = 0.25
brain_lateralized_ipsi_left = list(brain[(brain.lateralization>threshold)].leftid) 
brain_lateralized_ipsi_right = list(brain[(brain.lateralization>threshold)].rightid)
brain_lateralized_ipsi = brain_lateralized_ipsi_left + brain_lateralized_ipsi_right
brain_lateralized_ipsi_ct = ct.Celltype('lateralized', brain_lateralized_ipsi, color=sns.color_palette()[1])

brain_lateralized_contra_left = list(brain[(brain.lateralization<-threshold)].leftid) 
brain_lateralized_contra_right = list(brain[(brain.lateralization<-threshold)].rightid)
brain_lateralized_contra = brain_lateralized_contra_left + brain_lateralized_contra_right
brain_lateralized_contra_ct = ct.Celltype('lateralized', brain_lateralized_contra, color=sns.color_palette()[3])

brain_nonlat_left = list(brain[(brain.lateralization<=threshold) & (brain.lateralization>=-threshold)].leftid) 
brain_nonlat_right = list(brain[(brain.lateralization<=threshold) & (brain.lateralization>=-threshold)].rightid) 
brain_nonlateralized = brain_nonlat_left + brain_nonlat_right
brain_nonlateralized_ct = ct.Celltype('non_lateralized', brain_nonlateralized, color=sns.color_palette()[0])

pdiff = pymaid.get_skids_by_annotation('mw partially differentiated')
_, celltypes = ct.Celltype_Analyzer.default_celltypes(exclude=pdiff)
celltype_analyzer = ct.Celltype_Analyzer(celltypes)
celltype_analyzer.set_known_types([brain_lateralized_ipsi_ct, brain_lateralized_contra_ct, brain_nonlateralized_ct])
memberships = celltype_analyzer.memberships()
celltype_analyzer.plot_memberships('interhemisphere/plots/signal-lateralization_by-celltype.pdf', figsize=(4,2))

# %%
