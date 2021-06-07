#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random

import cmasher as cmr

import connectome_tools.process_matrix as pm

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

adj = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')

# load inputs and pair data
inputs = pd.read_csv('data/graphs/inputs.csv', index_col = 0)
pairs = pm.Promat.get_pairs()

# load cluster data
clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)

# separate meta file with median_node_visits from sensory for each node
# determined using iterative random walks
meta_with_order = pd.read_csv('data/deprecated/meta_data_w_order.csv', index_col = 0, header = 0)

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

order_df = []
for key in lvl7.groups:
    skids = lvl7.groups[key]
    node_visits = meta_with_order.loc[skids, :].median_node_visits
    order_df.append([key, np.nanmean(node_visits)])

order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
order_df = order_df.sort_values(by = 'node_visit_order')

order = list(order_df.cluster)

# %%
# number of contra/ipsi edges and synapses
# just creates confusion, leave it out 

ipsi_axon = pymaid.get_skids_by_annotation('mw ipsilateral axon')
contra_axon = pymaid.get_skids_by_annotation('mw contralateral axon')
bilateral_axon = pymaid.get_skids_by_annotation('mw bilateral axon')
ipsi_dendrite = pymaid.get_skids_by_annotation('mw ipsilateral dendrite')
contra_dendrite = pymaid.get_skids_by_annotation('mw contralateral dendrite')
bilateral_dendrite = pymaid.get_skids_by_annotation('mw bilateral dendrite')

ipsi_ipsi = list(np.intersect1d(ipsi_dendrite, ipsi_axon))
ipsi_bilateral = list(np.intersect1d(ipsi_dendrite, bilateral_axon))
ipsi_contra = list(np.intersect1d(ipsi_dendrite, contra_axon))
bilateral_ipsi = list(np.intersect1d(bilateral_dendrite, ipsi_axon))
bilateral_bilateral = list(np.intersect1d(bilateral_dendrite, bilateral_axon))
bilateral_contra = list(np.intersect1d(bilateral_dendrite, contra_axon))
contra_ipsi = list(np.intersect1d(contra_dendrite, ipsi_axon))
contra_bilateral = list(np.intersect1d(contra_dendrite, bilateral_axon))
contra_contra = list(np.intersect1d(contra_dendrite, contra_axon))
weird_bilaterals = bilateral_ipsi + bilateral_bilateral + bilateral_contra

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')
br = pymaid.get_skids_by_annotation('mw brain neurons')

contra_contra_left = np.intersect1d(left, np.intersect1d(contra_dendrite, contra_axon))
contra_contra_right = np.intersect1d(right, np.intersect1d(contra_dendrite, contra_axon))

left = np.setdiff1d(left, list(contra_contra_left) + weird_bilaterals)
left = list(left) + list(contra_contra_right)
left = list(np.intersect1d(left, adj.index))
left = list(np.intersect1d(left, br))

right = np.setdiff1d(right, list(contra_contra_right) + weird_bilaterals)
right = list(right) + list(contra_contra_left)
right = list(np.intersect1d(right, adj.index))
right = list(np.intersect1d(right, br))

ipsi_synapses = sum(sum(adj.loc[left, left].values)) + sum(sum(adj.loc[right, right].values))
contra_synapses = sum(sum(adj.loc[left, right].values)) + sum(sum(adj.loc[right, left].values))

ipsi_neuron_ipsi_synapses = sum(sum(adj.loc[np.intersect1d(ipsi_ipsi, left), left].values)) + sum(sum(adj.loc[np.intersect1d(ipsi_ipsi, right), right].values))
ipsi_neuron_contra_synapses = sum(sum(adj.loc[np.intersect1d(ipsi_ipsi, left), right].values)) + sum(sum(adj.loc[np.intersect1d(ipsi_ipsi, right), left].values))

bilateral_neuron_ipsi_synapses = sum(sum(adj.loc[bilateral_left, left].values)) + sum(sum(adj.loc[bilateral_right, right].values))
bilateral_neuron_contra_synapses = sum(sum(adj.loc[bilateral_left, right].values)) + sum(sum(adj.loc[bilateral_right, left].values))

contra_neuron_ipsi_synapses = sum(sum(adj.loc[contra_left, left].values)) + sum(sum(adj.loc[contra_right, right].values))
contra_neuron_contra_synapses = sum(sum(adj.loc[contra_left, right].values)) + sum(sum(adj.loc[contra_right, left].values))

# %%
# contra/bilateral character plot

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')
br = pymaid.get_skids_by_annotation('mw brain neurons')

projectome = pd.read_csv('data/projectome/projectome_mw brain paper all neurons_split.csv', index_col = 0, header = 0)

is_left = []
for skid in projectome.skeleton:
    if(skid in left):
        is_left.append([skid, 1])
    if(skid in right):
        is_left.append([skid, 0])
    if((skid not in right) & (skid not in left)):
        is_left.append([skid, -1])
is_left = pd.DataFrame(is_left, columns = ['skid', 'is_left'])

projectome['is_left']=is_left.is_left.values
proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['Brain Hemisphere left', 'Brain Hemisphere right'].sum()

right_contra_axon_outputs = proj_group.loc[(br, 0, 1, 0), :] # right side, axon outputs
left_contra_axon_outputs = proj_group.loc[(br, 1, 1, 0), :] # left side, axon outputs

right_contra_axon_outputs['ratio'] = right_contra_axon_outputs['Brain Hemisphere left']/(right_contra_axon_outputs['Brain Hemisphere left'] + right_contra_axon_outputs['Brain Hemisphere right'])
left_contra_axon_outputs['ratio'] = left_contra_axon_outputs['Brain Hemisphere right']/(left_contra_axon_outputs['Brain Hemisphere left'] + left_contra_axon_outputs['Brain Hemisphere right'])

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(right_contra_axon_outputs.ratio, ax=ax, kde=False, bins=20)
sns.distplot(left_contra_axon_outputs.ratio, ax=ax, kde=False, bins=20, color='gray')
ax.set(yticks=[], xticks=[])
fig.savefig('interhemisphere/plots/Br_axon-output_ipsi-vs-contralateral.pdf', format='pdf', bbox_inches='tight')

right_den_inputs = proj_group.loc[(br, 0, 0, 1), :] # right side, dendrite inputs
left_den_inputs = proj_group.loc[(br, 1, 0, 1), :] # left side, dendrite inputs

right_den_inputs['ratio'] = right_den_inputs['Brain Hemisphere left']/(right_den_inputs['Brain Hemisphere left'] + right_den_inputs['Brain Hemisphere right'])
left_den_inputs['ratio'] = left_den_inputs['Brain Hemisphere right']/(left_den_inputs['Brain Hemisphere left'] + left_den_inputs['Brain Hemisphere right'])

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(right_den_inputs.ratio, ax=ax, kde=False, bins=20)
sns.distplot(left_den_inputs.ratio, ax=ax, kde=False, bins=20, color='gray')
ax.set(yticks=[], xticks=[])
ax.set_yscale('log')
fig.savefig('interhemisphere/plots/Br_dendrite-input_ipsi-vs-contralateral.pdf', format='pdf', bbox_inches='tight')

# check these neurons
'''
right_den_inputs[right_den_inputs.sort_values(by='ratio', ascending=False).ratio>0]
left_den_inputs[left_den_inputs.sort_values(by='ratio', ascending=False).ratio>0]

left_asym_dendrite = [x[0] for x in right_den_inputs[right_den_inputs.sort_values(by='ratio', ascending=False).ratio>0].index]
right_asym_dendrite = [x[0] for x in left_den_inputs[left_den_inputs.sort_values(by='ratio', ascending=False).ratio>0].index]

pymaid.add_annotations(left_asym_dendrite, 'mw check dendrite')
pymaid.add_annotations(right_asym_dendrite, 'mw check dendrite')
'''
# same analysis with A1 for comparison
meshes_left = ['Brain Hemisphere left', 'SEZ_left', 'T1_left', 'T2_left', 'T3_left', 'A1_left', 'A2_left', 'A3_left', 'A4_left', 'A5_left', 'A6_left', 'A7_left', 'A8_left']
meshes_right = ['Brain Hemisphere right', 'SEZ_right', 'T1_right', 'T2_right', 'T3_right', 'A1_right', 'A2_right', 'A3_right', 'A4_right', 'A5_right', 'A6_right', 'A7_right', 'A8_right']
meshes = meshes_left + meshes_right

projectome['summed_left'] = projectome[meshes_left].sum(axis=1)
projectome['summed_right'] = projectome[meshes_right].sum(axis=1)
proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['summed_left', 'summed_right'].sum()

A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1_local = list(np.setdiff1d(A1, A1_ascending))
right_contra_axon_outputs = proj_group.loc[(A1, 0, 1, 0), :] # right side, axon outputs
left_contra_axon_outputs = proj_group.loc[(A1, 1, 1, 0), :] # left side, axon outputs

right_contra_axon_outputs['ratio'] = right_contra_axon_outputs['summed_left']/(right_contra_axon_outputs['summed_left'] + right_contra_axon_outputs['summed_right'])
left_contra_axon_outputs['ratio'] = left_contra_axon_outputs['summed_right']/(left_contra_axon_outputs['summed_left'] + left_contra_axon_outputs['summed_right'])

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(right_contra_axon_outputs.ratio, ax=ax, kde=False, bins=20)
sns.distplot(left_contra_axon_outputs.ratio, ax=ax, kde=False, bins=20, color='gray')
ax.set(yticks=[], xticks=[])
fig.savefig('interhemisphere/plots/A1_axon-output_ipsi-vs-contralateral.pdf', format='pdf', bbox_inches='tight')

right_den_inputs = proj_group.loc[(A1, 0, 0, 1), :] # right side, dendrite inputs
left_den_inputs = proj_group.loc[(A1, 1, 0, 1), :] # left side, dendrite inputs

right_den_inputs['ratio'] = right_den_inputs['summed_left']/(right_den_inputs['summed_left'] + right_den_inputs['summed_right'])
left_den_inputs['ratio'] = left_den_inputs['summed_right']/(left_den_inputs['summed_left'] + left_den_inputs['summed_right'])

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(right_den_inputs.ratio, ax=ax, kde=False, bins=20)
sns.distplot(left_den_inputs.ratio, ax=ax, kde=False, bins=20, color='gray')
ax.set(yticks=[], xticks=[])
ax.set_yscale('log')
fig.savefig('interhemisphere/plots/A1_dendrite-input_ipsi-vs-contralateral.pdf', format='pdf', bbox_inches='tight')
# %%
# pair-wise contra/bilateral character
# use this to proofread pairs and define all neurons as ipsi-, bi-, or contralateral

meshes_left = ['Brain Hemisphere left', 'SEZ_left', 'T1_left', 'T2_left', 'T3_left', 'A1_left', 'A2_left', 'A3_left', 'A4_left', 'A5_left', 'A6_left', 'A7_left', 'A8_left']
meshes_right = ['Brain Hemisphere right', 'SEZ_right', 'T1_right', 'T2_right', 'T3_right', 'A1_right', 'A2_right', 'A3_right', 'A4_right', 'A5_right', 'A6_right', 'A7_right', 'A8_right']
meshes = meshes_left + meshes_right

projectome['summed_left'] = projectome[meshes_left].sum(axis=1)
projectome['summed_right'] = projectome[meshes_right].sum(axis=1)
proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['summed_left', 'summed_right'].sum()
#proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['Brain Hemisphere left', 'Brain Hemisphere right'].sum()
br_pairs = pm.Promat.extract_pairs_from_list(br, pairs)[0]

right_contra_axon_outputs = proj_group.loc[(br, 0, 1, 0), :] # right side, axon outputs
left_contra_axon_outputs = proj_group.loc[(br, 1, 1, 0), :] # left side, axon outputs

right_contra_axon_outputs['ratio'] = right_contra_axon_outputs['summed_left']/(right_contra_axon_outputs['summed_left'] + right_contra_axon_outputs['summed_right'])
left_contra_axon_outputs['ratio'] = left_contra_axon_outputs['summed_right']/(left_contra_axon_outputs['summed_left'] + left_contra_axon_outputs['summed_right'])

br_pairs = pm.Promat.extract_pairs_from_list(br, pairs)[0]

contra_pairs = []
for i in range(len(br_pairs)):
    if((br_pairs.leftid[i] in [x[0] for x in left_contra_axon_outputs.index]) & (br_pairs.rightid[i] in [x[0] for x in right_contra_axon_outputs.index])):
        ratio_l = left_contra_axon_outputs.loc[br_pairs.leftid[i], 'ratio'].values[0]
        ratio_r = right_contra_axon_outputs.loc[br_pairs.rightid[i], 'ratio'].values[0]
        contra_pairs.append([br_pairs.leftid[i], br_pairs.rightid[i], ratio_l, ratio_r, (ratio_l+ratio_r)/2])

contra_pairs = pd.DataFrame(contra_pairs, columns = ['leftid', 'rightid', 'ratio_left', 'ratio_right', 'ratio_aver'])

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(contra_pairs.ratio_aver, ax=ax, kde=False, bins=20)
ax.set(yticks=[], xticks=[])
#ax.set_yscale('log')
fig.savefig('interhemisphere/plots/pairs_Br_axon-output_ipsi-vs-contralateral.pdf', format='pdf', bbox_inches='tight')


right_den_inputs = proj_group.loc[(br, 0, 0, 1), :] # right side, dendrite inputs
left_den_inputs = proj_group.loc[(br, 1, 0, 1), :] # left side, dendrite inputs

right_den_inputs['ratio'] = right_den_inputs['summed_left']/(right_den_inputs['summed_left'] + right_den_inputs['summed_right'])
left_den_inputs['ratio'] = left_den_inputs['summed_right']/(left_den_inputs['summed_left'] + left_den_inputs['summed_right'])

br_pairs = pm.Promat.extract_pairs_from_list(br, pairs)[0]

dend_pairs = []
for i in range(len(br_pairs)):
    if((br_pairs.leftid[i] in [x[0] for x in left_den_inputs.index]) & (br_pairs.rightid[i] in [x[0] for x in right_den_inputs.index])):
        ratio_l = left_den_inputs.loc[br_pairs.leftid[i], 'ratio'].values[0]
        ratio_r = right_den_inputs.loc[br_pairs.rightid[i], 'ratio'].values[0]
        dend_pairs.append([br_pairs.leftid[i], br_pairs.rightid[i], ratio_l, ratio_r, (ratio_l+ratio_r)/2])

dend_pairs = pd.DataFrame(dend_pairs, columns = ['leftid', 'rightid', 'ratio_left', 'ratio_right', 'ratio_aver'])
#pymaid.add_annotations(list(dend_pairs[dend_pairs.ratio_aver==0].leftid.values), 'mw ipsilateral dendrite')
#pymaid.add_annotations(list(dend_pairs[dend_pairs.ratio_aver==0].rightid.values), 'mw ipsilateral dendrite')

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(dend_pairs.ratio_aver, ax=ax, kde=False, bins=20)
ax.set(yticks=[], xticks=[])
ax.set_yscale('log')
fig.savefig('interhemisphere/plots/pairs_Br_dendrite_input_ipsi-vs-contralateral.pdf', format='pdf', bbox_inches='tight')
'''
# check neurons on the edge of 1 and 0
almost_contra = (contra_pairs.ratio_aver>0.8) & (contra_pairs.ratio_aver<1.0)
almost_contra_skids = list(contra_pairs[almost_contra].leftid.values) + list(contra_pairs[almost_contra].rightid.values)
pymaid.add_annotations(almost_contra_skids, 'mw check contra')

almost_ipsi = (contra_pairs.ratio_aver>0.0) & (contra_pairs.ratio_aver<0.35)
almost_ipsi = (contra_pairs.ratio_aver==0.0)
almost_ipsi_skids = np.intersect1d(list(contra_pairs[almost_ipsi].leftid.values) + list(contra_pairs[almost_ipsi].rightid.values), contra)
pymaid.add_annotations(almost_ipsi_skids, 'mw check ipsi')
'''
# %%
# annotate ipsilateral, bilateral, and contralateral axons
# remove neurons with "mw brain few synapses"
'''
ipsi_annotated = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral_annotated = pymaid.get_skids_by_annotation('mw bilateral axon')
contra_annotated = pymaid.get_skids_by_annotation('mw contralateral axon')
few_synapses = pymaid.get_skids_by_annotation('mw brain few synapses')

# exclude skids that were manually annotated or are only partially differentiated
excluded = ipsi_annotated + bilateral_annotated + contra_annotated + few_synapses

ipsi_skids = list(contra_pairs[contra_pairs.ratio_aver==0].leftid) + list(contra_pairs[contra_pairs.ratio_aver==0].rightid)
contra_skids = list(contra_pairs[contra_pairs.ratio_aver==1].leftid) + list(contra_pairs[contra_pairs.ratio_aver==1].rightid)
bilateral_skids = list(contra_pairs[(contra_pairs.ratio_aver<1) & (contra_pairs.ratio_aver>0)].leftid) + list(contra_pairs[(contra_pairs.ratio_aver<1) & (contra_pairs.ratio_aver>0)].rightid)

ipsi_skids = list(np.setdiff1d(ipsi_skids, excluded))
contra_skids = list(np.setdiff1d(contra_skids, excluded))
bilateral_skids = list(np.setdiff1d(bilateral_skids, excluded))

ipsi_skids_pair = Promat.extract_pairs_from_list(ipsi_skids, pairs)[0]
contra_skids_pair = Promat.extract_pairs_from_list(contra_skids, pairs)[0]
bilateral_skids_pair = Promat.extract_pairs_from_list(bilateral_skids, pairs)[0]

contra_pairs.index = contra_pairs.leftid
sns.distplot(contra_pairs.loc[ipsi_skids_pair.leftid].ratio_aver, kde=False, bins=20)
sns.distplot(contra_pairs.loc[bilateral_skids_pair.leftid].ratio_aver, kde=False, bins=20)
sns.distplot(contra_pairs.loc[contra_skids_pair.leftid].ratio_aver, kde=False, bins=20)

pymaid.add_annotations(ipsi_skids, 'mw ipsilateral axon')
pymaid.add_annotations(contra_skids, 'mw contralateral axon')
pymaid.add_annotations(bilateral_skids, 'mw bilateral axon')
'''
# %%
# plot ipsi, bi, contra axons
ipsi_annotated = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral_annotated = pymaid.get_skids_by_annotation('mw bilateral axon')
contra_annotated = pymaid.get_skids_by_annotation('mw contralateral axon')

ipsi_skids_pair = pm.Promat.extract_pairs_from_list(ipsi_annotated, pairs)[0]
contra_skids_pair = pm.Promat.extract_pairs_from_list(contra_annotated, pairs)[0]
bilateral_skids_pair = pm.Promat.extract_pairs_from_list(bilateral_annotated, pairs)[0]

contra_pairs.index = contra_pairs.leftid

# set up bins
max_val = 1
step = max_val/10
max_val = max_val + step + step/2
bins = [x for x in np.arange(0, max_val, step-step/2)]

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(contra_pairs.loc[np.intersect1d(ipsi_skids_pair.leftid.values, contra_pairs.index)].ratio_aver, kde=False, bins=bins, ax=ax)
sns.distplot(contra_pairs.loc[np.intersect1d(bilateral_skids_pair.leftid.values, contra_pairs.index)].ratio_aver, kde=False, bins=bins, ax=ax)
sns.distplot(contra_pairs.loc[np.intersect1d(contra_skids_pair.leftid.values, contra_pairs.index)].ratio_aver, kde=False, bins=bins, ax=ax)
ax.set(yticks=[])

fig, ax = plt.subplots(1,1, figsize=(1,2))
ax.hist(contra_pairs.loc[np.intersect1d(ipsi_skids_pair.leftid.values, contra_pairs.index)].ratio_aver, bins=bins)
ax.hist(contra_pairs.loc[np.intersect1d(bilateral_skids_pair.leftid.values, contra_pairs.index)].ratio_aver, bins=bins)
ax.hist(contra_pairs.loc[np.intersect1d(contra_skids_pair.leftid.values, contra_pairs.index)].ratio_aver, bins=bins)
fig.savefig('interhemisphere/plots/pairs_Br_axon-output_ipsi-vs-contralateral_colored.pdf', format='pdf', bbox_inches='tight')

# %%
# plot ipsi, bi, contra dendrites
ipsi_annotated = pymaid.get_skids_by_annotation('mw ipsilateral dendrite')
bilateral_annotated = pymaid.get_skids_by_annotation('mw bilateral dendrite')
contra_annotated = pymaid.get_skids_by_annotation('mw contralateral dendrite')

ipsi_skids_pair = pm.Promat.extract_pairs_from_list(ipsi_annotated, pairs)[0]
contra_skids_pair = pm.Promat.extract_pairs_from_list(contra_annotated, pairs)[0]
bilateral_skids_pair = pm.Promat.extract_pairs_from_list(bilateral_annotated, pairs)[0]

dend_pairs.index = dend_pairs.leftid

# set up bins
max_val = 1
step = max_val/10
max_val = max_val + step + step/2
bins = [x for x in np.arange(0, max_val, step-step/2)]

fig, ax = plt.subplots(1,1, figsize=(1,2))
sns.distplot(dend_pairs.loc[np.intersect1d(ipsi_skids_pair.leftid.values, dend_pairs.index)].ratio_aver, kde=False, bins=bins, ax=ax)
sns.distplot(dend_pairs.loc[np.intersect1d(bilateral_skids_pair.leftid.values, dend_pairs.index)].ratio_aver, kde=False, bins=bins, ax=ax)
sns.distplot(dend_pairs.loc[np.intersect1d(contra_skids_pair.leftid.values, dend_pairs.index)].ratio_aver, kde=False, bins=bins, ax=ax)
ax.set(yticks=[])

fig, ax = plt.subplots(1,1, figsize=(1,2))
ax.hist(dend_pairs.loc[np.intersect1d(ipsi_skids_pair.leftid.values, dend_pairs.index)].ratio_aver, bins=bins)
ax.hist(dend_pairs.loc[np.intersect1d(bilateral_skids_pair.leftid.values, dend_pairs.index)].ratio_aver, bins=bins)
ax.hist(dend_pairs.loc[np.intersect1d(contra_skids_pair.leftid.values, dend_pairs.index)].ratio_aver, bins=bins)
ax.set_yscale('log')
fig.savefig('interhemisphere/plots/pairs_Br_dendrite-input_ipsi-vs-contralateral_colored.pdf', format='pdf', bbox_inches='tight')

# %%
# ipsi/contra/bilateral per cluster

ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')
bi = pymaid.get_skids_by_annotation('mw bilateral axon')

# integration types per cluster
cluster_lvl7 = [[key, lvl7.groups[key].values] for key in lvl7.groups.keys()]
cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'skids'])
cluster_lvl7.set_index('key', inplace=True)

ipsi_contra_clusters = []
for key in cluster_lvl7.index:
    ipsi_sum = len(np.intersect1d(cluster_lvl7.loc[key].skids, ipsi))#/len(cluster_lvl7.loc[key].skids)
    contra_sum = len(np.intersect1d(cluster_lvl7.loc[key].skids, contra))#/len(cluster_lvl7.loc[key].skids)
    bi_sum = len(np.intersect1d(cluster_lvl7.loc[key].skids, bi))#/len(cluster_lvl7.loc[key].skids)

    if((ipsi_sum + contra_sum + bi_sum)>0):
        ipsi_frac = ipsi_sum/(ipsi_sum + contra_sum + bi_sum)
        contra_frac = contra_sum/(ipsi_sum + contra_sum + bi_sum)
        bi_frac = bi_sum/(ipsi_sum + contra_sum + bi_sum)
        ipsi_contra_clusters.append([key, ipsi_frac, bi_frac, contra_frac])
    else:
        ipsi_contra_clusters.append([key, 0, 0, 0])

ipsi_contra_clusters = pd.DataFrame(ipsi_contra_clusters, columns = ['key', 'ipsi', 'bi', 'contra'])
ipsi_contra_clusters.set_index('key', inplace=True)
ipsi_contra_clusters = ipsi_contra_clusters.loc[order, :]

ind = [x for x in range(0, len(cluster_lvl7))]
fig, ax = plt.subplots(1,1, figsize=(3,2))
plt.bar(ind, ipsi_contra_clusters.ipsi.values)
plt.bar(ind, ipsi_contra_clusters.bi.values, bottom = ipsi_contra_clusters.ipsi)
plt.bar(ind, ipsi_contra_clusters.contra.values, bottom = ipsi_contra_clusters.ipsi + ipsi_contra_clusters.bi)
fig.savefig('interhemisphere/plots/ipsi_contra_makeup_clusters.pdf', format='pdf', bbox_inches='tight')
# %%
# amount of ipsi/contra per cell type
ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')
bi = pymaid.get_skids_by_annotation('mw bilateral axon')

# set cell types
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
br = pymaid.get_skids_by_annotation('mw brain neurons')
MBON = pymaid.get_skids_by_annotation('mw MBON')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')
LHN = pymaid.get_skids_by_annotation('mw LHN')
CN = pymaid.get_skids_by_annotation('mw CN')
KC = pymaid.get_skids_by_annotation('mw KC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC 1%')
RGN = pymaid.get_skids_by_annotation('mw RGN')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
uPN = pymaid.get_skids_by_annotation('mw uPN')
tPN = pymaid.get_skids_by_annotation('mw tPN')
vPN = pymaid.get_skids_by_annotation('mw vPN')
mPN = pymaid.get_skids_by_annotation('mw mPN')
PN = uPN + tPN + vPN + mPN
FBN = pymaid.get_skids_by_annotation('mw FBN')
FB2N = pymaid.get_skids_by_annotation('mw FB2N')
FBN_all = FBN + FB2N

input_names = pymaid.get_annotated('mw brain inputs').name
input_skids_list = list(map(pymaid.get_skids_by_annotation, input_names))
sens_all = [x for sublist in input_skids_list for x in sublist]
A00c = pymaid.get_skids_by_annotation('mw A00c')

asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')
asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
asc_all = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')

LHN = list(np.setdiff1d(LHN, FBN_all))
CN = list(np.setdiff1d(CN, LHN + FBN_all)) # 'CN' means exclusive CNs that are not FBN or LHN
dSEZ = list(np.setdiff1d(dSEZ, MBON + MBIN + LHN + CN + KC + dVNC + PN + FBN_all))
pre_dVNC = list(np.setdiff1d(pre_dVNC, MBON + MBIN + LHN + CN + KC + dSEZ + dVNC + PN + FBN_all + asc_all + sens_all + A00c)) # 'pre_dVNC' must have no other category assignment

few_synapses = pymaid.get_skids_by_annotation('mw brain few synapses')

celltypes = [list(np.setdiff1d(br, RGN+few_synapses)), sens_all, PN, LHN, MBIN, list(np.setdiff1d(KC, few_synapses)), MBON, FBN_all, CN, pre_dVNC, dSEZ, dVNC]
celltype_names = ['Total', 'Sens', 'PN', 'LHN', 'MBIN', 'KC', 'MBON', 'MB-FBN', 'CN', 'pre-dVNC', 'dSEZ', 'dVNC']

ipsi_contra_celltypes = []
unknown_list = []
for i, celltype in enumerate(celltypes):
    ipsi_sum = len(np.intersect1d(celltype, ipsi))
    contra_sum = len(np.intersect1d(celltype, contra))
    bi_sum = len(np.intersect1d(celltype, bi))

    ipsi_frac = ipsi_sum/len(celltype)
    contra_frac = contra_sum/len(celltype)
    bi_frac = bi_sum/len(celltype)

    ipsi_contra_celltypes.append([celltype_names[i], ipsi_frac, bi_frac, contra_frac])

    unknown = np.setdiff1d(celltype, (ipsi+contra+bi))
    unknown_list.append(unknown)

ipsi_contra_celltypes = pd.DataFrame(ipsi_contra_celltypes, columns = ['celltype', 'ipsi', 'bi', 'contra'])

ind = [x for x in range(0, len(ipsi_contra_celltypes))]
fig, ax = plt.subplots(1,1, figsize=(2,2))
plt.bar(ind, ipsi_contra_celltypes.ipsi.values)
plt.bar(ind, ipsi_contra_celltypes.bi.values, bottom = ipsi_contra_celltypes.ipsi)
plt.bar(ind, ipsi_contra_celltypes.contra.values, bottom = ipsi_contra_celltypes.ipsi+ipsi_contra_celltypes.bi)
plt.xticks(rotation=45, ha='right')
ax.set(xticklabels = celltype_names, xticks=np.arange(0, len(ipsi_contra_celltypes), 1))
fig.savefig('interhemisphere/plots/ipsi_contra_makeup_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# different types of cell types in each ipsi, bi, contra neuron types

import connectome_tools.cascade_analysis as casc
import connectome_tools.process_matrix as pm
import pymaid as pymaid

ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

celltypes, celltype_names = pm.Promat.celltypes()
all_celltypes = list(map(lambda pair: casc.Celltype(pair[0], pair[1]), zip(celltype_names, celltypes)))

all_cats = [casc.Celltype('ipsi', ipsi), casc.Celltype('bilateral', bilateral), casc.Celltype('contra', contra)]
all_cats_analyzer = casc.Celltype_Analyzer(all_cats)
all_cats_analyzer.set_known_types(all_celltypes)
cats_memberships = all_cats_analyzer.memberships()

fraction_types_names = cats_memberships.index
colors = ['#00753F', '#1D79B7', '#D4E29E', '#FF8734', '#E55560', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC', '#E0B1AD', 'tab:purple','#D88052', '#A52A2A', 'tab:grey']
#plt.bar(x=fraction_types_names,height=[1]*len(colors),color=colors)

plts=[]
fig, ax = plt.subplots(figsize=(0.3,1.2))
plt1 = plt.bar(cats_memberships.columns, cats_memberships.iloc[0, :], color=colors[0])
bottom = cats_memberships.iloc[0, :]
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(0,1))
plts.append(plt1)

for i in range(1, len(cats_memberships.iloc[:, 0])):
    plt_next = plt.bar(cats_memberships.columns, cats_memberships.iloc[i, :], bottom = bottom, color = colors[i])
    bottom = bottom + cats_memberships.iloc[i, :]
    plts.append(plt_next)
    ax.set(ylim=(0,1))
    plt.xticks(rotation=45, ha='right')

plt.savefig(f'interhemisphere/plots/ipsi_bi_contra_identities/ipsi_bi_contra_identities.pdf', format='pdf', bbox_inches='tight')

# %%
