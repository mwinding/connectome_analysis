#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 6})

rm = pymaid.CatmaidInstance(url, token, name, password)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_sens = [val for sublist in list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw A1 sensories').name)) for val in sublist]
brain_only = np.setdiff1d(adj.index.values, A1 + A1_sens)

adj = adj.loc[brain_only, brain_only]
#adj = mg.adj  # adjacency matrix from the "mg" object

# pull skids of different output types
output_order = [1, 0, 2]
output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

output_names_reordered = [output_names[i] for i in output_order]
output_skids_list_reordered = [output_skids_list[i] for i in output_order]

def skid_to_index(skid, mg):
    index_match = np.where(mg.meta.index == skid)[0]
    if(len(index_match)==1):
        return(index_match[0])
    if(len(index_match)!=1):
        print('Not one match for skid %i!' %skid)

# convert skids to indices
output_indices_list = []
for skids in output_skids_list_reordered:
    indices = []
    for skid in skids:
        index = skid_to_index(skid, mg)
        indices.append(index)
    output_indices_list.append(indices)
# %%
# identify pre-descending types
# defined as 50% outputs to a particular descending type
# doesn't work well (no neurons with high fraction output to descending neurons)

# ** is this normal? perhaps test with random sample of neuron same size as descendings; started at bottom

dVNC_mat = pd.DataFrame(adj[:, output_indices_list[0]], index = mg.meta.index, columns = output_skids_list_reordered[0])
dSEZ_mat = pd.DataFrame(adj[:, output_indices_list[1]], index = mg.meta.index, columns = output_skids_list_reordered[1])
RG_mat = pd.DataFrame(adj[:, output_indices_list[2]], index = mg.meta.index, columns = output_skids_list_reordered[2])

# number of outputs per neuron
# how many go to output types

outputs = pd.read_csv('data/mw_brain_matrix_skeleton_measurements.csv', header=0, index_col=0)

dVNC_mat_sum = dVNC_mat.sum(axis = 1)
dSEZ_mat_sum = dSEZ_mat.sum(axis = 1)
RG_mat_sum = RG_mat.sum(axis = 1)

fraction_output_dVNC = []
fraction_output_dSEZ = []
fraction_output_RG = []

# determine fraction of output from each brain neuron to each brain output type
for i in range(0, len(dVNC_mat_sum)):
    output_total = outputs[outputs.Skeleton == dVNC_mat_sum.index[i]]['N outputs'].values[0]
    if(output_total > 0):
        fraction_output_dVNC.append(dVNC_mat_sum.iloc[i]/output_total)
        fraction_output_dSEZ.append(dSEZ_mat_sum.iloc[i]/output_total)
        fraction_output_RG.append(RG_mat_sum.iloc[i]/output_total)

    if(output_total == 0):
        fraction_output_dVNC.append(0)
        fraction_output_dSEZ.append(0)
        fraction_output_RG.append(0)

# convert to np arrays
fraction_output_dVNC = np.array(fraction_output_dVNC)
fraction_output_dSEZ = np.array(fraction_output_dSEZ)
fraction_output_RG = np.array(fraction_output_RG)

fig, ax = plt.subplots(1,1,figsize=(5,5))
sns.distplot(fraction_output_dVNC[fraction_output_dVNC>0], hist = False, ax = ax)
sns.distplot(fraction_output_dSEZ[fraction_output_dSEZ>0], hist = False, ax = ax)
sns.distplot(fraction_output_RG[fraction_output_RG>0], hist = False, ax = ax)
ax.set(xlabel = 'Fraction output to dVNC(blue), dSEZ(orange), or RG(green)')
ax.get_yaxis().set_visible(False)

plt.savefig('cascades/feedback_through_brain/plots/output_fraction_to_descendings.pdf', format='pdf', bbox_inches='tight')
'''
import numpy.random as random

rand_mat_list = []
for i in range(0, 2):
    random.seed(i)
    random_indices = random.choice(len(dVNC_mat_sum), 183, replace = False)
    rand_mat = pd.DataFrame(adj[:, random_indices], index = mg.meta.index, columns = random_indices)
    rand_mat_sum = rand_mat.sum(axis = 1)

    fraction_rand_mat = []
    for j in range(0, len(rand_mat_sum)):
        output_total = outputs[outputs.Skeleton == rand_mat_sum.index[j]]['N outputs'].values[0]
        if(output_total > 0):
            fraction_rand_mat.append(rand_mat_sum.iloc[j]/output_total)
        if(output_total == 0):
            fraction_rand_mat.append(0)

    rand_mat_list.append(np.array(fraction_rand_mat))
'''

# %%
# identify pre-descending types
# contributing 5% input to a particular descending neuron
# old method, using the new one below
from connectome_tools.process_matrix import Promat

def skid_to_index(skid, mg):
    index_match = np.where(mg.meta.index == skid)[0]
    if(len(index_match)==1):
        return(index_match[0])
    if(len(index_match)!=1):
        print('Not one match for skid %i!' %skid)

def index_to_skid(index, mg):
    return(mg.meta.iloc[index, :].name)

# import pairs
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0)

dVNC_pairs = Promat.extract_pairs_from_list(output_skids_list_reordered[0], pairs)
dSEZ_pairs = Promat.extract_pairs_from_list(output_skids_list_reordered[1], pairs)
RG_pairs = Promat.extract_pairs_from_list(output_skids_list_reordered[2], pairs)
brain_skids_pairs = Promat.extract_pairs_from_list(mg.meta.index, pairs)

# left_right interlaced order for dVNC pairs
dVNC_pair_order = []
for i in range(0, len(dVNC_pairs)):
    dVNC_pair_order.append(dVNC_pairs.iloc[i].leftid)
    dVNC_pair_order.append(dVNC_pairs.iloc[i].rightid)

dSEZ_pair_order = []
for i in range(0, len(dSEZ_pairs)):
    dSEZ_pair_order.append(dSEZ_pairs.iloc[i].leftid)
    dSEZ_pair_order.append(dSEZ_pairs.iloc[i].rightid)

RG_pair_order = []
for i in range(0, len(RG_pairs)):
    RG_pair_order.append(RG_pairs.iloc[i].leftid)
    RG_pair_order.append(RG_pairs.iloc[i].rightid)

# left_right interlaced order for brain matrix
brain_pair_order = []
for i in range(0, len(brain_skids_pairs)):
    brain_pair_order.append(brain_skids_pairs.iloc[i].leftid)
    brain_pair_order.append(brain_skids_pairs.iloc[i].rightid)

interlaced_dVNC_mat = dVNC_mat.loc[brain_pair_order, dVNC_pair_order]
interlaced_dSEZ_mat = dSEZ_mat.loc[brain_pair_order, dSEZ_pair_order]
interlaced_RG_mat = RG_mat.loc[brain_pair_order, RG_pair_order]

# convert to %input of descendings' dendrite
for column in interlaced_dVNC_mat.columns:
    dendrite_input = mg.meta.loc[column].dendrite_input
    interlaced_dVNC_mat.loc[:, column] = interlaced_dVNC_mat.loc[:, column]/dendrite_input

for column in interlaced_dSEZ_mat.columns:
    dendrite_input = mg.meta.loc[column].dendrite_input
    interlaced_dSEZ_mat.loc[:, column] = interlaced_dSEZ_mat.loc[:, column]/dendrite_input

for column in interlaced_RG_mat.columns:
    dendrite_input = mg.meta.loc[column].dendrite_input
    interlaced_RG_mat.loc[:, column] = interlaced_RG_mat.loc[:, column]/dendrite_input

# summed input onto descending between pairs
oddCols_dVNC = np.arange(0, len(interlaced_dVNC_mat.columns), 2)
oddCols_dSEZ = np.arange(0, len(interlaced_dSEZ_mat.columns), 2)
oddCols_RG = np.arange(0, len(interlaced_RG_mat.columns), 2)

oddRows = np.arange(0, len(interlaced_dVNC_mat.index), 2)

# initializing summed matrices for each descending type
sumMat_dVNC = np.zeros(shape=(len(oddRows),len(oddCols_dVNC)))
sumMat_dVNC = pd.DataFrame(sumMat_dVNC, columns = interlaced_dVNC_mat.columns[oddCols_dVNC], index = interlaced_dVNC_mat.index[oddRows])

sumMat_dSEZ = np.zeros(shape=(len(oddRows),len(oddCols_dSEZ)))
sumMat_dSEZ = pd.DataFrame(sumMat_dVNC, columns = interlaced_dSEZ_mat.columns[oddCols_dSEZ], index = interlaced_dSEZ_mat.index[oddRows])

sumMat_RG = np.zeros(shape=(len(oddRows),len(oddCols_RG)))
sumMat_RG = pd.DataFrame(sumMat_RG, columns = interlaced_RG_mat.columns[oddCols_RG], index = interlaced_RG_mat.index[oddRows])

for i_iter, i in enumerate(oddRows):
    for j_iter, j in enumerate(oddCols_dVNC):
        summed_pairs = interlaced_dVNC_mat.iat[i, j] + interlaced_dVNC_mat.iat[i+1, j+1] + interlaced_dVNC_mat.iat[i+1, j] + interlaced_dVNC_mat.iat[i, j+1]
        sumMat_dVNC.iat[i_iter, j_iter] = summed_pairs/2

for i_iter, i in enumerate(oddRows):
    for j_iter, j in enumerate(oddCols_dSEZ):
        summed_pairs = interlaced_dSEZ_mat.iat[i, j] + interlaced_dSEZ_mat.iat[i+1, j+1] + interlaced_dSEZ_mat.iat[i+1, j] + interlaced_dSEZ_mat.iat[i, j+1]
        sumMat_dSEZ.iat[i_iter, j_iter] = summed_pairs/2

for i_iter, i in enumerate(oddRows):
    for j_iter, j in enumerate(oddCols_RG):
        summed_pairs = interlaced_RG_mat.iat[i, j] + interlaced_RG_mat.iat[i+1, j+1] + interlaced_RG_mat.iat[i+1, j] + interlaced_RG_mat.iat[i, j+1]
        sumMat_RG.iat[i_iter, j_iter] = summed_pairs/2

# %%
# identifying pre-descendings based on an individual pair of neurons contributing 1% input to descending neuron

from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

threshold = 0.01
inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RGN')

brain_adj = Adjacency_matrix(adj, adj.index, pairs, inputs,'axo-dendritic')

pre_dVNC, pre_dVNC_edges = brain_adj.upstream(dVNC, threshold, exclude = dVNC)
_, pre_dVNC = brain_adj.edge_threshold(pre_dVNC_edges, threshold, direction='upstream')

# compare to other cell types
MBON = pymaid.get_skids_by_annotation('mw MBON')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')
LHN = pymaid.get_skids_by_annotation('mw LHN')
CN = pymaid.get_skids_by_annotation('mw CN')
KC = pymaid.get_skids_by_annotation('mw KC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
uPN = pymaid.get_skids_by_annotation('mw uPN')
tPN = pymaid.get_skids_by_annotation('mw tPN')
vPN = pymaid.get_skids_by_annotation('mw vPN')
mPN = pymaid.get_skids_by_annotation('mw mPN')
PN = uPN + tPN + vPN + mPN
FBN = pymaid.get_skids_by_annotation('mw FBN')
FB2N = pymaid.get_skids_by_annotation('mw FB2N')
FBN_all = FBN + FB2N

CN = list(np.setdiff1d(CN, LHN + FBN_all)) # 'CN' means exclusive CNs that are not FBN or LHN
pre_dVNC2 = list(np.setdiff1d(pre_dVNC, MBON + MBIN + LHN + CN + KC + dSEZ + dVNC + PN + FBN_all)) # 'pre_dVNC' must have no other category assignment
#pymaid.add_annotations(pre_dVNC, 'mw pre-dVNC 1%')

pre_dSEZ, pre_dSEZ_edges = brain_adj.upstream(dSEZ, threshold, exclude = dSEZ)
_, pre_dSEZ = brain_adj.edge_threshold(pre_dSEZ_edges, threshold, direction='upstream')
#pymaid.add_annotations(pre_dSEZ, 'mw pre-dSEZ 1%')

pre_RGN, pre_RGN_edges = brain_adj.upstream(RGN, threshold, exclude = RGN)
_, pre_RGN = brain_adj.edge_threshold(pre_RGN_edges, threshold, direction='upstream')
#pymaid.add_annotations(pre_RGN, 'mw pre-RGN 1%')

# %%
# plotting number of connections to and from descendings
### *****align/bin issues here***** ####
### see upstream_MNs.py for solution ###

fig, axs = plt.subplots(
    3, 2, figsize=(2.5, 3)
)

fig.tight_layout(pad = 2.5)
threshold = 0.05 # average 5% input threshold
binwidth = 1
x_range = list(range(0, 11))
align = 'left'

ax = axs[0, 0]
count_per_us_neuron = (sumMat_dVNC.values>threshold).sum(axis=1)
data = count_per_us_neuron[count_per_us_neuron>0]
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Neuron pairs')
ax.set_xlabel('Connection(s) to dVNCs')
ax.set_xticks(x_range)
ax.set(xlim = (0.5, 10))

ax = axs[0, 1]
count_per_descending = (sumMat_dVNC.values>threshold).sum(axis=0)
data = count_per_descending
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('dVNC pairs')
ax.set_xlabel('Connection(s) received')
ax.set_xticks(x_range)
ax.set(xlim = (-0.5, 7))

ax = axs[1, 0]
count_per_us_neuron = (sumMat_dSEZ.values>threshold).sum(axis=1)
data = count_per_us_neuron[count_per_us_neuron>0]
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Neuron pairs')
ax.set_xlabel('Connection(s) to dSEZs')
ax.set_xticks(x_range)
ax.set(xlim = (0.5, 10))

ax = axs[1, 1]
count_per_descending = (sumMat_dSEZ.values>threshold).sum(axis=0)
data = count_per_descending
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('dSEZ pairs')
ax.set_xlabel('Connection(s) received')
ax.set_xticks(x_range)
ax.set(xlim = (-0.5, 7))

ax = axs[2, 0]
count_per_us_neuron = (sumMat_RG.values>threshold).sum(axis=1)
data = count_per_us_neuron[count_per_us_neuron>0]
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Neuron pairs')
ax.set_xlabel('Connection(s) to RGNs')
ax.set_xticks(x_range)
ax.set(xlim = (0.5, 10))

ax = axs[2, 1]
count_per_descending = (sumMat_RG.values>threshold).sum(axis=0)
data = count_per_descending
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('RGN pairs')
ax.set_xlabel('Connection(s) received')
ax.set_xticks(x_range)
ax.set(xlim = (-0.5, 7))

plt.savefig('cascades/feedback_through_brain/plots/connections_from_to_descendings_5percent_threshold.pdf', bbox_inches='tight', transparent = True)
# %%
# export pre-descending neurons
pre_dVNC = (sumMat_dVNC.values>threshold).sum(axis=1)>0
pre_dSEZ = (sumMat_dSEZ.values>threshold).sum(axis=1)>0
pre_RGN = (sumMat_RG.values>threshold).sum(axis=1)>0

# identify indices == True, i.e. left skids of pairs that are pre-descendings
indices_pre_dVNC = np.where(pre_dVNC)[0]
indices_pre_dSEZ = np.where(pre_dSEZ)[0]
indices_pre_RGN = np.where(pre_RGN)[0]

pre_dVNC_skidleft = [sumMat_dVNC.index[x] for x in indices_pre_dVNC]
pre_dSEZ_skidleft = [sumMat_dSEZ.index[x] for x in indices_pre_dSEZ]
pre_RGN_skidleft = [sumMat_RG.index[x] for x in indices_pre_RGN]

# select pre-descending pair skids
brain_skids_pairs.index = brain_skids_pairs.leftid
pre_dVNC_skids = brain_skids_pairs.loc[pre_dVNC_skidleft, :]
pre_dSEZ_skids = brain_skids_pairs.loc[pre_dSEZ_skidleft, :]
pre_RGN_skids = brain_skids_pairs.loc[pre_RGN_skidleft, :]

#pre_dVNC_skids.to_csv('cascades/feedback_through_brain/plots/pre_dVNC_skids.csv', index = False)
#pre_dSEZ_skids.to_csv('cascades/feedback_through_brain/plots/pre_dSEZ_skids.csv', index = False)
#pre_RGN_skids.to_csv('cascades/feedback_through_brain/plots/pre_RGN_skids.csv', index = False)

# %%
# plot connectivity matrices of pre-output to output
from tqdm import tqdm

pre_dVNC_dSEZ_RGN = list(np.intersect1d(np.intersect1d(pre_dVNC_skidleft, pre_dSEZ_skidleft), pre_RGN_skidleft))
pre_dVNC_dSEZ = list(np.setdiff1d(np.intersect1d(pre_dVNC_skidleft, pre_dSEZ_skidleft), pre_dVNC_dSEZ_RGN))
pre_dVNC_RGN = list(np.setdiff1d(np.intersect1d(pre_dVNC_skidleft, pre_RGN_skidleft), pre_dVNC_dSEZ_RGN))
pre_dSEZ_RGN = list(np.setdiff1d(np.intersect1d(pre_dSEZ_skidleft, pre_RGN_skidleft), pre_dVNC_dSEZ_RGN))
combos = pre_dVNC_dSEZ_RGN + pre_dVNC_dSEZ + pre_dVNC_RGN + pre_dSEZ_RGN
pre_dVNC = list(np.setdiff1d(pre_dVNC_skidleft, combos))
pre_dSEZ = list(np.setdiff1d(pre_dSEZ_skidleft, combos))
pre_RGN = list(np.setdiff1d(pre_RGN_skidleft, combos))

output_mat = pd.concat([sumMat_dVNC, sumMat_dSEZ, sumMat_RG], axis = 1)

plt.savefig('cascades/feedback_through_brain/plots/preoutput_to_output.pdf')
# full interlaced adj matrix, summed pairs
# FUTURE: add colored bars to side of matrix to indicate cell type

interlaced_mat = pd.DataFrame(adj, index = mg.meta.index, columns = mg.meta.index)
interlaced_mat = interlaced_mat.loc[brain_pair_order, brain_pair_order]

# convert to %input
for column in interlaced_mat.columns:
    dendrite_input = mg.meta.loc[column].dendrite_input
    if(dendrite_input>0):
        interlaced_mat.loc[:, column] = interlaced_mat.loc[:, column]/dendrite_input
    if(dendrite_input==0):
        interlaced_mat.loc[:, column] = 0

oddRows = np.arange(0, len(interlaced_mat.index), 2)
oddCols = np.arange(0, len(interlaced_mat.columns), 2)

# summing partners
sumMat = np.zeros(shape=(len(oddRows),len(oddCols)))
sumMat = pd.DataFrame(sumMat, columns = interlaced_mat.columns[oddCols], index = interlaced_mat.index[oddRows])

for i_iter, i in tqdm(enumerate(oddRows)):
    for j_iter, j in enumerate(oddCols):
        summed_pairs = interlaced_mat.iloc[i, j] + interlaced_mat.iloc[i+1, j+1] + interlaced_mat.iloc[i+1, j] + interlaced_mat.iloc[i, j+1]
        sumMat.iat[i_iter, j_iter] = summed_pairs/2

sns.heatmap(sumMat.loc[(pre_dVNC + pre_dVNC_dSEZ + pre_dSEZ + pre_RGN + pre_dSEZ_RGN + pre_dVNC_RGN + pre_dVNC_dSEZ_RGN + list(dVNC_pairs.leftid) + list(dSEZ_pairs.leftid) + list(RG_pairs.leftid)),
            (pre_dVNC + pre_dVNC_dSEZ + pre_dSEZ + pre_RGN + pre_dSEZ_RGN + pre_dVNC_RGN + pre_dVNC_dSEZ_RGN + list(dVNC_pairs.leftid) + list(dSEZ_pairs.leftid) + list(RG_pairs.leftid))], 
            cmap = 'Blues', rasterized = True, vmax = 0.2, square = True, ax = ax)
plt.savefig('cascades/feedback_through_brain/plots/preoutput_output_adj_matrix.pdf')

fig, ax = plt.subplots(1,1,figsize=(6,5))
sns.heatmap(sumMat.loc[(pre_dVNC + pre_dVNC_dSEZ + pre_dSEZ + pre_RGN + pre_dSEZ_RGN + pre_dVNC_RGN + pre_dVNC_dSEZ_RGN),
            (list(dVNC_pairs.leftid) + list(dSEZ_pairs.leftid) + list(RG_pairs.leftid))], 
            cmap = 'Blues', rasterized = True, vmax = 0.2, ax = ax)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('cascades/feedback_through_brain/plots/preoutput_to_output_adj_matrix.pdf')

fig, ax = plt.subplots(1,1,figsize=(6,5))
sns.heatmap(sumMat.loc[(list(dVNC_pairs.leftid) + list(dSEZ_pairs.leftid) + list(RG_pairs.leftid)),
            (pre_dVNC + pre_dVNC_dSEZ + pre_dSEZ + pre_RGN + pre_dSEZ_RGN + pre_dVNC_RGN + pre_dVNC_dSEZ_RGN)], 
            cmap = 'Blues', rasterized = True, vmax = 0.2, square = True, ax = ax)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('cascades/feedback_through_brain/plots/output_to_preoutput_adj_matrix.pdf')
# %%
# downstream of outputs (by connectivity)

ds_dVNCs = sumMat.loc[dVNC_pairs.leftid, :]
ds_dSEZs = sumMat.loc[dSEZ_pairs.leftid, :]
ds_RGNs = sumMat.loc[RG_pairs.leftid, :]

meta_with_order = pd.read_csv('data/meta_data_w_order.csv', index_col = 0, header = 0)

# level 7 clusters
clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)
lvl7 = clusters.groupby('lvl7_labels')

# order of clusters
order_df = []
for key in lvl7.groups:
    skids = lvl7.groups[key]
    node_visits = meta_with_order.loc[skids, :].median_node_visits
    order_df.append([key, np.nanmean(node_visits)])

order_df = pd.DataFrame(order_df, columns = ['cluster', 'node_visit_order'])
order_df = order_df.sort_values(by = 'node_visit_order')

order = list(order_df.cluster)

# getting skids of each cluster
cluster_lvl7 = []
for key in order:
    cluster_lvl7.append(lvl7.groups[key].values)

# order skids within groups
cluster_lvl7_indices_list = []
sorted_skids = []
for skids in cluster_lvl7:
    skids_median_visit = meta_with_order.loc[skids, 'median_node_visits']
    skids_sorted = skids_median_visit.sort_values().index

    indices = []
    for skid in skids_sorted:
        index = skid_to_index(skid, mg)
        indices.append(index)
    cluster_lvl7_indices_list.append(indices)
    sorted_skids.append(skids_sorted)

# delist
sorted_skids = [val for sublist in sorted_skids for val in sublist]
sorted_skids_left = list(np.intersect1d(sumMat.columns, sorted_skids))

import cmasher as cmr

fig, axs = plt.subplots(
    3, 1, figsize = (4, 1.25)
)
ax = axs[0]
ax.imshow(ds_dVNCs.loc[:, sorted_skids_left], cmap = 'Blues', interpolation = 'none', vmax = 0.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Individual Brain Neurons')
ax.set_ylabel('dVNCs')

ax = axs[1]
ax.imshow(ds_dSEZs.loc[:, sorted_skids_left], cmap = 'Blues', interpolation = 'none', vmax = 0.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('dSEZs')

ax = axs[2]
ax.imshow(ds_RGNs.loc[:, sorted_skids_left], cmap = 'Blues', interpolation = 'none', vmax = 0.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('RGNs')

plt.savefig('cascades/feedback_through_brain/plots/downstream_outputs_connectivity_matrix.pdf', bbox_inches='tight')


# plot connectivity from all outputs
summed_ds_outputs = pd.DataFrame([ds_dVNCs.loc[:, sorted_skids_left].sum(axis=0), ds_dSEZs.loc[:, sorted_skids_left].sum(axis=0), ds_RGNs.loc[:, sorted_skids_left].sum(axis=0)])

fig, axs = plt.subplots(
    1, 1, figsize = (2, .5)
)
ax = axs
sns.heatmap(summed_ds_outputs, ax = ax, cmap = 'Blues', vmax = 0.4)
ax.set_title('Individual Brain Neurons')
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('cascades/feedback_through_brain/plots/downstream_outputs_summed_connectivity_matrix.pdf', bbox_inches='tight')

# %%
# plot connectivity between all pre-output types and outputs
output_skids_left = list(dVNC_pairs.leftid) + list(dSEZ_pairs.leftid) + list(RG_pairs.leftid)
summed_types = [list(sumMat.loc[pre_dVNC, output_skids_left].sum(axis = 0)),
            list(sumMat.loc[pre_dSEZ, output_skids_left].sum(axis = 0)),
            list(sumMat.loc[pre_RGN, output_skids_left].sum(axis = 0)),
            list(sumMat.loc[pre_dVNC_dSEZ, output_skids_left].sum(axis = 0)),
            list(sumMat.loc[pre_dSEZ_RGN, output_skids_left].sum(axis=0)),
            list(sumMat.loc[pre_dVNC_dSEZ_RGN, output_skids_left].sum(axis=0))]

summed_types = pd.DataFrame(summed_types, columns = output_skids_left, index = ['pre-dVNC', 'pre-dSEZ', 'pre-RGN', 'pre-dVNCs/dSEZ', 'pre-dSEZ/RGN', 'pre-all'])

fig, axs = plt.subplots(
    1, 1, figsize = (2, .9)
)
ax = axs
sns.heatmap(summed_types/2, ax = ax, cmap = 'Blues', vmax = 0.4)
ax.set_title('dVNCs dSEZs RGNs')
ax.set_xticks([])

plt.savefig('cascades/feedback_through_brain/plots/preoutputs_outputs_summed_connectivity_matrix.pdf', bbox_inches='tight')

# plot connectivity from pre-outputs to brain
non_output = list(np.intersect1d(sorted_skids_left, sumMat.columns.drop(output_skids_left)))

summed_types = [list(sumMat.loc[pre_dVNC, non_output].sum(axis = 0)),
            list(sumMat.loc[pre_dSEZ, non_output].sum(axis = 0)),
            list(sumMat.loc[pre_RGN, non_output].sum(axis = 0)),
            list(sumMat.loc[pre_dVNC_dSEZ, non_output].sum(axis = 0)),
            list(sumMat.loc[pre_dSEZ_RGN, non_output].sum(axis=0)),
            list(sumMat.loc[pre_dVNC_dSEZ_RGN, non_output].sum(axis=0))]

summed_types = pd.DataFrame(summed_types, columns = non_output, index = ['pre-dVNC', 'pre-dSEZ', 'pre-RGN', 'pre-dVNCs/dSEZ', 'pre-dSEZ/RGN', 'pre-all'])

fig, axs = plt.subplots(
    1, 1, figsize = (2, .9)
)
ax = axs
sns.heatmap(summed_types, ax = ax, cmap = 'Blues', vmax = 0.4)
ax.set_title('Non-output brain neurons')
ax.set_xticks([])

plt.savefig('cascades/feedback_through_brain/plots/preoutputs_to_brain_summed_connectivity_matrix.pdf', bbox_inches='tight')


# %%
# identify neurons ds of brain outputs

import connectome_tools.process_matrix as promat

threshold = 0.05 # %5 input threshold
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

source_dVNC = np.where((ds_dVNCs > threshold).sum(axis = 1))[0]
destination_dVNC = np.where((ds_dVNCs > threshold).sum(axis = 0))[0]
dVNC2 = pd.DataFrame([promat.get_paired_skids(x, pairs) for x in ds_dVNCs.columns[destination_dVNC]]
                    , columns = ['skid_left', 'skid_right'])
dVNC2.to_csv('cascades/feedback_through_brain/plots/dVNC_2nd_order.csv', index = False)

source_dSEZ = np.where((ds_dSEZs > threshold).sum(axis = 1))[0]
destination_dSEZ = np.where((ds_dSEZs > threshold).sum(axis = 0))[0]
dSEZ2 = pd.DataFrame([promat.get_paired_skids(x, pairs) for x in ds_dSEZs.columns[destination_dSEZ]]
                    , columns = ['skid_left', 'skid_right'])
dSEZ2.to_csv('cascades/feedback_through_brain/plots/dSEZ_2nd_order.csv', index = False)

source_RGN = np.where((ds_RGNs > threshold).sum(axis = 1))[0]
destination_RGN = np.where((ds_RGNs > threshold).sum(axis = 0))[0]

# %%
# identify neurons ds of pre-brain outputs

import connectome_tools.process_matrix as promat

threshold = 0.05 # %5 input threshold
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

pre_dVNC_CATMAID = pymaid.get_skids_by_annotation('mw pre-dVNC')
pre_dSEZ_CATMAID = pymaid.get_skids_by_annotation('mw pre-dSEZ')
pre_RGN_CATMAID = pymaid.get_skids_by_annotation('mw pre-RG')

pre_dVNC_CATMAID_pairs = promat.extract_pairs_from_list(pre_dVNC_CATMAID, pairs)
pre_dSEZ_CATMAID_pairs = promat.extract_pairs_from_list(pre_dSEZ_CATMAID, pairs)
pre_RGN_CATMAID_pairs = promat.extract_pairs_from_list(pre_RGN_CATMAID, pairs)

# generating downstream matrix from pre-outputs; already calculated as average percent input across pairs
ds_predVNCs = sumMat.loc[pre_dVNC_CATMAID_pairs.leftid, :]
ds_predSEZs = sumMat.loc[pre_dSEZ_CATMAID_pairs.leftid, :]
ds_preRGNs = sumMat.loc[pre_RGN_CATMAID_pairs.leftid, :]

# identify indices over threshold and pull associated skeleton IDs
destination_ds_predVNC = np.where((ds_predVNCs > threshold).sum(axis = 0)>0)[0]
destination_ds_predVNC = ds_predVNCs.columns[destination_ds_predVNC]
destination_ds_predVNC = np.setdiff1d(destination_ds_predVNC, (output_skids + pre_dVNC_CATMAID)) # exclude output neurons and source from ds_pre category

destination_ds_predSEZ = np.where((ds_predSEZs > threshold).sum(axis = 0)>0)[0]
destination_ds_predSEZ = ds_predSEZs.columns[destination_ds_predSEZ]
destination_ds_predSEZ = np.setdiff1d(destination_ds_predSEZ, (output_skids + pre_dSEZ_CATMAID)) # exclude output neurons and source from ds_pre category

destination_ds_preRGN = np.where((ds_preRGNs > threshold).sum(axis = 0)>0)[0]
destination_ds_preRGN = ds_preRGNs.columns[destination_ds_preRGN]
destination_ds_preRGN = np.setdiff1d(destination_ds_preRGN, (output_skids + pre_RGN_CATMAID)) # exclude output neurons and source from ds_pre category

ds_predVNC_skids = pd.DataFrame([promat.get_paired_skids(x, pairs) for x in destination_ds_predVNC]
                    , columns = ['skid_left', 'skid_right'])
ds_predVNC_skids.to_csv('cascades/feedback_through_brain/plots/ds_pre_dVNCs.csv', index = False)

ds_predSEZ_skids = pd.DataFrame([promat.get_paired_skids(x, pairs) for x in destination_ds_predSEZ]
                    , columns = ['skid_left', 'skid_right'])
ds_predSEZ_skids.to_csv('cascades/feedback_through_brain/plots/ds_pre_dSEZs.csv', index = False)

ds_preRGN_skids = pd.DataFrame([promat.get_paired_skids(x, pairs) for x in destination_ds_preRGN]
                    , columns = ['skid_left', 'skid_right'])
ds_preRGN_skids.to_csv('cascades/feedback_through_brain/plots/ds_pre_RGNs.csv', index = False)

# %%
