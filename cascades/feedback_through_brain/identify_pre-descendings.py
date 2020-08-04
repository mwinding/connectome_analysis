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
#plt.rcParams.update({'font.size': 6})

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

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
import connectome_tools.process_matrix as promat

def skid_to_index(skid, mg):
    index_match = np.where(mg.meta.index == skid)[0]
    if(len(index_match)==1):
        return(index_match[0])
    if(len(index_match)!=1):
        print('Not one match for skid %i!' %skid)

def index_to_skid(index, mg):
    return(mg.meta.iloc[index, :].name)

# import pairs
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

dVNC_pairs = promat.extract_pairs_from_list(output_skids_list_reordered[0], pairs)
dSEZ_pairs = promat.extract_pairs_from_list(output_skids_list_reordered[1], pairs)
RG_pairs = promat.extract_pairs_from_list(output_skids_list_reordered[2], pairs)
brain_skids_pairs = promat.extract_pairs_from_list(mg.meta.index, pairs)

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
# plotting number of connections to and from descendings
fig, axs = plt.subplots(
    3, 2, figsize=(8, 8)
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
ax.set_ylabel('Number of brain neuron pairs')
ax.set_xlabel('Connection(s) to dVNCs')
ax.set_xticks(x_range)

ax = axs[0, 1]
count_per_descending = (sumMat_dVNC.values>threshold).sum(axis=0)
data = count_per_descending
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Number of dVNC neuron pairs')
ax.set_xlabel('Connection(s) received')
ax.set_xticks(x_range)

ax = axs[1, 0]
count_per_us_neuron = (sumMat_dSEZ.values>threshold).sum(axis=1)
data = count_per_us_neuron[count_per_us_neuron>0]
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Number of brain neuron pairs')
ax.set_xlabel('Connection(s) to dSEZs')
ax.set_xticks(x_range)

ax = axs[1, 1]
count_per_descending = (sumMat_dSEZ.values>threshold).sum(axis=0)
data = count_per_descending
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Number of dSEZ neuron pairs')
ax.set_xlabel('Connection(s) received')
ax.set_xticks(x_range)

ax = axs[2, 0]
count_per_us_neuron = (sumMat_RG.values>threshold).sum(axis=1)
data = count_per_us_neuron[count_per_us_neuron>0]
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Number of brain neuron pairs')
ax.set_xlabel('Connection(s) to RGNs')
ax.set_xticks(x_range)

ax = axs[2, 1]
count_per_descending = (sumMat_RG.values>threshold).sum(axis=0)
data = count_per_descending
ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), align = align)
ax.set_ylabel('Number of RGN pairs')
ax.set_xlabel('Connection(s) received')
ax.set_xticks(x_range)

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

pre_dVNC_skids.to_csv('cascades/feedback_through_brain/plots/pre_dVNC_skids.csv', index = False)
pre_dSEZ_skids.to_csv('cascades/feedback_through_brain/plots/pre_dSEZ_skids.csv', index = False)
pre_RGN_skids.to_csv('cascades/feedback_through_brain/plots/pre_RGN_skids.csv', index = False)

# %%
