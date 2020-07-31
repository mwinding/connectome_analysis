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

# convert skids to indices
output_indices_list = []
for skids in output_skids_list_reordered:
    indices = np.where([x in skids for x in mg.meta.index])[0]
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
# top hits per descending neuron?

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

# left_right interlaced order for brain matrix
brain_pair_order = []
for i in range(0, len(brain_skids_pairs)):
    brain_pair_order.append(brain_skids_pairs.iloc[i].leftid)
    brain_pair_order.append(brain_skids_pairs.iloc[i].rightid)

interlaced_dVNC_mat = dVNC_mat.loc[brain_pair_order, dVNC_pair_order]


# summed input onto descending between pairs
oddCols = np.arange(0, len(interlaced_dVNC_mat.columns), 2)
oddRows = np.arange(0, len(interlaced_dVNC_mat.index), 2)

sumMat = np.zeros(shape=(len(oddRows),len(oddCols)))
sumMat = pd.DataFrame(sumMat, columns = interlaced_dVNC_mat.columns[oddCols], 
                            index = interlaced_dVNC_mat.index[oddRows])

for i_iter, i in enumerate(oddRows):
    for j_iter, j in enumerate(oddCols):
        summed_pairs = interlaced_dVNC_mat.iat[i, j] + interlaced_dVNC_mat.iat[i+1, j+1] + interlaced_dVNC_mat.iat[i+1, j] + interlaced_dVNC_mat.iat[i, j+1]
        sumMat.iat[i_iter, j_iter] = summed_pairs

# convert to %input of descendings' dendrite
for column in interlaced_dVNC_mat.columns:
    dendrite_input = mg.meta.loc[column].dendrite_input
    interlaced_dVNC_mat.loc[:, column] = interlaced_dVNC_mat.loc[:, column]/dendrite_input
# doesn't seem to work
'''
# converts a interlaced left-right pair adjacency matrix into a binary connection matrix based on some threshold
def binary_matrix(data, threshold, total_threshold): 

    oddCols = np.arange(0, len(data.columns), 2)
    oddRows = np.arange(0, len(data.index), 2)

    # column names are the skid of left neuron from pair
    binMat = np.zeros(shape=(len(oddRows),len(oddCols)))
    binMat = pd.DataFrame(binMat, columns = data.columns[oddCols], index = data.index[oddRows])

    for i in oddRows:
        for j in oddCols:
            sum_all = data.iat[i, j] + data.iat[i+1, j+1] + data.iat[i+1, j] + data.iat[i, j+1]
            if(data.iat[i, j] >= threshold and data.iat[i+1, j+1] >= threshold and sum_all >= total_threshold):
                binMat.iat[int(i/2), int(j/2)] = 1

            if(data.iat[i+1, j] >= threshold and data.iat[i, j+1] >= threshold and sum_all >= total_threshold):
                binMat.iat[int(i/2), int(j/2)] = 1
        
    return(binMat)

dVNC_connections = binary_matrix(interlaced_dVNC_mat, threshold = 3, total_threshold = 6)
'''

# add columns for adj indices
'''
indices_left = []
indices_right = []
for i in range(0, len(dVNC_pairs)):
    index_left = skid_to_index(dVNC_pairs.loc[i, 'leftid'], mg)
    index_right = skid_to_index(dVNC_pairs.loc[i, 'rightid'], mg)
    indices_left.append(index_left)
    indices_right.append(index_right)

dVNC_pairs['left_index'] = indices_left
dVNC_pairs['right_index'] = indices_right
'''


# %%
