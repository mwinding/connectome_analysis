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
import connectome_tools.process_matrix as promat
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token

# convert pair-sorted brain/sensories matrix to binary matrix based on synapse threshold
matrix_ad = pd.read_csv('data/axon-dendrite.csv', header=0, index_col=0)
matrix_dd = pd.read_csv('data/dendrite-dendrite.csv', header=0, index_col=0)
matrix_aa = pd.read_csv('data/axon-axon.csv', header=0, index_col=0)
matrix_da = pd.read_csv('data/dendrite-axon.csv', header=0, index_col=0)

# the columns are string by default and the indices int; now both are int
matrix_ad.columns = pd.to_numeric(matrix_ad.columns)
matrix_dd.columns = pd.to_numeric(matrix_dd.columns)
matrix_aa.columns = pd.to_numeric(matrix_aa.columns)
matrix_da.columns = pd.to_numeric(matrix_da.columns)

matrix = matrix_ad + matrix_dd + matrix_aa + matrix_da
matrix_axon = matrix_aa + matrix_da


# import pair list CSV, manually generated
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)
paired = pairs.values.flatten()

# %%
# identify skids of brain input neurons and 2nd order brain inputs
rm = pymaid.CatmaidInstance(url, name, password, token)

# pull sensory annotations and then pull associated skids
sensories = pymaid.get_annotated('mw brain inputs')
sens_skids = []
for i in np.arange(0, len(sensories), 1):
    sens = sensories['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    sens_skids.append(sens)

sensories_2ndorder = pymaid.get_annotated('mw brain inputs 2nd_order')
sensories_2ndorder = sensories_2ndorder.iloc[[2, 3, 6, 0, 1, 4, 5],:] # reorder to match sensories
sensories_2ndorder.index = np.arange(0, len(sensories_2ndorder['name']), 1) # reset indices
sens2_skids = []
for i in np.arange(0, len(sensories_2ndorder), 1):
    sens2 = sensories_2ndorder['name'][i]
    sens2 = pymaid.get_skids_by_annotation(sens2)
    sens2_skids.append(sens2)

# %%
# connections among group members

outputs = pd.read_csv('data/mw_brain_matrix_skeleton_measurements.csv', header=0, index_col=0)

def sortPairs(mat, pairList):
    cols = ['leftid', 'rightid', 'leftid_output', 'rightid_output', 'average_output']
    summed_paired = []

    for i in range(0, len(pairList['leftid'])):
        if(pairList['leftid'][i] in mat.index):
            left_identifier = pairList['leftid'][i]
            left_sum = mat.loc[left_identifier]
        
            right_identifier = promat.identify_pair(pairList['leftid'][i], pairList)
            right_sum = mat.loc[right_identifier]
                
            summed_paired.append([left_identifier, right_identifier, left_sum, right_sum, (left_sum + right_sum)/2])

    summed_paired = pd.DataFrame(summed_paired, columns= cols)
    return(summed_paired)

def intragroup_connections(matrix, skids, input_skids, outputs, pairs = pairs, sort = True):
    mat = matrix.loc[skids, skids + input_skids]
    mat = mat.sum(axis=1)

    # convert to % outputs
    for i in np.arange(0, len(mat.index), 1):
        output = outputs.loc[outputs['Skeleton']==mat.index[i], 'N outputs'].values
        if(output != 0):
            mat.loc[mat.index[i]] = mat.loc[mat.index[i]]/output

    if(sort):
        mat = sortPairs(mat, pairs)

    return(mat)

def intragroup_connections2(matrix, matrix_axon, skids, input_skids, outputs, pairs = pairs, sort = True):
    mat = matrix.loc[skids, skids]
    mat = mat.sum(axis=1)

    mat_axon = matrix_axon.loc[skids, input_skids]
    mat_axon = mat_axon.sum(axis=1)

    # convert to % outputs
    for i in np.arange(0, len(mat.index), 1):
        output = outputs.loc[outputs['Skeleton']==mat.index[i], 'N outputs'].values
        if(output != 0):
            mat.loc[mat.index[i]] = mat.loc[mat.index[i]]/output
            mat_axon.loc[mat.index[i]] = mat_axon.loc[mat.index[i]]/output

            #combining outputs to axons of this layer's input neurons and all->all connections intragroup
            mat_combo = mat + mat_axon

    if(sort):
        mat = sortPairs(mat_combo, pairs)

    return(mat)

# checking 50% output (all to all) intragroup
ORN2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[0], sens_skids[0], outputs)
thermo2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[1], sens_skids[1], outputs)
photo2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[2], sens_skids[2], outputs)
AN2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[3], sens_skids[3], outputs)
MN2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[4], sens_skids[4], outputs)
AN2_MN2_mat = intragroup_connections2(matrix, matrix_axon, np.unique(sens2_skids[3] + sens2_skids[4]).tolist(), np.unique(sens_skids[3] + sens_skids[4]).tolist(), outputs)
ORN_AN2_MN2_mat = intragroup_connections2(matrix, matrix_axon, np.unique(sens2_skids[0] + sens2_skids[3] + sens2_skids[4]).tolist(), np.unique(sens_skids[0] + sens_skids[3] + sens_skids[4]).tolist(), outputs)
vtd2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[5], sens_skids[5], outputs)
A00c2_mat = intragroup_connections2(matrix, matrix_axon, sens2_skids[6], sens_skids[6], outputs)

# %%
ORN2LN = ORN2_mat[ORN2_mat['average_output']>=0.5]
thermo2LN = thermo2_mat[thermo2_mat['average_output']>=0.5]
photo2LN = photo2_mat[photo2_mat['average_output']>=0.5]
AN2LN = AN2_mat[AN2_mat['average_output']>=0.5]
MN2LN = MN2_mat[MN2_mat['average_output']>=0.5]
vtd2LN = vtd2_mat[vtd2_mat['average_output']>=0.5]
A00c2LN = A00c2_mat[A00c2_mat['average_output']>=0.5]
AN2_MN2LN = AN2_MN2_mat[AN2_MN2_mat['average_output']>=0.5]
ORN2_AN2_MN2LN = ORN_AN2_MN2_mat[ORN_AN2_MN2_mat['average_output']>=0.5]

# %%
# output CSVs of each putative LN and non-LN for each sensory modality
ORN2LN.to_csv('identify_neuron_classes/csv/order2LN_ORN.csv')
thermo2LN.to_csv('identify_neuron_classes/csv/order2LN_thermo.csv')
photo2LN.to_csv('identify_neuron_classes/csv/order2LN_photo.csv')
AN2LN.to_csv('identify_neuron_classes/csv/order2LN_AN.csv')
MN2LN.to_csv('identify_neuron_classes/csv/order2LN_MN.csv')
vtd2LN.to_csv('identify_neuron_classes/csv/order2LN_vtd.csv')
A00c2LN.to_csv('identify_neuron_classes/csv/order2LN_A00c.csv')
AN2_MN2LN.to_csv('identify_neuron_classes/csv/order2LN_AN_MN.csv')
ORN2_AN2_MN2LN.to_csv('identify_neuron_classes/csv/order2LN_ORN_AN_MN.csv')

# %%
sns.distplot(ORN2_mat['average_output'], bins = 50)


# %%
print('Percent LNs per modality (2nd-order)\nORN: %f\nthermo: %f\nphoto: %f\nAN: %f\nMN: %f\nvtd2: %f\nA00c2: %f\n'
    %(len(ORN2LN)/len(ORN2_mat)*100,
    len(thermo2LN)/len(thermo2_mat)*100,
    len(photo2LN)/len(photo2_mat)*100,
    len(AN2LN)/len(AN2_mat)*100,
    len(MN2LN)/len(MN2_mat)*100,
    len(vtd2LN)/len(vtd2_mat)*100,
    len(A00c2LN)/len(A00c2_mat)*100))

print('Number of ORN2_AN2_MN2 local neurons: %i' %len(ORN2_AN2_MN2LN))
# %%