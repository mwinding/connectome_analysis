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
matrix.columns = pd.to_numeric(matrix.columns)

matrix = matrix_ad + matrix_dd + matrix_aa + matrix_da
matrix_axon = matrix_aa + matrix_da

# import pair list CSV, manually generated
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)
paired = pairs.values.flatten()

# %%
# identify skids of brain input neurons and 2nd order brain inputs
rm = pymaid.CatmaidInstance(url, name, password, token)

# pull skids for 2nd order PNs and 3rd order neurons
PN_A00c = pymaid.get_skids_by_annotation('mw A00c 2nd_order PN')
PN_AN = pymaid.get_skids_by_annotation('mw AN 2nd_order PN')
PN_MN = pymaid.get_skids_by_annotation('mw MN 2nd_order PN')
PN_ORN = pymaid.get_skids_by_annotation('mw ORN 2nd_order PN')
PN_photo = pymaid.get_skids_by_annotation('mw photo 2nd_order PN')
PN_thermo = pymaid.get_skids_by_annotation('mw thermo 2nd_order PN')
PN_vtd = pymaid.get_skids_by_annotation('mw vtd 2nd_order PN')

order3 = pymaid.get_annotated('mw brain inputs 3rd_order')
order3.index = np.arange(0, len(order3['name']), 1) # reset indices
skids3 = []
for i in np.arange(0, len(order3), 1):
    skids_temp = order3['name'][i]
    skids_temp = pymaid.get_skids_by_annotation(skids_temp)
    skids3.append(skids_temp)

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

# checking 50% output (all to all) intragroup + a-a or d-a to input neurons
A00c3_mat = intragroup_connections2(matrix, matrix_axon, skids3[0], PN_A00c, outputs)
AN3_mat = intragroup_connections2(matrix, matrix_axon, skids3[1], PN_AN, outputs)
MN3_mat = intragroup_connections2(matrix, matrix_axon, skids3[2], PN_MN, outputs)
ORN3_mat = intragroup_connections2(matrix, matrix_axon, skids3[3], PN_ORN, outputs)
photo3_mat = intragroup_connections2(matrix, matrix_axon, skids3[4], PN_photo, outputs)
thermo3_mat = intragroup_connections2(matrix, matrix_axon, skids3[5], PN_thermo, outputs)
vtd3_mat = intragroup_connections2(matrix, matrix_axon, skids3[6], PN_vtd, outputs)

# %%
A00c3LN = A00c3_mat[A00c3_mat['average_output']>=0.5]
AN3LN = AN3_mat[AN3_mat['average_output']>=0.5]
MN3LN = MN3_mat[MN3_mat['average_output']>=0.5]
ORN3LN = ORN3_mat[ORN3_mat['average_output']>=0.5]
photo3LN = photo3_mat[photo3_mat['average_output']>=0.5]
thermo3LN = thermo3_mat[thermo3_mat['average_output']>=0.5]
vtd3LN = vtd3_mat[vtd3_mat['average_output']>=0.5]


# %%
# output CSVs of each putative LN for each sensory modality
A00c3LN.to_csv('identify_neuron_classes/csv/order3LN_A00c.csv')
AN3LN.to_csv('identify_neuron_classes/csv/order3LN_AN.csv')
MN3LN.to_csv('identify_neuron_classes/csv/order3LN_MN.csv')
ORN3LN.to_csv('identify_neuron_classes/csv/order3LN_ORN.csv')
photo3LN.to_csv('identify_neuron_classes/csv/order3LN_photo.csv')
thermo3LN.to_csv('identify_neuron_classes/csv/order3LN_thermo.csv')
vtd3LN.to_csv('identify_neuron_classes/csv/order3LN_vtd.csv')


# %%
