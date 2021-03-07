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
rm = pymaid.CatmaidInstance(url, token, name, password)

# pull skids for 2nd order PNs and 3rd order neurons
PN_A00c = pymaid.get_skids_by_annotation('mw A00c 3rd_order PN')
PN_AN = pymaid.get_skids_by_annotation('mw AN 3rd_order PN')
PN_MN = pymaid.get_skids_by_annotation('mw MN 3rd_order PN')
PN_ORN = pymaid.get_skids_by_annotation('mw ORN 3rd_order PN')
PN_photo = pymaid.get_skids_by_annotation('mw photo 3rd_order PN')
PN_thermo = pymaid.get_skids_by_annotation('mw thermo 3rd_order PN')
PN_vtd = pymaid.get_skids_by_annotation('mw vtd 3rd_order PN')

order4 = pymaid.get_annotated('mw brain inputs 4th_order')
order4.index = np.arange(0, len(order4['name']), 1) # reset indices
skids4 = []
for i in np.arange(0, len(order4), 1):
    skids_temp = order4['name'][i]
    skids_temp = pymaid.get_skids_by_annotation(skids_temp)
    skids4.append(skids_temp)

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
A00c4_mat = intragroup_connections2(matrix, matrix_axon, skids4[0], PN_A00c, outputs)
AN4_mat = intragroup_connections2(matrix, matrix_axon, skids4[1], PN_AN, outputs)
MN4_mat = intragroup_connections2(matrix, matrix_axon, skids4[2], PN_MN, outputs)
ORN4_mat = intragroup_connections2(matrix, matrix_axon, skids4[3], PN_ORN, outputs)
photo4_mat = intragroup_connections2(matrix, matrix_axon, skids4[4], PN_photo, outputs)
thermo4_mat = intragroup_connections2(matrix, matrix_axon, skids4[5], PN_thermo, outputs)
vtd4_mat = intragroup_connections2(matrix, matrix_axon, skids4[6], PN_vtd, outputs)

# %%
A00c4LN = A00c4_mat[A00c4_mat['average_output']>=0.5]
AN4LN = AN4_mat[AN4_mat['average_output']>=0.5]
MN4LN = MN4_mat[MN4_mat['average_output']>=0.5]
ORN4LN = ORN4_mat[ORN4_mat['average_output']>=0.5]
photo4LN = photo4_mat[photo4_mat['average_output']>=0.5]
thermo4LN = thermo4_mat[thermo4_mat['average_output']>=0.5]
vtd4LN = vtd4_mat[vtd4_mat['average_output']>=0.5]


# %%
# output CSVs of each putative LN for each sensory modality
A00c4LN.to_csv('identify_neuron_classes/csv/order4LN_A00c.csv')
AN4LN.to_csv('identify_neuron_classes/csv/order4LN_AN.csv')
MN4LN.to_csv('identify_neuron_classes/csv/order4LN_MN.csv')
ORN4LN.to_csv('identify_neuron_classes/csv/order4LN_ORN.csv')
photo4LN.to_csv('identify_neuron_classes/csv/order4LN_photo.csv')
thermo4LN.to_csv('identify_neuron_classes/csv/order4LN_thermo.csv')
vtd4LN.to_csv('identify_neuron_classes/csv/order4LN_vtd.csv')

# %%
