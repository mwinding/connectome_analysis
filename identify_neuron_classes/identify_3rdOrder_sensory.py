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

matrix = matrix_ad + matrix_dd + matrix_aa + matrix_da


# the columns are string by default and the indices int; now both are int
matrix_ad.columns = pd.to_numeric(matrix_ad.columns)
matrix_dd.columns = pd.to_numeric(matrix_dd.columns)
matrix_aa.columns = pd.to_numeric(matrix_aa.columns)
matrix_da.columns = pd.to_numeric(matrix_da.columns)
matrix.columns = pd.to_numeric(matrix.columns)

# import pair list CSV, manually generated
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)
paired = pairs.values.flatten()

# %%
rm = pymaid.CatmaidInstance(url, name, password, token)

# pull sensory annotations and then pull associated skids
order2 = pymaid.get_annotated('mw brain inputs 2nd_order')

# %%
# identifying neurons downstream of sensories based on percent-input
matrix_ad = pd.read_csv('data/axon-dendrite.csv', header=0, index_col=0)

# convert to %input
total_inputs = pd.read_csv('data/input_counts.csv', header = 0, index_col = 0)

for i in np.arange(0, len(matrix_ad.index), 1):
    inputs = total_inputs.loc[matrix_ad.index[i] == total_inputs.index, ' dendrite_inputs'].values
    if(inputs != 0):
        matrix_ad.loc[:, str(matrix_ad.index[i])] = matrix_ad.loc[:, str(matrix_ad.index[i])]/inputs

# %%
def summed_input(group_skids, matrix, pairList):
    submatrix = matrix.loc[group_skids, :]
    submatrix = submatrix.sum(axis = 0)
    submatrix.index = pd.to_numeric(submatrix.index)

    cols = ['leftid', 'rightid', 'leftid_input', 'rightid_input']
    summed_paired = []

    for i in range(0, len(pairList['leftid'])):
        if(pairList['leftid'][i] in submatrix.index):
            left_identifier = pairList['leftid'][i]
            left_sum = submatrix.loc[left_identifier]
        
            right_identifier = promat.identify_pair(pairList['leftid'][i], pairList)
            right_sum = submatrix.loc[right_identifier]
                
            summed_paired.append([left_identifier, right_identifier, left_sum, right_sum])

    summed_paired = pd.DataFrame(summed_paired, columns= cols)
    return(summed_paired)


order2_skids = []
for i in np.arange(0, len(order2), 1):
    sens = order2['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    order2_skids.append(sens)

sum_AN2 = summed_input(order2_skids[0], matrix_ad, pairs)
sum_MN2 = summed_input(order2_skids[1], matrix_ad, pairs)
sum_ORN2 = summed_input(order2_skids[2], matrix_ad, pairs)
sum_thermo2 = summed_input(order2_skids[3], matrix_ad, pairs)
sum_vtd2 = summed_input(order2_skids[4], matrix_ad, pairs)
sum_A00c2 = summed_input(order2_skids[5], matrix_ad, pairs)
sum_photo2 = summed_input(order2_skids[6], matrix_ad, pairs)

# %%
def identify_downstream(sum_df, summed_threshold, low_threshold):
    downstream = []
    for i in np.arange(0, len(sum_df['leftid']), 1):
        if((sum_df['leftid_input'].iloc[i] + sum_df['rightid_input'].iloc[i])>=summed_threshold):

            if(sum_df['leftid_input'].iloc[i]>sum_df['rightid_input'].iloc[i] and sum_df['rightid_input'].iloc[i]>=low_threshold):
                downstream.append(sum_df.iloc[i])

            if(sum_df['rightid_input'].iloc[i]>sum_df['leftid_input'].iloc[i] and sum_df['leftid_input'].iloc[i]>=low_threshold):
                downstream.append(sum_df.iloc[i])

        
    return(pd.DataFrame(downstream))


ORN_3o = identify_downstream(sum_ORN2, 0.1, 0.00001)
thermo_3o = identify_downstream(sum_thermo2, 0.1, 0.00001)
visual_3o = identify_downstream(sum_photo2, 0.1, 0.00001)
AN_3o = identify_downstream(sum_AN2, 0.1, 0.00001)
MN_3o = identify_downstream(sum_MN2, 0.1, 0.00001)
vtd_3o = identify_downstream(sum_vtd2, 0.1, 0.00001)
A00c_3o = identify_downstream(sum_A00c2, 0.1, 0.00001)


# %%
pd.DataFrame(thermo_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_thermo2.csv')
pd.DataFrame(visual_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_visual2.csv')
pd.DataFrame(vtd_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_vtd2.csv')
pd.DataFrame(A00c_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_A00c2.csv')

# the AN/MN/ORN groups are too big
# below I rerun using PNs only
# %%

# will look at 3rd order of the exclusive PNs for each modality and then the mixed PNs
ORN_PNonly = pymaid.get_skids_by_annotation("mw ORN only PN")
AN_PNonly = pymaid.get_skids_by_annotation("mw AN only PN")
MN_PNonly = pymaid.get_skids_by_annotation("mw MN only PN")

ORN_PN = pymaid.get_skids_by_annotation("mw ORN PN")
AN_PN = pymaid.get_skids_by_annotation("mw AN PN")
MN_PN = pymaid.get_skids_by_annotation("mw MN PN")
three_PNtypes = np.unique(ORN_PN + AN_PN + MN_PN).tolist()

# mixed PN skids
ORN_AN_PN = np.intersect1d(ORN_PN, AN_PN).tolist()
MN_AN_PN = np.intersect1d(MN_PN, AN_PN).tolist()
ORN_MN_PN = np.intersect1d(ORN_PN, MN_PN).tolist()
mixed_PN = np.unique(inter1 + inter2 + inter3).tolist()

sum_AN2 = summed_input(AN_PNonly, matrix_ad, pairs)
sum_MN2 = summed_input(MN_PNonly, matrix_ad, pairs)
sum_ORN2 = summed_input(ORN_PNonly, matrix_ad, pairs)
sum_AN_MN_ORN2 = summed_input(mixed_PN, matrix_ad, pairs)
sum_ORN_AN2 = summed_input(ORN_AN_PN, matrix_ad, pairs)
sum_MN_AN2 = summed_input(MN_AN_PN, matrix_ad, pairs)
sum_ORN_MN2 = summed_input(ORN_MN_PN, matrix_ad, pairs)


# identify downstream neurons
AN_3o = identify_downstream(sum_AN2, 0.1, 0.00001)
MN_3o = identify_downstream(sum_MN2, 0.1, 0.00001)
ORN_3o = identify_downstream(sum_ORN2, 0.1, 0.00001)
AN_MN_ORN_3o = identify_downstream(sum_AN_MN_ORN2, 0.1, 0.00001)
ORN_AN_3o = identify_downstream(sum_ORN_AN2, 0.1, 0.00001)
MN_AN_3o = identify_downstream(sum_MN_AN2, 0.1, 0.00001)
ORN_MN_3o = identify_downstream(sum_ORN_MN2, 0.1, 0.00001)

# %%
# output CSVs for mixed 3rd order neurons
pd.DataFrame(ORN_AN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ORN_AN_3o.csv')
pd.DataFrame(MN_AN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/MN_AN_3o.csv')
pd.DataFrame(ORN_MN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ORN_MN_3o.csv')
pd.DataFrame(AN_MN_ORN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/AN_MN_ORN_3o.csv')
pd.DataFrame(ORN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ORN_3o.csv')
pd.DataFrame(AN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/AN_3o.csv')
pd.DataFrame(MN_3o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/MN_3o.csv')

# %%
# check numbers of each type
AN_MN_ORN_3o_numbers = AN_MN_ORN_3o[['leftid', 'rightid']].values.flatten().tolist()
ORN_AN_3o_numbers = ORN_AN_3o[['leftid', 'rightid']].values.flatten().tolist()
MN_AN_3o_numbers = MN_AN_3o[['leftid', 'rightid']].values.flatten().tolist()
ORN_MN_3o_numbers = ORN_MN_3o[['leftid', 'rightid']].values.flatten().tolist()

numbers = np.unique(ORN_AN_3o_numbers + MN_AN_3o_numbers + ORN_MN_3o_numbers).tolist()

print('%i vs %i' %(len(AN_MN_ORN_3o_numbers), len(numbers)) )
# there are less when the PNs of the mixed type are divided
# %%
