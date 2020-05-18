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
order3 = pymaid.get_annotated('mw brain inputs 3rd_order')

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


order3_skids = []
for i in np.arange(0, len(order3), 1):
    sens = order3['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    order3_skids.append(sens)

sum_A00c3 = summed_input(order3_skids[0], matrix_ad, pairs)
sum_thermo3 = summed_input(order3_skids[1], matrix_ad, pairs)
sum_photo3 = summed_input(order3_skids[2], matrix_ad, pairs)
sum_vtd3 = summed_input(order3_skids[3], matrix_ad, pairs)
sum_ORN3 = summed_input(order3_skids[4], matrix_ad, pairs)
sum_MN3 = summed_input(order3_skids[5], matrix_ad, pairs)
sum_AN3 = summed_input(order3_skids[6], matrix_ad, pairs)
sum_AN_MN3 = summed_input(order3_skids[7], matrix_ad, pairs)
sum_ORN_AN3 = summed_input(order3_skids[8], matrix_ad, pairs)
sum_ORN_MN3 = summed_input(order3_skids[9], matrix_ad, pairs)

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

A00c_4o = identify_downstream(sum_A00c3, 0.1, 0.00001)
thermo_4o = identify_downstream(sum_thermo3, 0.1, 0.00001)
photo_4o = identify_downstream(sum_photo3, 0.1, 0.00001)
vtd_4o = identify_downstream(sum_vtd3, 0.1, 0.00001)
ORN_4o = identify_downstream(sum_ORN3, 0.1, 0.00001)
MN_4o = identify_downstream(sum_MN3, 0.1, 0.00001)
AN_4o = identify_downstream(sum_AN3, 0.1, 0.00001)
AN_MN_4o = identify_downstream(sum_AN_MN3, 0.1, 0.00001)
ORN_AN_4o = identify_downstream(sum_ORN_AN3, 0.1, 0.00001)
ORN_MN_4o = identify_downstream(sum_ORN_MN3, 0.1, 0.00001)

# %%
# investigate numbers of modalities

A00c4_skids = A00c_4o[['leftid', 'rightid']].values.flatten().tolist()
thermo4_skids = thermo_4o[['leftid', 'rightid']].values.flatten().tolist()
photo4_skids = photo_4o[['leftid', 'rightid']].values.flatten().tolist()
vtd4_skids = vtd_4o[['leftid', 'rightid']].values.flatten().tolist()
ORN4_skids = ORN_4o[['leftid', 'rightid']].values.flatten().tolist()
MN4_skids = MN_4o[['leftid', 'rightid']].values.flatten().tolist()
AN4_skids = AN_4o[['leftid', 'rightid']].values.flatten().tolist()
ANMN4_skids = AN_MN_4o[['leftid', 'rightid']].values.flatten().tolist()
ORNAN4_skids = ORN_AN_4o[['leftid', 'rightid']].values.flatten().tolist()
ORNMN4_skids = ORN_MN_4o[['leftid', 'rightid']].values.flatten().tolist()

mixed_skids = np.unique(ANMN4_skids + ORNAN4_skids + ORNMN4_skids).tolist()
all_skids = np.unique(A00c4_skids + thermo4_skids + photo4_skids + ORN4_skids + MN4_skids + AN4_skids + ANMN4_skids + ORNAN4_skids + ORNMN4_skids).tolist()

print('Number of skids per 4th-order input category\nA00c: %i\nthermo: %i\nphoto: %i\nvtd: %i\nORN: %i\nMN: %i\nAN: %i\nMixed: %i\nAll: %i' 
        %(len(A00c4_skids), len(thermo4_skids), len(photo4_skids), len(vtd4_skids), len(ORN4_skids), len(MN4_skids), len(AN4_skids), len(mixed_skids), len(all_skids)))
# %%
# output CSVs for 4th order neurons
pd.DataFrame(A00c4_skids).to_csv('identify_neuron_classes/csv/4o_A00c.csv')
pd.DataFrame(thermo4_skids).to_csv('identify_neuron_classes/csv/4o_thermo.csv')
pd.DataFrame(photo4_skids).to_csv('identify_neuron_classes/csv/4o_photo.csv')
pd.DataFrame(vtd4_skids).to_csv('identify_neuron_classes/csv/4o_vtd.csv')
pd.DataFrame(ORN4_skids).to_csv('identify_neuron_classes/csv/4o_ORN.csv')
pd.DataFrame(MN4_skids).to_csv('identify_neuron_classes/csv/4o_MN.csv')
pd.DataFrame(AN4_skids).to_csv('identify_neuron_classes/csv/4o_AN.csv')
pd.DataFrame(mixed_skids).to_csv('identify_neuron_classes/csv/4o_mixed_MN_AN_ORN.csv')

# %%
