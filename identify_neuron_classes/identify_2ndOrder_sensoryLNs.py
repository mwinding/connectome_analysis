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

# import connector list CSV
connectors = pd.read_csv('data/connectors.csv', header=0, index_col=0)
#connectors = connectors[~connectors['connector_id'].duplicated(keep='first')] # remove duplicated presynaptic sites
skids = pd.unique(connectors['presynaptic_to'])

outputs = []
for skid in skids:
    temp = connectors[connectors['presynaptic_to'] == skid]
    #temp = temp[temp['presynaptic_type'] == 'axon']
    #temp = temp[temp['postsynaptic_type'] == 'dendrite']
    outputs.append([skid, len(temp['connector_id'])])

outputs = pd.DataFrame(outputs, columns = ['skeleton_ID', 'outputs'])

outputs_axon = []
for skid in skids:
    temp = connectors[connectors['presynaptic_to'] == skid]
    temp = temp[temp['presynaptic_type'] == 'axon']
    #temp = temp[temp['postsynaptic_type'] == 'dendrite']
    outputs_axon.append([skid, len(temp['connector_id'])])

outputs_axon = pd.DataFrame(outputs, columns = ['skeleton_ID', 'outputs'])

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

def intragroup_connections(matrix, skids, sens_skids, outputs, pairs = pairs, sort = True):
    mat = matrix.loc[skids, skids + sens_skids]
    mat = mat.sum(axis=1)

    # convert to % outputs
    for i in np.arange(0, len(mat.index), 1):
        axon_output = outputs.loc[outputs['skeleton_ID']==mat.index[i], 'outputs'].values
        if(axon_output != 0):
            mat.loc[mat.index[i]] = mat.loc[mat.index[i]]/axon_output

    if(sort):
        mat = sortPairs(mat, pairs)

    return(mat)

# checking 50% output (all to all) intragroup
ORN2_mat = intragroup_connections(matrix, sens2_skids[0], sens_skids[0], outputs)
thermo2_mat = intragroup_connections(matrix, sens2_skids[1], sens_skids[1], outputs)
photo2_mat = intragroup_connections(matrix, sens2_skids[2], sens_skids[2], outputs)
AN2_mat = intragroup_connections(matrix, sens2_skids[3], sens_skids[3], outputs)
MN2_mat = intragroup_connections(matrix, sens2_skids[4], sens_skids[4], outputs)
AN2_MN2_mat = intragroup_connections(matrix, np.unique(sens2_skids[3] + sens2_skids[4]).tolist(), np.unique(sens_skids[3] + sens_skids[4]).tolist(), outputs)
ORN_AN2_MN2_mat = intragroup_connections(matrix, np.unique(sens2_skids[0] + sens2_skids[3] + sens2_skids[4]).tolist(), np.unique(sens_skids[0] + sens_skids[3] + sens_skids[4]).tolist(), outputs)
vtd2_mat = intragroup_connections(matrix, sens2_skids[5], sens_skids[5], outputs)
A00c2_mat = intragroup_connections(matrix, sens2_skids[6], sens_skids[6], outputs)

# checking 50% output from axon intragroup
ORN2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[0], sens_skids[0], outputs_axon)
thermo2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[1], sens_skids[1], outputs_axon)
photo2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[2], sens_skids[2], outputs_axon)
AN2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[3], sens_skids[3], outputs_axon)
MN2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[4], sens_skids[4], outputs_axon)
AN2_MN2_mat_ad = intragroup_connections(matrix_ad, np.unique(sens2_skids[3] + sens2_skids[4]).tolist(), np.unique(sens_skids[3] + sens_skids[4]).tolist(), outputs_axon)
ORN_AN2_MN2_mat_ad = intragroup_connections(matrix_ad, np.unique(sens2_skids[0] + sens2_skids[3] + sens2_skids[4]).tolist(), np.unique(sens_skids[0] + sens_skids[3] + sens_skids[4]).tolist(), outputs_axon)
vtd2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[5], sens_skids[5], outputs_axon)
A00c2_mat_ad = intragroup_connections(matrix_ad, sens2_skids[6], sens_skids[6], outputs_axon)

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
ORN2_mat.loc[ORN2_mat>=0.5].to_csv('identify_neuron_classes/csv/ORN_2o_LN.csv')
ORN2_mat.loc[ORN2_mat<0.5].to_csv('identify_neuron_classes/csv/ORN_2o_nonLN.csv')

thermo2_mat.loc[thermo2_mat>=0.5].to_csv('identify_neuron_classes/csv/thermo_2o_LN.csv')
thermo2_mat.loc[thermo2_mat<0.5].to_csv('identify_neuron_classes/csv/thermo_2o_nonLN.csv')

photo2_mat.loc[photo2_mat>=0.5].to_csv('identify_neuron_classes/csv/photo_2o_LN.csv')
photo2_mat.loc[photo2_mat<0.5].to_csv('identify_neuron_classes/csv/photo_2o_nonLN.csv')

AN2_mat.loc[AN2_mat>=0.5].to_csv('identify_neuron_classes/csv/AN_2o_LN.csv')
AN2_mat.loc[AN2_mat<0.5].to_csv('identify_neuron_classes/csv/AN_2o_nonLN.csv')

MN2_mat.loc[MN2_mat>=0.5].to_csv('identify_neuron_classes/csv/MN_2o_LN.csv')
MN2_mat.loc[MN2_mat<0.5].to_csv('identify_neuron_classes/csv/MN_2o_nonLN.csv')

AN2_MN2_mat.loc[AN2_MN2_mat>=0.5].to_csv('identify_neuron_classes/csv/AN_MN_2o_LN.csv')
AN2_MN2_mat.loc[AN2_MN2_mat<0.5].to_csv('identify_neuron_classes/csv/AN_MN_2o_nonLN.csv')

ORN_AN2_MN2_mat.loc[ORN_AN2_MN2_mat>=0.5].to_csv('identify_neuron_classes/csv/ORN_AN_MN_2o_LN.csv')
ORN_AN2_MN2_mat.loc[ORN_AN2_MN2_mat<0.5].to_csv('identify_neuron_classes/csv/ORN_AN_MN_2o_nonLN.csv')

vtd2_mat.loc[vtd2_mat>=0.5].to_csv('identify_neuron_classes/csv/vtd_2o_LN.csv')
vtd2_mat.loc[vtd2_mat<0.5].to_csv('identify_neuron_classes/csv/vtd_2o_nonLN.csv')

A00c2_mat.loc[A00c2_mat>=0.5].to_csv('identify_neuron_classes/csv/A00c_2o_LN.csv')
A00c2_mat.loc[A00c2_mat<0.5].to_csv('identify_neuron_classes/csv/A00c_2o_nonLN.csv')
# %%
sns.distplot(ORN_AN2_MN2_mat, bins = 50)

# %%
sns.distplot(ORN2_mat, bins = 50)


# %%
