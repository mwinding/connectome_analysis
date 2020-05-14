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
sensories = pymaid.get_annotated('mw brain inputs')

# %%
# signal coming from source_group, identify neurons with certain threshold
#source_group = sens
#threshold = 3
#matrix = matrix_ad

def downstream_search(matrix, source_group, threshold):
    downstream_neurons = []
    for i in np.arange(0, len(matrix.index), 1):

        # looking through all paired neurons
        if(matrix.index[i] in paired):

            partner_id =  promat.identify_pair(matrix.index[i], pairs)
            pair = [matrix.index[i], partner_id]
            #print(pair)
            
            print(pair)
            upstream = matrix.loc[source_group , pair]
            #downstream = matrix.loc[pair, source_group]

            #print(upstream)

            if(source_group not in paired):
                upstream_bin = upstream >= threshold
                #print(upstream_bin)
                if(sum(upstream_bin.loc[: ,pair[0]]) > 0 and sum(upstream_bin.loc[:, pair[1]] > 0)):
                    downstream_neurons.append(pair[0])
                    downstream_neurons.append(pair[1])
        
        
            # don't do anything different currently
            if(source_group in paired):
                upstream_bin = upstream >= threshold
                #print(upstream_bin)
                if(sum(upstream_bin.loc[: ,pair[0]]) > 0 and sum(upstream_bin.loc[:, pair[1]] > 0)):
                    downstream_neurons.append(pair[0])
                    downstream_neurons.append(pair[1])
                    
        # dealing with unpaired neurons
        if(matrix.index[i] not in paired):
            upstream = matrix.loc[source_group , matrix.index[i]]
            #downstream = matrix.loc[matrix.index[i], source_group]

            if(source_group not in paired):
                upstream_bin = upstream >= threshold
                if(sum(upstream_bin) > 0):
                    downstream_neurons.append(matrix.index[i])
        
            # don't do anything different currently
            if(source_group in paired):
                upstream_bin = upstream >= threshold
                if(sum(upstream_bin) > 0):
                    downstream_neurons.append(matrix.index[i])


    downstream_neurons = np.unique(downstream_neurons)

    return(downstream_neurons)

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


sens_skids = []
for i in np.arange(0, len(sensories), 1):
    sens = sensories['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    sens_skids.append(sens)

sum_ORN = summed_input(sens_skids[0], matrix_ad, pairs)
sum_thermo = summed_input(sens_skids[1], matrix_ad, pairs)
sum_visual = summed_input(sens_skids[2], matrix_ad, pairs)
sum_AN = summed_input(sens_skids[3], matrix_ad, pairs)
sum_MN = summed_input(sens_skids[4], matrix_ad, pairs)
sum_PaN = summed_input(sens_skids[5], matrix_ad, pairs)
sum_vtd = summed_input(sens_skids[6], matrix_ad, pairs)
sum_A00c = summed_input(sens_skids[7], matrix_ad, pairs)

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


ORN_2o = identify_downstream(sum_ORN, 0.1, 0.00001)
thermo_2o = identify_downstream(sum_thermo, 0.1, 0.00001)
visual_2o = identify_downstream(sum_visual, 0.1, 0.00001)
AN_2o = identify_downstream(sum_AN, 0.1, 0.00001)
MN_2o = identify_downstream(sum_MN, 0.1, 0.00001)
PaN_2o = identify_downstream(sum_PaN, 0.1, 0.00001)
vtd_2o = identify_downstream(sum_vtd, 0.1, 0.00001)
A00c_2o = identify_downstream(sum_A00c, 0.1, 0.00001)

pd.DataFrame(ORN_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_ORN.csv')
pd.DataFrame(thermo_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_thermo.csv')
pd.DataFrame(visual_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_visual.csv')
pd.DataFrame(AN_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_AN.csv')
pd.DataFrame(MN_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_MN.csv')
#pd.DataFrame(PaN_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_PaN.csv')
pd.DataFrame(vtd_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_vtd.csv')
pd.DataFrame(A00c_2o[['leftid', 'rightid']].values.flatten()).to_csv('identify_neuron_classes/csv/ds_A00c.csv')
print('finish csvs')
# %%
# identifying neurons downstream of sensories based on synapse-count
downstream_sensories = []
for i in np.arange(0, len(sensories), 1):
    sens = sensories['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    #print(sens)
    downstream_sens = downstream_search(matrix_ad, sens, 6)

    downstream_sensories.append(downstream_sens)

# %%
# outputing CSVs based on synapse-count threshold
pd.DataFrame(downstream_sensories[0]).to_csv('identify_neuron_classes/csv/ds_ORN.csv')
pd.DataFrame(downstream_sensories[1]).to_csv('identify_neuron_classes/csv/ds_thermo.csv')
pd.DataFrame(downstream_sensories[2]).to_csv('identify_neuron_classes/csv/ds_visual.csv')
pd.DataFrame(downstream_sensories[3]).to_csv('identify_neuron_classes/csv/ds_AN.csv')
pd.DataFrame(downstream_sensories[4]).to_csv('identify_neuron_classes/csv/ds_MN.csv')
pd.DataFrame(downstream_sensories[6]).to_csv('identify_neuron_classes/csv/ds_vtd.csv')
pd.DataFrame(downstream_sensories[7]).to_csv('identify_neuron_classes/csv/ds_A00c.csv')
