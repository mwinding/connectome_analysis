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
rm = pymaid.CatmaidInstance(url, token, name, password)

# pull sensory annotations and then pull associated skids
order3 = pymaid.get_annotated('mw brain inputs 3rd_order PN')

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

    cols = ['leftid', 'rightid', 'leftid_input', 'rightid_input', 'average_input']
    summed_paired = []

    for i in range(0, len(pairList['leftid'])):
        if(pairList['leftid'][i] in submatrix.index):
            left_identifier = pairList['leftid'][i]
            left_sum = submatrix.loc[left_identifier]
        
            right_identifier = promat.identify_pair(pairList['leftid'][i], pairList)
            right_sum = submatrix.loc[right_identifier]
                
            summed_paired.append([left_identifier, right_identifier, left_sum, right_sum, (left_sum + right_sum)/2])

    summed_paired = pd.DataFrame(summed_paired, columns= cols)
    return(summed_paired)


order3_skids = []
for i in np.arange(0, len(order3), 1):
    sens = order3['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    order3_skids.append(sens)

sum_A00c3 = summed_input(order3_skids[0], matrix_ad, pairs)
sum_MN3 = summed_input(order3_skids[1], matrix_ad, pairs)
sum_photo3 = summed_input(order3_skids[2], matrix_ad, pairs)
sum_vtd3 = summed_input(order3_skids[3], matrix_ad, pairs)
sum_AN3 = summed_input(order3_skids[4], matrix_ad, pairs)
sum_ORN3 = summed_input(order3_skids[5], matrix_ad, pairs)
sum_thermo3 = summed_input(order3_skids[6], matrix_ad, pairs)

#sum_AN_MN3 = summed_input(order3_skids[7], matrix_ad, pairs)
#sum_ORN_AN3 = summed_input(order3_skids[8], matrix_ad, pairs)
#sum_ORN_MN3 = summed_input(order3_skids[9], matrix_ad, pairs)

data = [sum_AN3['leftid'], sum_AN3['rightid'], sum_AN3['average_input'],
                                                sum_MN3['average_input'],
                                                sum_ORN3['average_input'],
                                                sum_thermo3['average_input'],
                                                sum_vtd3['average_input'],
                                                sum_A00c3['average_input'],
                                                sum_photo3['average_input']]
headers = ["leftid", "rightid", "AN", "MN", "ORN","thermo", "vtd", "A00c", "photo"]
input_all = pd.concat(data, axis=1, keys=headers)


# use threshold to identify neurons
threshold = 0.05

thermo4 = input_all.loc[(input_all['thermo']>=threshold) & (sum_thermo3['leftid_input']>0) & (sum_thermo3['rightid_input']>0)]
photo4 = input_all.loc[(input_all['photo']>=threshold) & (sum_photo3['leftid_input']>0) & (sum_photo3['rightid_input']>0)]
vtd4 = input_all.loc[(input_all['vtd']>=threshold) & (sum_vtd3['leftid_input']>0) & (sum_vtd3['rightid_input']>0)]
A00c4 = input_all.loc[(input_all['A00c']>=threshold) & (sum_A00c3['leftid_input']>0) & (sum_A00c3['rightid_input']>0)]
AN4 = input_all.loc[(input_all['AN']>=threshold) & (sum_AN3['leftid_input']>0) & (sum_AN3['rightid_input']>0)]
MN4 = input_all.loc[(input_all['MN']>=threshold) & (sum_MN3['leftid_input']>0) & (sum_MN3['rightid_input']>0)]
ORN4 = input_all.loc[(input_all['ORN']>=threshold) & (sum_ORN3['leftid_input']>0) & (sum_ORN3['rightid_input']>0)]

input_all4 = input_all.sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False])
sns.heatmap(input_all4.iloc[:, 2:9], cmap = 'Reds')


# %%
# output CSVs 
ORN4.to_csv('identify_neuron_classes/csv/order4_ORN.csv')
thermo4.to_csv('identify_neuron_classes/csv/order4_thermo.csv')
photo4.to_csv('identify_neuron_classes/csv/order4_photo.csv')
AN4.to_csv('identify_neuron_classes/csv/order4_AN.csv')
MN4.to_csv('identify_neuron_classes/csv/order4_MN.csv')
vtd4.to_csv('identify_neuron_classes/csv/order4_vtd.csv')
A00c4.to_csv('identify_neuron_classes/csv/order4_A00c.csv')

# %%
