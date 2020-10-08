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

sens_skids = []
for i in np.arange(0, len(sensories), 1):
    sens = sensories['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    sens_skids.append(sens)

sum_ORN = summed_input(sens_skids[0], matrix_ad, pairs)
sum_thermo = summed_input(sens_skids[1], matrix_ad, pairs)
sum_photo = summed_input(sens_skids[2], matrix_ad, pairs)
sum_AN = summed_input(sens_skids[3], matrix_ad, pairs)
sum_MN = summed_input(sens_skids[4], matrix_ad, pairs)
sum_vtd = summed_input(sens_skids[5], matrix_ad, pairs)
sum_A00c = summed_input(sens_skids[6], matrix_ad, pairs)


data = [sum_AN['leftid'], sum_AN['rightid'], sum_AN['average_input'],
                                                sum_MN['average_input'],
                                                sum_ORN['average_input'],
                                                sum_thermo['average_input'],
                                                sum_vtd['average_input'],
                                                sum_A00c['average_input'],
                                                sum_photo['average_input']]
headers = ["leftid", "rightid", "AN", "MN", "ORN", "thermo", "vtd", "A00c", "photo"]
input_all = pd.concat(data, axis=1, keys=headers)
# %%

ORN2 = input_all.loc[(input_all['ORN']>=0.05) & (sum_ORN['leftid_input']>0) & (sum_ORN['rightid_input']>0)]
thermo2 = input_all.loc[(input_all['thermo']>=0.05) & (sum_thermo['leftid_input']>0) & (sum_thermo['rightid_input']>0)]
photo2 = input_all.loc[(input_all['photo']>=0.05) & (sum_photo['leftid_input']>0) & (sum_photo['rightid_input']>0)]
AN2 = input_all.loc[(input_all['AN']>=0.05) & (sum_AN['leftid_input']>0) & (sum_AN['rightid_input']>0)]
MN2 = input_all.loc[(input_all['MN']>=0.05) & (sum_MN['leftid_input']>0) & (sum_MN['rightid_input']>0)]
vtd2 = input_all.loc[(input_all['vtd']>=0.05) & (sum_vtd['leftid_input']>0) & (sum_vtd['rightid_input']>0)]
A00c2 = input_all.loc[(input_all['A00c']>=0.05) & (sum_A00c['leftid_input']>0) & (sum_A00c['rightid_input']>0)]

all_index = ORN2.index.values.tolist() + thermo2.index.values.tolist() + photo2.index.values.tolist() + AN2.index.values.tolist() + MN2.index.values.tolist() + vtd2.index.values.tolist() + A00c2.index.values.tolist() 
all_index = np.unique(all_index)

all2 = input_all.iloc[all_index, :]

# sorting data by %input
ORN2 = ORN2.loc[ORN2['ORN'].sort_values(ascending=False).index, :]
thermo2 = thermo2.loc[thermo2['thermo'].sort_values(ascending=False).index, :]
photo2 = photo2.loc[photo2['photo'].sort_values(ascending=False).index, :]
AN2 = AN2.loc[AN2['AN'].sort_values(ascending=False).index, :]
MN2 = MN2.loc[MN2['MN'].sort_values(ascending=False).index, :]
vtd2 = vtd2.loc[vtd2['vtd'].sort_values(ascending=False).index, :]
A00c2 = A00c2.loc[A00c2['A00c'].sort_values(ascending=False).index, :]

all2 = all2.sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False])
sns.heatmap(all2.iloc[:, 2:9], cmap = 'Reds')

# %%
ORN2.to_csv('identify_neuron_classes/csv/order2_ORN.csv')
thermo2.to_csv('identify_neuron_classes/csv/order2_thermo.csv')
photo2.to_csv('identify_neuron_classes/csv/order2_photo.csv')
AN2.to_csv('identify_neuron_classes/csv/order2_AN.csv')
MN2.to_csv('identify_neuron_classes/csv/order2_MN.csv')
vtd2.to_csv('identify_neuron_classes/csv/order2_vtd.csv')
A00c2.to_csv('identify_neuron_classes/csv/order2_A00c.csv')

# %%
sns.distplot(ORN2['ORN'])
sns.distplot(thermo2['thermo'])
sns.distplot(photo2['photo'])
sns.distplot(AN2['AN'])
sns.distplot(MN2['MN'])
sns.distplot(vtd2['vtd'])
sns.distplot(A00c2['A00c'])

#sns.distplot(ORN2['ORN'], bins = 50)

# %%
sns.heatmap(ORN2.iloc[:, 2:9])
sns.heatmap(thermo2.iloc[:, 2:9])
sns.heatmap(photo2.iloc[:, 2:9])
sns.heatmap(AN2.iloc[:, 2:9])
sns.heatmap(MN2.iloc[:, 2:9])
sns.heatmap(vtd2.iloc[:, 2:9])
sns.heatmap(A00c2.iloc[:, 2:9])

# %%
input_all2 = input_all.sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False])
sns.heatmap(input_all2.iloc[:, 2:9], cmap = 'Reds')

# %%
