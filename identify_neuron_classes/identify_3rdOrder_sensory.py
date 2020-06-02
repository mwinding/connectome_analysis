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
order2 = pymaid.get_annotated('mw brain inputs 2nd_order PN')

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

order2_skids = []
for i in np.arange(0, len(order2), 1):
    sens = order2['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    order2_skids.append(sens)

sum_ORN2 = summed_input(order2_skids[0], matrix_ad, pairs)
sum_AN2 = summed_input(order2_skids[1], matrix_ad, pairs)
sum_MN2 = summed_input(order2_skids[2], matrix_ad, pairs)
sum_vtd2 = summed_input(order2_skids[3], matrix_ad, pairs)
sum_thermo2 = summed_input(order2_skids[4], matrix_ad, pairs)
sum_photo2 = summed_input(order2_skids[5], matrix_ad, pairs)
sum_A00c2 = summed_input(order2_skids[6], matrix_ad, pairs)

data = [sum_AN2['leftid'], sum_AN2['rightid'], sum_AN2['average_input'],
                                                sum_MN2['average_input'],
                                                sum_ORN2['average_input'],
                                                sum_thermo2['average_input'],
                                                sum_vtd2['average_input'],
                                                sum_A00c2['average_input'],
                                                sum_photo2['average_input']]
headers = ["leftid", "rightid", "AN", "MN", "ORN", "thermo", "vtd", "A00c", "photo"]
input_all = pd.concat(data, axis=1, keys=headers)
# %%
threshold = 0.05

#ORN3 = input_all.loc[(input_all['ORN']>=threshold) & (sum_ORN2['leftid_input']>0) & (sum_ORN2['rightid_input']>0)]
thermo3 = input_all.loc[(input_all['thermo']>=threshold) & (sum_thermo2['leftid_input']>0) & (sum_thermo2['rightid_input']>0)]
photo3 = input_all.loc[(input_all['photo']>=threshold) & (sum_photo2['leftid_input']>0) & (sum_photo2['rightid_input']>0)]
#AN3 = input_all.loc[(input_all['AN']>=threshold) & (sum_AN2['leftid_input']>0) & (sum_AN2['rightid_input']>0)]
#MN3 = input_all.loc[(input_all['MN']>=threshold) & (sum_MN2['leftid_input']>0) & (sum_MN2['rightid_input']>0)]
vtd3 = input_all.loc[(input_all['vtd']>=threshold) & (sum_vtd2['leftid_input']>0) & (sum_vtd2['rightid_input']>0)]
A00c3 = input_all.loc[(input_all['A00c']>=threshold) & (sum_A00c2['leftid_input']>0) & (sum_A00c2['rightid_input']>0)]

#all_index = ORN3.index.values.tolist() + thermo3.index.values.tolist() + photo3.index.values.tolist() + AN3.index.values.tolist() + MN3.index.values.tolist() + vtd3.index.values.tolist() + A00c3.index.values.tolist() 
#all_index = np.unique(all_index)

#all3 = input_all.iloc[all_index, :]

# sorting data by %input
#ORN3 = ORN3.loc[ORN3['ORN'].sort_values(ascending=False).index, :]
thermo3 = thermo3.loc[thermo3['thermo'].sort_values(ascending=False).index, :]
photo3 = photo3.loc[photo3['photo'].sort_values(ascending=False).index, :]
#AN3 = AN3.loc[AN3['AN'].sort_values(ascending=False).index, :]
#MN3 = MN3.loc[MN3['MN'].sort_values(ascending=False).index, :]
vtd3 = vtd3.loc[vtd3['vtd'].sort_values(ascending=False).index, :]
A00c3 = A00c3.loc[A00c3['A00c'].sort_values(ascending=False).index, :]

#all3 = all3.sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False])
#input_all3 = input_all.sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False])
#sns.heatmap(input_all3.iloc[:, 2:9], cmap = 'Reds')

# the AN/MN/ORN groups are too big
# below I rerun using PNs only
# %%
# will look at 3rd order of the exclusive PNs for each modality and then the mixed PNs
ORN_PNonly = pymaid.get_skids_by_annotation("mw ORN only PN")
AN_PNonly = pymaid.get_skids_by_annotation("mw AN only PN")
MN_PNonly = pymaid.get_skids_by_annotation("mw MN only PN")

ORN_PN = pymaid.get_skids_by_annotation("mw ORN 2nd_order PN")
AN_PN = pymaid.get_skids_by_annotation("mw AN 2nd_order PN")
MN_PN = pymaid.get_skids_by_annotation("mw MN 2nd_order PN")
three_PNtypes = np.unique(ORN_PN + AN_PN + MN_PN).tolist()

# mixed PN skids
ORN_AN_PN = np.intersect1d(ORN_PN, AN_PN).tolist()
MN_AN_PN = np.intersect1d(MN_PN, AN_PN).tolist()
ORN_MN_PN = np.intersect1d(ORN_PN, MN_PN).tolist()
mixed_PN = np.unique(ORN_AN_PN + MN_AN_PN + ORN_MN_PN).tolist()

sum_AN2 = summed_input(AN_PNonly, matrix_ad, pairs)
sum_MN2 = summed_input(MN_PNonly, matrix_ad, pairs)
sum_ORN2 = summed_input(ORN_PNonly, matrix_ad, pairs)
sum_AN_MN_ORN2 = summed_input(mixed_PN, matrix_ad, pairs)
sum_ORN_AN2 = summed_input(ORN_AN_PN, matrix_ad, pairs)
sum_MN_AN2 = summed_input(MN_AN_PN, matrix_ad, pairs)
sum_ORN_MN2 = summed_input(ORN_MN_PN, matrix_ad, pairs)

threshold = 0.05

AN3 = input_all.loc[(input_all['ORN']>=threshold) & (sum_ORN2['leftid_input']>0) & (sum_ORN2['rightid_input']>0)]
MN3 = input_all.loc[(input_all['thermo']>=threshold) & (sum_thermo2['leftid_input']>0) & (sum_thermo2['rightid_input']>0)]
ORN3 = input_all.loc[(input_all['photo']>=threshold) & (sum_photo2['leftid_input']>0) & (sum_photo2['rightid_input']>0)]
AN_MN_ORN3 = input_all.loc[(input_all['AN']>=threshold) & (sum_AN2['leftid_input']>0) & (sum_AN2['rightid_input']>0)]
ORN_AN3 = input_all.loc[(input_all['MN']>=threshold) & (sum_MN2['leftid_input']>0) & (sum_MN2['rightid_input']>0)]
MN_AN3 = input_all.loc[(input_all['vtd']>=threshold) & (sum_vtd2['leftid_input']>0) & (sum_vtd2['rightid_input']>0)]
ORN_MN3 = input_all.loc[(input_all['A00c']>=threshold) & (sum_A00c2['leftid_input']>0) & (sum_A00c2['rightid_input']>0)]

data = [sum_AN2['leftid'], sum_AN2['rightid'], sum_AN2['average_input'],
                                                sum_MN2['average_input'],
                                                sum_ORN2['average_input'],
                                                sum_AN_MN_ORN2['average_input'],
                                                sum_thermo2['average_input'],
                                                sum_vtd2['average_input'],
                                                sum_A00c2['average_input'],
                                                sum_photo2['average_input']]
headers = ["leftid", "rightid", "AN", "MN", "ORN", "AN_MN_ORN","thermo", "vtd", "A00c", "photo"]
input_all = pd.concat(data, axis=1, keys=headers)

all_index = ORN3.index.values.tolist() + thermo3.index.values.tolist() + photo3.index.values.tolist() + AN3.index.values.tolist() + MN3.index.values.tolist() + vtd3.index.values.tolist() + A00c3.index.values.tolist() + AN_MN_ORN3.index.values.tolist()
all_index = np.unique(all_index)

input_all3 = input_all.sort_values(['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c'], ascending=[False, False, False, False, False, False, False])
sns.heatmap(input_all3.iloc[:, 2:9], cmap = 'Reds')
# %%
# output CSVs 
ORN3.to_csv('identify_neuron_classes/csv/order3_ORN.csv')
thermo3.to_csv('identify_neuron_classes/csv/order3_thermo.csv')
photo3.to_csv('identify_neuron_classes/csv/order3_photo.csv')
AN3.to_csv('identify_neuron_classes/csv/order3_AN.csv')
MN3.to_csv('identify_neuron_classes/csv/order3_MN.csv')
vtd3.to_csv('identify_neuron_classes/csv/order3_vtd.csv')
A00c3.to_csv('identify_neuron_classes/csv/order3_A00c.csv')
AN_MN_ORN3.to_csv('identify_neuron_classes/csv/order3_AN_MN_ORN.csv')

# %%
