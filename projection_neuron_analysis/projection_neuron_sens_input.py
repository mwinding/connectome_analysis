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
from tqdm import tqdm


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
matrix_ad.columns = pd.to_numeric(matrix_ad.columns)

# convert to %input
total_inputs = pd.read_csv('data/input_counts.csv', header = 0, index_col = 0)

for i in tqdm(np.arange(0, len(matrix_ad.index), 1)):
    inputs = total_inputs.loc[matrix_ad.index[i] == total_inputs.index, ' dendrite_inputs'].values
    if(inputs != 0):
        matrix_ad.loc[:, matrix_ad.index[i]] = matrix_ad.loc[:, matrix_ad.index[i]]/inputs

# %%

def summed_input(group_skids, matrix, pairList, input_type):
    submatrix = matrix.loc[group_skids, :]
    submatrix = submatrix.sum(axis = 0)

    cols = ['leftid', 'rightid', 'leftid_'+ input_type + '_input', 'rightid_'+ input_type + '_input']
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

sum_ORN = summed_input(sens_skids[0], matrix_ad, pairs, 'ORN')
sum_thermo = summed_input(sens_skids[1], matrix_ad, pairs, 'thermo')
sum_visual = summed_input(sens_skids[2], matrix_ad, pairs, 'visual')
sum_AN = summed_input(sens_skids[3], matrix_ad, pairs, 'AN')
sum_MN = summed_input(sens_skids[4], matrix_ad, pairs, 'MN')
sum_vtd = summed_input(sens_skids[5], matrix_ad, pairs, 'vtd')
sum_A00c = summed_input(sens_skids[6], matrix_ad, pairs, 'A00c')

sensory_sum = pd.concat([sum_ORN, sum_AN.iloc[:,2:4], sum_MN.iloc[:,2:4], sum_thermo.iloc[:,2:4], sum_visual.iloc[:,2:4], sum_vtd.iloc[:,2:4], sum_A00c.iloc[:,2:4]], axis=1)

sum_ORN_combined = sum_ORN.iloc[:, 2:4].sum(axis = 1)/2
sum_thermo_combined = sum_thermo.iloc[:, 2:4].sum(axis = 1)/2
sum_visual_combined = sum_visual.iloc[:, 2:4].sum(axis = 1)/2
sum_AN_combined = sum_AN.iloc[:, 2:4].sum(axis = 1)/2
sum_MN_combined = sum_MN.iloc[:, 2:4].sum(axis = 1)/2
sum_vtd_combined = sum_vtd.iloc[:, 2:4].sum(axis = 1)/2
sum_A00c_combined = sum_A00c.iloc[:, 2:4].sum(axis = 1)/2

sensory_sum_combined = pd.concat([sum_ORN.iloc[:, 0:2], sum_ORN_combined, sum_thermo_combined,
                        sum_visual_combined, sum_AN_combined, sum_MN_combined,
                        sum_vtd_combined, sum_A00c_combined], axis=1)


# %%
# projection neurons skids

def load_skids_pairs(annotation, pairList):
    skids = pymaid.get_skids_by_annotation(annotation)
    pairs = promat.extract_pairs_from_list(skids, pairList)
    return(pairs)

# get PN skids and arrange them into pair columns
PNs_pairs = load_skids_pairs("mw all PNs", pairs)

# call pair columns in whole matrix
sensory_sum.index = sensory_sum['leftid'].values
sensory_sum_combined.index = sensory_sum['leftid'].values

# sort properly
uPNs = load_skids_pairs("mw uPN", pairs)
vPNs = load_skids_pairs("mw vPN", pairs)
tPNs = load_skids_pairs("mw tPN", pairs)
AN_PNs = load_skids_pairs("mw AN PN", pairs)
MN_PNs = load_skids_pairs("mw MN PN", pairs)
ANMN_PNs = load_skids_pairs("mw AN_MN PN putative", pairs)
vtd_PNs = load_skids_pairs("mw vtd PN putative", pairs)

PNs_sum = sensory_sum.loc[np.concatenate([uPNs['leftid'].values, 
                                            tPNs['leftid'].values, 
                                            vPNs['leftid'].values, 
                                            AN_PNs['leftid'].values, 
                                            MN_PNs['leftid'].values, 
                                            ANMN_PNs['leftid'].values, 
                                            vtd_PNs['leftid'].values]), :]

PNs_sum_combined = sensory_sum_combined.loc[np.concatenate([uPNs['leftid'].values, 
                                            tPNs['leftid'].values, 
                                            vPNs['leftid'].values, 
                                            AN_PNs['leftid'].values, 
                                            MN_PNs['leftid'].values, 
                                            ANMN_PNs['leftid'].values, 
                                            vtd_PNs['leftid'].values]), :]


# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.heatmap(np.transpose(PNs_sum.iloc[:, 2:]), ax = ax)

# %%
# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

PNs_sum_combined.columns = ['leftid', 'rightid', 'ORN', 'thermo', 'visual', 'AN', 'MN', 'vtd', 'A00c']

fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.heatmap(np.transpose(PNs_sum_combined.iloc[:, 2:]), ax = ax, robust = True)
#plt.savefig('identify_neuron_classes/plots/PNs.pdf', format='pdf', bbox_inches='tight')

# %%
