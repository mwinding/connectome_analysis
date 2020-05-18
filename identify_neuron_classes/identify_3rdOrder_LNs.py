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

# pull skids for 3rd order neurons
order3 = pymaid.get_annotated('mw brain inputs 3rd_order')
order3.index = np.arange(0, len(order3['name']), 1) # reset indices
skids3 = []
for i in np.arange(0, len(order3), 1):
    skids_temp = order3['name'][i]
    skids_temp = pymaid.get_skids_by_annotation(skids_temp)
    skids3.append(skids_temp)

# %%
# connections among group members

# import connector list CSV
connectors = pd.read_csv('data/connectors.csv', header=0, index_col=0)
skids = pd.unique(connectors['presynaptic_to'])

outputs = []
for skid in skids:
    temp = connectors[connectors['presynaptic_to'] == skid]
    outputs.append([skid, len(temp['connector_id'])])

outputs = pd.DataFrame(outputs, columns = ['skeleton_ID', 'outputs'])

outputs_axon = []
for skid in skids:
    temp = connectors[connectors['presynaptic_to'] == skid]
    temp = temp[temp['presynaptic_type'] == 'axon']
    #temp = temp[temp['postsynaptic_type'] == 'dendrite']
    outputs_axon.append([skid, len(temp['connector_id'])])

outputs_axon = pd.DataFrame(outputs, columns = ['skeleton_ID', 'outputs'])

def intragroup_connections(matrix, skids, outputs):
    mat = matrix.loc[skids, skids]
    mat = mat.sum(axis=1)

    # convert to % outputs
    for i in np.arange(0, len(mat.index), 1):
        axon_output = outputs.loc[outputs['skeleton_ID']==mat.index[i], 'outputs'].values
        if(axon_output != 0):
            mat.loc[mat.index[i]] = mat.loc[mat.index[i]]/axon_output

    return(mat)

# checking 50% output (all to all) intragroup
A00c3_mat = intragroup_connections(matrix, skids3[0], outputs) # one here
thermo3_mat = intragroup_connections(matrix, skids3[1], outputs) # two here
photo3_mat = intragroup_connections(matrix, skids3[2], outputs)
vtd3_mat = intragroup_connections(matrix, skids3[3], outputs)
ORN3_mat = intragroup_connections(matrix, skids3[4], outputs) # one here
MN3_mat = intragroup_connections(matrix, skids3[5], outputs)
AN3_mat = intragroup_connections(matrix, skids3[6], outputs) # seven here
AN_MN3_mat = intragroup_connections(matrix, skids3[7], outputs) # ten here
ORN_AN3_mat = intragroup_connections(matrix, skids3[8], outputs)
ORN_MN3_mat = intragroup_connections(matrix, skids3[9], outputs)


# checking 50% output from axon intragroup
A00c3_mat_ad = intragroup_connections(matrix_ad, skids3[0], outputs)
thermo3_mat_ad = intragroup_connections(matrix_ad, skids3[1], outputs)
photo3_mat_ad = intragroup_connections(matrix_ad, skids3[2], outputs)
vtd3_mat_ad = intragroup_connections(matrix_ad, skids3[3], outputs)
ORN3_mat_ad = intragroup_connections(matrix_ad, skids3[4], outputs)
MN3_mat_ad = intragroup_connections(matrix_ad, skids3[5], outputs)
AN3_mat_ad = intragroup_connections(matrix_ad, skids3[6], outputs) # seven LNs here; all repeats from non-A graph
AN_MN3_mat_ad = intragroup_connections(matrix_ad, skids3[7], outputs) # one here; repeat from non-Ad graph
ORN_AN3_mat_ad = intragroup_connections(matrix_ad, skids3[8], outputs)
ORN_MN3_mat_ad = intragroup_connections(matrix_ad, skids3[9], outputs)

# %%
threshold = 0.5
len(np.unique(A00c3_mat[A00c3_mat>=threshold].index.tolist() + 
    thermo3_mat[thermo3_mat>=threshold].index.tolist() +
    photo3_mat[photo3_mat>=threshold].index.tolist() +
    vtd3_mat[vtd3_mat>=threshold].index.tolist() +
    ORN3_mat[ORN3_mat>=threshold].index.tolist() +
    MN3_mat[MN3_mat>=threshold].index.tolist() +
    AN3_mat[AN3_mat>=threshold].index.tolist() +
    AN_MN3_mat[AN_MN3_mat>=threshold].index.tolist() +
    ORN_AN3_mat[ORN_AN3_mat>=threshold].index.tolist() +
    ORN_MN3_mat[ORN_MN3_mat>=threshold].index.tolist() + 
    A00c3_mat_ad[A00c3_mat_ad>=threshold].index.tolist() + 
    thermo3_mat_ad[thermo3_mat_ad>=threshold].index.tolist() +
    photo3_mat_ad[photo3_mat_ad>=threshold].index.tolist() +
    vtd3_mat_ad[vtd3_mat_ad>=threshold].index.tolist() +
    ORN3_mat_ad[ORN3_mat_ad>=threshold].index.tolist() +
    MN3_mat_ad[MN3_mat_ad>=threshold].index.tolist() +
    AN3_mat_ad[AN3_mat_ad>=threshold].index.tolist() +
    AN_MN3_mat_ad[AN_MN3_mat_ad>=threshold].index.tolist() +
    ORN_AN3_mat_ad[ORN_AN3_mat_ad>=threshold].index.tolist() +
    ORN_MN3_mat_ad[ORN_MN3_mat_ad>=threshold].index.tolist()).tolist() )
    
# %%
# output CSVs of each putative LN and non-LN for each sensory modality
A00c3_mat.loc[A00c3_mat>=0.5].to_csv('identify_neuron_classes/csv/A00c3_LN.csv')
thermo3_mat.loc[thermo3_mat>=0.5].to_csv('identify_neuron_classes/csv/thermo3_LN.csv')
ORN3_mat.loc[ORN3_mat>=0.5].to_csv('identify_neuron_classes/csv/ORN3_LN.csv')
AN3_mat.loc[AN3_mat>=0.5].to_csv('identify_neuron_classes/csv/AN3_LN.csv')
AN_MN3_mat.loc[AN_MN3_mat>=0.5].to_csv('identify_neuron_classes/csv/AN_MN3_LN.csv')


# %%
