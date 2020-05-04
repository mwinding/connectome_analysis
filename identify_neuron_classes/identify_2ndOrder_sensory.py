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
pairs = pd.read_csv('data/pairs-2020-05-04.csv', header = 0)
paired = pairs.values.flatten()

# %%
rm = pymaid.CatmaidInstance(url, name, password, token)

# pull sensory annotations and then pull associated skids
sensories = pymaid.get_annotated('mw brain sensories')

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
            downstream = matrix.loc[pair, source_group]

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
            downstream = matrix.loc[matrix.index[i], source_group]

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

downstream_sensories = []
for i in np.arange(0, len(sensories), 1):
    sens = sensories['name'][i]
    sens = pymaid.get_skids_by_annotation(sens)
    print(sens)
    downstream_sens = downstream_search(matrix_ad, sens, 3)

    downstream_sensories.append(downstream_sens)

# %%
pd.DataFrame(downstream_sensories[0]).to_csv('identify_neuron_classes/csv/ds_ORN.csv')
pd.DataFrame(downstream_sensories[1]).to_csv('identify_neuron_classes/csv/ds_thermo.csv')
pd.DataFrame(downstream_sensories[2]).to_csv('identify_neuron_classes/csv/ds_visual.csv')
pd.DataFrame(downstream_sensories[3]).to_csv('identify_neuron_classes/csv/ds_AN.csv')
pd.DataFrame(downstream_sensories[4]).to_csv('identify_neuron_classes/csv/ds_MN.csv')
pd.DataFrame(downstream_sensories[6]).to_csv('identify_neuron_classes/csv/ds_vtd.csv')

# %%
