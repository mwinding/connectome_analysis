#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

from data_settings import data_date, pairs_path
from contools import Promat

projectome = pd.read_csv('data/projectome/projectome_adjacency.csv', index_col = 0, header = 0)

# %%
# identify skeleton ID of hemilateral neuron pair, based on CSV pair list

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RGN')

pairs = Promat.get_pairs(pairs_path=pairs_path)

pairOrder_dVNC = []
for skid in dVNC:
    if(skid in pairs["leftid"].values):
        pair_skid = pairs["rightid"][pairs["leftid"]==skid].iloc[0]
        pairOrder_dVNC.append(skid)
        pairOrder_dVNC.append(pair_skid)

pairOrder_dSEZ = []
for skid in dSEZ:
    if(skid in pairs["leftid"].values):
        pair_skid = pairs["rightid"][pairs["leftid"]==skid].iloc[0]
        pairOrder_dSEZ.append(skid)
        pairOrder_dSEZ.append(pair_skid)

pairOrder_RGN = []
for skid in RGN:
    if(skid in pairs["leftid"].values):
        pair_skid = pairs["rightid"][pairs["leftid"]==skid].iloc[0]
        pairOrder_RGN.append(skid)
        pairOrder_RGN.append(pair_skid)

# identify meshes
meshes = ['SEZ_left', 'SEZ_right', 'T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right', 'A1_left', 'A1_right', 'A2_left', 'A2_right', 'A3_left', 'A3_right', 'A4_left', 'A4_right', 'A5_left', 'A5_right', 'A6_left', 'A6_right', 'A7_left', 'A7_right', 'A8_left', 'A8_right']

# %%
projectome = project_mat # some of the int indices are currently str in projectome, not sure why

dVNC_projectome = projectome.loc[pairOrder_dVNC, meshes]
'''
dVNC_projectome_pairs = []
indices = []
for i in np.arange(0, len(dVNC_projectome.index), 2):
    dVNC_projectome_pairs.append((dVNC_projectome.iloc[i, :] + dVNC_projectome.iloc[i+1, :])/2)
    indices.append(dVNC_projectome.index[i])
'''
dVNC_projectome_pairs = []
indices = []
for i in np.arange(0, len(dVNC_projectome.index), 2):
    combined_pairs = (dVNC_projectome.iloc[i, :] + dVNC_projectome.iloc[i+1, :])/2

    combined_hemisegs = []
    for j in np.arange(0, len(combined_pairs), 2):
        combined_hemisegs.append((combined_pairs[j] + combined_pairs[j+1])/2)
    
    dVNC_projectome_pairs.append(combined_hemisegs)
    indices.append(dVNC_projectome.index[i])

dVNC_projectome_pairs = pd.DataFrame(dVNC_projectome_pairs, index = indices, columns = ['SEZ', 'T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'])
dVNC_projectome_pairs_VNC = dVNC_projectome_pairs.iloc[:, 1:len(dVNC_projectome_pairs)]

cluster = sns.clustermap(dVNC_projectome_pairs.iloc[:, 1:len(dVNC_projectome_pairs)], col_cluster = False, rasterized = True, figsize=(10,10))
plt.savefig('plots/projectome_cluster.pdf', bbox_inches='tight', transparent = True)


#cluster = sns.clustermap(dVNC_projectome_pairs.iloc[:, 1:len(dVNC_projectome_pairs)], col_cluster = False, rasterized = True, figsize=(40,40))
#plt.savefig('descending_neurons_analysis/plots/projectome_cluster_readable_skids.pdf', bbox_inches='tight', transparent = True)

# programmatically access dendrogram
# implement later

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

cluster_skids = dVNC_projectome_pairs.index[cluster.dendrogram_row.reordered_ind]
cluster_dendrogram = cluster.dendrogram_row.dendrogram 

#pd.DataFrame(cluster_skids).to_csv('plots/cluster_skids_top_to_bottom.csv')

#assignments = fcluster(linkage(cluster_dendrogram.Z, method='complete'),4,'distance')

# %%
