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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token

projectome = pd.read_csv('data/projectome_2_7_2020.csv')
rm = pymaid.CatmaidInstance(url, token, name, password)

# %%
# format into adjacency matrix
#unique_skids = list(map(str, np.unique(projectome['skeleton'])))
unique_skids = np.unique(projectome['skeleton']).tolist()
meshes = projectome.columns[7:33].tolist()
meshes.sort()
meshes = meshes[16:20] + meshes[20:26] + meshes[0:16]

# projectome matrix
zero_data = np.zeros(shape=(len(meshes) + len(unique_skids),len(meshes) + len(unique_skids)))
project_mat = pd.DataFrame(zero_data, columns=(unique_skids + meshes), index = (unique_skids + meshes))

no_mesh_inputs = 0
no_mesh_outputs = 0
multi_mesh_inputs = 0
multi_mesh_outputs = 0

# add num outputs here?
# currently just sums up presynaptic sites
for skeleton in tqdm(unique_skids):
    skel_projectome = projectome.loc[projectome.skeleton == skeleton, :]
    skel_projectome = skel_projectome.reset_index()
    #print(skeleton)
    for i in range(0, len(skel_projectome.index)):
        location = skel_projectome.loc[i, meshes]
        if(skel_projectome.is_input[i]==0):
            if(sum(location) == 1): # outputs to only one mesh
                # how many outputs from connector?
                skel_projectome.iloc[i, :].connector 
                mesh = location[location==1].index.tolist()[0]
                project_mat.loc[skeleton, mesh] = project_mat.loc[skeleton, mesh] + 1

            if(sum(location) > 1):
                #print('>1 meshes for output connector %i from skeleton %i' %(skel_projectome.connector[i], skeleton))
                multi_mesh_outputs = multi_mesh_outputs + 1

            if(sum(location) == 0):
                #print('0 meshes for output connector %i from skeleton %i' %(skel_projectome.connector[i], skeleton))
                no_mesh_outputs = no_mesh_outputs + 1

        if(projectome['is_input'][i]==1):
            if(sum(location) == 1): # inputs from only one mesh
                mesh = location[location==1].index.tolist()[0]
                project_mat.loc[mesh, skeleton] = project_mat.loc[mesh, skeleton] + 1

            if(sum(location) > 1):
                #print('>1 meshes for input connector %i from skeleton %i' %(skel_projectome.connector[i], skeleton))
                multi_mesh_inputs = multi_mesh_inputs + 1

            if(sum(location) == 0):
                #print('0 meshes for input connector %i from skeleton %i' %(skel_projectome.connector[i], skeleton))
                no_mesh_inputs = no_mesh_inputs + 1
# %%
# identify skeleton ID of hemilateral neuron pair, based on CSV pair list

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

pairOrder = []
for skid in dVNC:
    if(skid in pairs["leftid"].values):
        pair_skid = pairs["rightid"][pairs["leftid"]==skid].iloc[0]
        pairOrder.append(skid)
        pairOrder.append(pair_skid)

# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
#sns.heatmap(matrix, ax = ax, cmap='OrRd')

plt.imshow(project_mat.loc[pairOrder, meshes[2:len(meshes)]], cmap='OrRd', vmax = 20)
# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
#sns.heatmap(matrix, ax = ax, cmap='OrRd')

plt.imshow(project_mat.loc[meshes[2:len(meshes)], pairOrder], cmap='OrRd', vmax = 20)

# %%
# export 

project_mat.to_csv('descending_neurons_analysis/data/projectome_adjacency.csv')

# %%
