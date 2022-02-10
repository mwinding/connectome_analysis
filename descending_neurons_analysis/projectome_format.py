#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)
import connectome_tools.process_matrix as pm

projectome = pd.read_csv('data/projectome/projectome_mw brain paper all neurons_split.csv', header = 0)

# %%
# format into adjacency matrix
skids = np.unique(projectome['skeleton']).tolist()
meshes = projectome.columns[8:34].tolist()
meshes.sort()
meshes = meshes[16:20] + meshes[20:26] + meshes[0:16]

# projectome matrix
zeros_array = np.zeros(shape=(len(meshes) + len(skids),len(meshes) + len(skids)))
project_mat = pd.DataFrame(zeros_array, columns=(skids + meshes), index = (skids + meshes))

no_mesh_inputs = 0
no_mesh_outputs = 0
multi_mesh_inputs = 0
multi_mesh_outputs = 0

# add num outputs here?
# currently just sums up presynaptic sites
for skid in tqdm(skids):
    
    skel_projectome = projectome.loc[projectome.skeleton == skid, :]
    skel_projectome = skel_projectome.reset_index()
    #print(skeleton)
    for i in range(0, len(skel_projectome.index)):
        location = skel_projectome.loc[i, meshes]
        if(skel_projectome.is_input[i]==0):
            if(sum(location) == 1): # outputs to only one mesh
                # how many outputs from connector?
                skel_projectome.iloc[i, :].connector 
                mesh = location[location==1].index.tolist()[0]
                project_mat.loc[skid, mesh] = project_mat.loc[skid, mesh] + 1

            if(sum(location) > 1):
                #print('>1 meshes for output connector %i from skid %i' %(skel_projectome.connector[i], skid))
                multi_mesh_outputs = multi_mesh_outputs + 1

            if(sum(location) == 0):
                #print('0 meshes for output connector %i from skid %i' %(skel_projectome.connector[i], skid))
                no_mesh_outputs = no_mesh_outputs + 1

        if(projectome['is_input'][i]==1):
            if(sum(location) == 1): # inputs from only one mesh
                mesh = location[location==1].index.tolist()[0]
                project_mat.loc[mesh, skid] = project_mat.loc[mesh, skid] + 1

            if(sum(location) > 1):
                #print('>1 meshes for input connector %i from skid %i' %(skel_projectome.connector[i], skid))
                multi_mesh_inputs = multi_mesh_inputs + 1

            if(sum(location) == 0):
                #print('0 meshes for input connector %i from skid %i' %(skel_projectome.connector[i], skid))
                no_mesh_inputs = no_mesh_inputs + 1
# %%
# export adjacency to csv

project_mat.to_csv('data/projectome/projectome_adjacency.csv')

# %%
