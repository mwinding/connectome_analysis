#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import connectome_tools.process_matrix as promat
import pymaid
from pymaid_creds import url, name, password, token

ascending = pd.read_csv('data/brain_boundary.csv', header = 0)

#%%
soma = (ascending['n_soma_like']!=0)
outputs_inBrain = (ascending['n_outputs_inside_brain']>0)
openends = (ascending['n_open_ends_inside_brain']>0)

# position of U1 T1 soma: 57505.4, 89799.7, 72550
SEZ = (ascending['soma_z']<72550)
VNC = (ascending['soma_z']>72550)


# general skeletons with soma that are ascending (SEZ and VNC)
soma_index = np.where(soma)
soma_output_index = np.where(soma & outputs_inBrain)
soma_output_open_index = np.where(soma & outputs_inBrain & openends)

print(len(ascending['skeleton_id'].iloc[soma_index].values))
print(len(ascending['skeleton_id'].iloc[soma_output_index].values))
print(len(ascending['skeleton_id'].iloc[soma_output_open_index].values))

# SEZ skeletons with soma that are ascending
soma_output_open_SEZ_index = np.where(soma & outputs_inBrain & openends & SEZ)
soma_open_SEZ_index = np.where(soma & openends & SEZ)
soma_output_SEZ_index = np.where(soma & SEZ & outputs_inBrain)
soma_SEZ_index = np.where(soma & SEZ)


print(len(ascending['skeleton_id'].iloc[soma_output_open_SEZ_index].values))
print(len(ascending['skeleton_id'].iloc[soma_open_SEZ_index].values))
print(len(ascending['skeleton_id'].iloc[soma_output_SEZ_index].values))
print(len(ascending['skeleton_id'].iloc[soma_SEZ_index].values))


# VNC skeletons with soma that are ascending
soma_output_open_VNC_index = np.where(soma & outputs_inBrain & openends & VNC)
soma_open_VNC_index = np.where(soma & openends & VNC)
soma_output_VNC_index = np.where(soma & outputs_inBrain & VNC)
soma_VNC_index = np.where(soma & VNC)

print(len(ascending['skeleton_id'].iloc[soma_output_open_VNC_index].values))
print(len(ascending['skeleton_id'].iloc[soma_open_VNC_index].values))
print(len(ascending['skeleton_id'].iloc[soma_output_VNC_index].values))
print(len(ascending['skeleton_id'].iloc[soma_VNC_index].values))



# %%

# output CSV of SEZ ascending
ascending['skeleton_id'].iloc[soma_output_open_SEZ_index].to_csv('ascending_neurons/SEZneurons_output_openends.csv')
ascending['skeleton_id'].iloc[soma_open_SEZ_index].to_csv('ascending_neurons/SEZneurons_openends.csv')
ascending['skeleton_id'].iloc[soma_output_SEZ_index].to_csv('ascending_neurons/SEZ_outputs.csv')
ascending['skeleton_id'].iloc[soma_SEZ_index].to_csv('ascending_neurons/SEZneurons.csv')

# %%
# output CSV of VNC ascending
