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

import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

connectors_unsplittable = pd.read_csv('axon_dendrite_data/unsplittable_connectordists_norm_dividebymax.csv')
#connectors_raw = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_raw.csv')
#connectors_norm_divide = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_norm_dividebymax.csv')
connectors_norm = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_norm_dividebymax.csv')
#connectors_norm2 = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_norm2.csv')
connectors_norm_right1 = pd.read_csv('axon_dendrite_data/splittable_connectors_right1_norm.csv')



inputs = []
outputs = []
for i in range(len(connectors_unsplittable)):
    if(connectors_unsplittable.iloc[i]['type']=='postsynaptic'):
        inputs.append(connectors_unsplittable.iloc[i]['distance_root'])
    if(connectors_unsplittable.iloc[i]['type']=='presynaptic'):
        outputs.append(connectors_unsplittable.iloc[i]['distance_root'])
'''
inputs_raw = []
outputs_raw = []
for i in range(len(connectors_raw)):
    if(connectors_raw.iloc[i]['type']=='postsynaptic'):
        inputs_raw.append(connectors_raw.iloc[i]['distance_root'])
    if(connectors_raw.iloc[i]['type']=='presynaptic'):
        outputs_raw.append(connectors_raw.iloc[i]['distance_root'])

inputs_norm_d = []
outputs_norm_d = []
for i in range(len(connectors_norm_divide)):
    if(connectors_norm_divide.iloc[i]['type']=='postsynaptic'):
        inputs_norm_d.append(connectors_norm_divide.iloc[i]['distance_root'])
    if(connectors_norm_divide.iloc[i]['type']=='presynaptic'):
        outputs_norm_d.append(connectors_norm_divide.iloc[i]['distance_root'])
'''
inputs_norm = []
outputs_norm = []
for i in range(len(connectors_norm)):
    if(connectors_norm.iloc[i]['type']=='postsynaptic'):
        inputs_norm.append(connectors_norm.iloc[i]['distance_root'])
    if(connectors_norm.iloc[i]['type']=='presynaptic'):
        outputs_norm.append(connectors_norm.iloc[i]['distance_root'])
'''
inputs_norm2 = []
outputs_norm2 = []
for i in range(len(connectors_norm2)):
    if(connectors_norm2.iloc[i]['type']=='postsynaptic'):
        inputs_norm2.append(connectors_norm2.iloc[i]['distance_root'])
    if(connectors_norm2.iloc[i]['type']=='presynaptic'):
        outputs_norm2.append(connectors_norm2.iloc[i]['distance_root'])
'''

inputs_norm_right1 = []
outputs_norm_right1 = []
for i in range(len(connectors_norm_right1)):
    if(connectors_norm_right1.iloc[i]['type']=='postsynaptic'):
        inputs_norm_right1.append(connectors_norm.iloc[i]['distance_root'])
    if(connectors_norm_right1.iloc[i]['type']=='presynaptic'):
        outputs_norm_right1.append(connectors_norm_right1.iloc[i]['distance_root'])
#%%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs, ax = ax)
sns.distplot(outputs, ax = ax)


# %%
'''
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs_raw, ax = ax)
sns.distplot(outputs_raw, ax = ax)
'''
#%%
'''
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs_norm_d, ax = ax)
sns.distplot(outputs_norm_d, ax = ax)
'''
# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs_norm, ax = ax)
sns.distplot(outputs_norm, ax = ax)

# %%
'''
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs_norm2, ax = ax)
sns.distplot(outputs_norm2, ax = ax)
'''
# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs_norm_right1, ax = ax)
sns.distplot(outputs_norm_right1, ax = ax)

# %%
