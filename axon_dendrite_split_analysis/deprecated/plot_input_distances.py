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

connectors_unsplittable = pd.read_csv('axon_dendrite_data/unsplittable_connectordists_norm.csv')
connectors_norm = pd.read_csv('axon_dendrite_data/splittable_connectordists_all_2020-3-5.csv')


unsplittable_inputs = []
unsplittable_outputs = []
for i in tqdm(range(len(connectors_unsplittable))):
    if(connectors_unsplittable.iloc[i]['type']=='postsynaptic'):
        unsplittable_inputs.append(connectors_unsplittable.iloc[i]['distance_root'])
    if(connectors_unsplittable.iloc[i]['type']=='presynaptic'):
        unsplittable_outputs.append(connectors_unsplittable.iloc[i]['distance_root'])

splittable_inputs = []
splittable_outputs = []
for i in tqdm(range(len(connectors_norm))):
    if(connectors_norm.iloc[i]['type']=='postsynaptic'):
        splittable_inputs.append(connectors_norm.iloc[i]['distance_root'])
    if(connectors_norm.iloc[i]['type']=='presynaptic'):
        splittable_outputs.append(connectors_norm.iloc[i]['distance_root'])


#%%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(unsplittable_inputs, ax = ax)
sns.distplot(unsplittable_outputs, ax = ax)


# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(splittable_inputs, ax = ax)
sns.distplot(splittable_outputs, ax = ax)
