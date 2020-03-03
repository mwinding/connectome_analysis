#%%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

#%%
import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
import matplotlib.pyplot as plt
import seaborn as sns

connectors = pd.read_csv('outputs/connectdists.csv')
connectors_raw = pd.read_csv('outputs/connectdists_raw.csv')

#print(connectors)

inputs = []
outputs = []
for i in range(len(connectors)):
    if(connectors.iloc[i]['type']=='postsynaptic'):
        inputs.append(connectors.iloc[i]['distance_root'])
    if(connectors.iloc[i]['type']=='presynaptic'):
        outputs.append(connectors.iloc[i]['distance_root'])

inputs_raw = []
outputs_raw = []
for i in range(len(connectors_raw)):
    if(connectors_raw.iloc[i]['type']=='postsynaptic'):
        inputs_raw.append(connectors_raw.iloc[i]['distance_root'])
    if(connectors_raw.iloc[i]['type']=='presynaptic'):
        outputs_raw.append(connectors_raw.iloc[i]['distance_root'])

#%%
fig, ax = plt.subplots(1,1,figsize=(8,4))
#sns.distplot(data = inputs, ax = ax, )
#sns.distplot(data = outputs, ax = ax, )
sns.distplot(inputs, ax = ax)
sns.distplot(outputs, ax = ax)

#ax.hist(outputs, bins=20, density = True)
#ax.hist(inputs, bins=20, density = True)
#plt.show()

# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(inputs_raw, ax = ax)
sns.distplot(outputs_raw, ax = ax)
#ax.hist(outputs_raw, bins=20, density = True)
#ax.hist(inputs_raw, bins=20, density = True)
#plt.show()

# %%
print(inputs_raw)

# %%
