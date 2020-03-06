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

connectors_split_center = pd.read_csv('axon_dendrite_data/testdists_centeredsplit.csv')

splittable_inputs = []
splittable_outputs = []
for i in tqdm(range(len(connectors_split_center))):
    if(connectors_split_center.iloc[i]['type']=='postsynaptic'):
        splittable_inputs.append(connectors_split_center.iloc[i]['distance'])
    if(connectors_split_center.iloc[i]['type']=='presynaptic'):
        splittable_outputs.append(connectors_split_center.iloc[i]['distance'])

#%%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(splittable_inputs, ax = ax)
sns.distplot(splittable_outputs, ax = ax)


# %%
