# %%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

connectors = pd.read_csv('brain_completeness/brain_connector_completeness_2021_03_05.csv')
print(connectors)

# postsynaptic completeness
print(np.mean(connectors['postsyn_complete_ppn']))

# presynaptic completeness
print(np.mean(connectors['presyn_complete']))

# edge completeness
print(np.sum(connectors['complete_edges'])/np.sum(connectors['total_partners']-1))


# %%
# import pairs
pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

# calculate completeness based on number of edges touching two complete neurons
completeness = []

for i in range(len(pairs['leftid'])):
    skid_bool_left = connectors['presyn_skeleton'] == pairs['leftid'][i]
    left_complete = np.sum(connectors['complete_edges'][skid_bool_left])/np.sum(connectors['total_partners'][skid_bool_left]-1)

    skid_bool_right = connectors['presyn_skeleton'] == pairs['rightid'][i]
    right_complete = np.sum(connectors['complete_edges'][skid_bool_right])/np.sum(connectors['total_partners'][skid_bool_right]-1)

    if(~np.isnan(left_complete) and ~np.isnan(right_complete)):
        completeness.append({'leftid': pairs['leftid'][i], 'rightid': pairs['rightid'][i], 'left_complete': left_complete, 'right_complete': right_complete})

completeness = pd.DataFrame(completeness)
print(completeness)

#connectors.groupby("presyn_skeleton")["hemi_completeness"].mean()


# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(completeness['left_complete'], ax = ax)
sns.distplot(completeness['right_complete'], ax = ax)



# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(completeness['left_complete'] - completeness['right_complete'], ax = ax)


# %%
