# %%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

# %%
# signal expansion

# load neurons
KCs = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw KC'), pymaid.get_skids_by_annotation('mw partially differentiated')))
LHNs = pymaid.get_skids_by_annotation('mw LHN')

uPNs = pymaid.get_skids_by_annotation('mw uPN')

# KC, LHNs from uPNs
KC_counts = (adj_ad.loc[uPNs, KCs]>0).sum(axis=0)
LHN_counts = (adj_ad.loc[uPNs, LHNs]>0).sum(axis=0)

print(f'KCs receive from: {KC_counts.mean():.2f} +/- {KC_counts.std():.2f} PNs')
print(f'LHNs receive from: {LHN_counts.mean():.2f} +/- {LHN_counts.std():.2f} PNs')

# uPN to KCs, LHNs
uPN_KC_outputs = (adj_ad.loc[uPNs, KCs]>0).sum(axis=1)
uPN_LHN_outputs = (adj_ad.loc[uPNs, LHNs]>0).sum(axis=1)

print(f'uPNs output to: {uPN_KC_outputs.mean():.2f} +/- {uPN_KC_outputs.std():.2f} KCs')
print(f'uPNs output to: {uPN_LHN_outputs.mean():.2f} +/- {uPN_LHN_outputs.std():.2f} LHNs')

# %%
