#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

import pandas as pd

from pymaid_creds import url, name, password, token
import pymaid as pymaid
import numpy as np

rm = pymaid.CatmaidInstance(url, token, name, password)
pairs = pd.read_csv('data/pairs-2021-04-06.csv')

# %%
#

all_neurons = pymaid.get_skids_by_annotation('mw brain paper all neurons')

np.setdiff1d(pairs.leftid.values, all_neurons) # check these neurons in CATMAID and modify annotations as necessary
np.setdiff1d(pairs.rightid.values, all_neurons) # check these neurons in CATMAID and modify annotations as necessary
pymaid.get_neurons
# %%
