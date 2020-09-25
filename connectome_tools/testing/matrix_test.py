#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from pymaid_creds import url, name, password, token
import pymaid
import numpy as np
import pandas as pd
from src.data import load_metagraph

rm = pymaid.CatmaidInstance(url, name, password, token)
mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0) # import pairs

KC = pymaid.get_skids_by_annotation('mw KC')
MBON = pymaid.get_skids_by_annotation('mw MBON')

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat

test = Adjacency_matrix(mg.adj, mg.meta.index, pairs, mg.meta.loc[:, ['dendrite_input', 'axon_input']],'axo-dendritic')
test.adj_inter.loc[(slice(None), slice(None), KC), (slice(None), slice(None), MBON)]

# KC left and nonpaired IDs for testing purposes
KC_pair_id = np.unique([x[1] for x in test.adj_inter.loc[(slice(None), slice(None), KC), (slice(None), slice(None), MBON)].index])
MBON_pair_id = np.unique([x[1] for x in test.adj_inter.loc[(slice(None), slice(None), KC), (slice(None), slice(None), MBON)].columns])

# %%

