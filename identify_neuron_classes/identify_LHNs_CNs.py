#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

rm = pymaid.CatmaidInstance(url, token, name, password)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
inputs = pd.DataFrame([mg.meta.axon_input, mg.meta.dendrite_input]).T
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-09-22.csv', header = 0) # import pairs

# %%
from connectome_tools.process_matrix import Adjacency_matrix, Promat
from datetime import date

# set up initial variables for cell identification
brain_adj = Adjacency_matrix(mg.adj, mg.meta.index, pairs, inputs,'axo-dendritic')

uPN = pymaid.get_skids_by_annotation('mw uPN')
tPN = pymaid.get_skids_by_annotation('mw tPN')
vPN = pymaid.get_skids_by_annotation('mw vPN')
KC = pymaid.get_skids_by_annotation('mw KC')
mPN = pymaid.get_skids_by_annotation('mw mPN')

MBON = pymaid.get_skids_by_annotation('mw MBON')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')
# %%
# identify LHNs

threshold = 0.01
threshold_group = 0.05
excluded = uPN + tPN + vPN + mPN + KC + MBON + MBIN

# identification by pairwise downstream threhold
_, ds_uPN = brain_adj.downstream(uPN, threshold, excluded, exclude_unpaired=True)
edges_uPN, ds_uPN = brain_adj.edge_threshold(uPN, ds_uPN, threshold, direction='downstream')

_, ds_tPN = brain_adj.downstream(tPN, threshold, excluded, exclude_unpaired=True)
edges_tPN, ds_tPN = brain_adj.edge_threshold(tPN, ds_tPN, threshold, direction='downstream')

_, ds_vPN = brain_adj.downstream(vPN, threshold, excluded, exclude_unpaired=True)
edges_vPN, ds_vPN = brain_adj.edge_threshold(vPN, ds_vPN, threshold, direction='downstream')

LHN = list(np.unique(ds_uPN + ds_tPN + ds_vPN))
pd.DataFrame(LHN).to_csv(f'identify_neuron_classes/csv/LHN_threshold-{threshold}_pairwise-from-uPN-tPN-vPN_{str(date.today())}.csv',
                        index = False, header = False)

# compare to identification by groups
ds_uPN_group = brain_adj.downstream(uPN, threshold_group, excluded, by_group=True, exclude_unpaired=True)
ds_tPN_group = brain_adj.downstream(tPN, threshold_group, excluded, by_group=True, exclude_unpaired=True)
ds_vPN_group = brain_adj.downstream(vPN, threshold_group, excluded, by_group=True, exclude_unpaired=True)

LHN_group = np.unique(ds_uPN_group + ds_tPN_group + ds_vPN_group)
pd.DataFrame(LHN_group).to_csv(f'identify_neuron_classes/csv/LHN_threshold-{threshold}_bygroup-from-uPN-tPN-vPN_{str(date.today())}.csv',
                        index = False, header = False)

len(np.intersect1d(LHN_group, LHN))/len(np.union1d(LHN_group, LHN))
# %%
# identify CNs

# identification by pairwise downstream threhold
_, ds_MBON = brain_adj.downstream(MBON, threshold, excluded, exclude_unpaired=True)
edges_MBON, ds_MBON = brain_adj.edge_threshold(MBON, ds_MBON, threshold, direction='downstream')

_, ds_LHN = brain_adj.downstream(LHN, threshold, excluded + LHN, exclude_unpaired=True)
edges_LHN, ds_LHN = brain_adj.edge_threshold(LHN, ds_LHN, threshold, direction='downstream')

CN1 = list(np.intersect1d(ds_MBON, ds_LHN))
CN2 = list(np.intersect1d(ds_MBON, LHN))

CN = np.unique(CN1 + CN2)
pd.DataFrame(CN).to_csv(f'identify_neuron_classes/csv/CN_threshold-{threshold}_pairwise-from-LHN-MBON-PNs_{str(date.today())}.csv',
                        index = False, header = False)

# compare to identification by groups
ds_MBON_group = brain_adj.downstream(MBON, threshold_group, excluded, by_group=True, exclude_unpaired=True)
ds_LHN_group = brain_adj.downstream(LHN_group, threshold_group, excluded, by_group=True, exclude_unpaired=True)

CN1_group = list(np.intersect1d(ds_MBON_group, ds_LHN_group))
CN2_group = list(np.intersect1d(ds_MBON_group, LHN_group))

CN_group = np.unique(CN1_group + CN2_group)
pd.DataFrame(CN_group).to_csv(f'identify_neuron_classes/csv/CN_threshold-{threshold}_bygroup-from-LHN-MBON-PNs_{str(date.today())}.csv',
                        index = False, header = False)

len(np.intersect1d(CN_group, CN))/len(np.union1d(CN_group, CN))
# %%
# LH2N neurons

pd.DataFrame(ds_LHN).to_csv(f'identify_neuron_classes/csv/LH2N_threshold-{threshold}_pairwise-from-LHN_{str(date.today())}.csv',
                        index = False, header = False)

# %%
