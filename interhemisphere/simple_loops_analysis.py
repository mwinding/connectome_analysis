#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
graph = pg.Analyze_Nx_G(ad_edges)

# %%
# load neuron types

# majority types
ipsi = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
ipsi = ipsi + list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite')))
bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))
contralateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw ipsilateral dendrite')))

# minority types
ipsi_bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw ipsilateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
bilateral_bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw bilateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))
contra_bilateral = list(np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw bilateral dendrite')))

# %%
# 


