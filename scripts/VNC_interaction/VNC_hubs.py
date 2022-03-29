#%%
from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date_A1_brain
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

from contools import Promat, Celltype, Celltype_Analyzer, Analyze_Nx_G

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)
select_neurons = pymaid.get_skids_by_annotation(['mw A1 neurons paired', 'mw dVNC', 'mw brain neurons', 'mw brain accessory neurons'])
sens = Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories')
select_neurons = select_neurons + sens
ad_edges_A1 = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date_A1_brain, pairs_combined=True, select_neurons=select_neurons)
pairs = Promat.get_pairs(pairs_path=pairs_path)

A1_neurons = pymaid.get_skids_by_annotation('mw A1 neurons paired') + Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories')
# %%
# identify A1 hubs; include connections to MNs, sensory, brain, etc but only analyze A1 interneurons

A1_interneurons = np.setdiff1d(pymaid.get_skids_by_annotation('mw A1 neurons paired'), pymaid.get_skids_by_annotation('mw A1 MN'))

A1_G = Analyze_Nx_G(edges=ad_edges_A1)
A1_G_degrees = A1_G.get_node_degrees()
A1_G_degrees = A1_G_degrees.loc[np.intersect1d(A1_interneurons, A1_G_degrees.index), :]
hub_thres = np.round((A1_G_degrees.std()*1.5 + A1_G_degrees.mean()).mean()) # hub threshold >= 1.5 * standard deviation

A1_G_hubs = A1_G.get_node_degrees(hub_threshold=hub_thres).loc[np.intersect1d(A1_interneurons, A1_G_degrees.index), :]
in_hubs = A1_G_hubs[A1_G_hubs.type=='in_hub']
out_hubs = A1_G_hubs[A1_G_hubs.type=='out_hub']
in_out_hubs = A1_G_hubs[A1_G_hubs.type=='in_out_hub']

in_hub_skids = Promat.get_paired_skids(list(in_hubs.index), pairs, unlist=True)
out_hub_skids = Promat.get_paired_skids(list(out_hubs.index), pairs, unlist=True)
in_out_hub_skids = Promat.get_paired_skids(list(in_out_hubs.index), pairs, unlist=True)

pymaid.add_annotations(in_hub_skids, 'mw A1 in_hubs ad')
pymaid.add_annotations(out_hub_skids, 'mw A1 out_hubs ad')
pymaid.add_annotations(in_out_hub_skids, 'mw A1 in_out_hubs ad')

# %%
