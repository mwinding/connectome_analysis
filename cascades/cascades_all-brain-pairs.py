#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cmasher as cmr

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot 

import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm

# load adjacency matrix for cascades
adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

# prep start and stop nodes
brain_pairs = pm.Promat.load_pairs_from_annotation('mw brain neurons', pm.Promat.get_pairs())
brain_pair_list = [list(brain_pairs.loc[i]) for i in brain_pairs.index]

output_skids = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

#%%
# cascades from each neuron pair
# save as pickle to use later because cascades are stochastic; prevents the need to remake plots everytime
import pickle

p = 0.05
max_hops = 10
n_init = 1000
simultaneous = True
adj=adj_ad

pair_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=brain_pair_list, source_names = brain_pairs.leftid, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(pair_hist_list, open(f'data/cascades/all-brain-pairs_{n_init}-n_init.p', 'wb'))

#pair_hist_list = pickle.load(open(f'data/cascades/all-brain-pairs_{n_init}-n_init.p', 'rb'))

# %%
