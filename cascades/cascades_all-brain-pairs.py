#%%
import sys
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd

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
max_hops = 8
n_init = 1000
simultaneous = True
adj=adj_ad

pair_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=brain_pair_list, source_names = brain_pairs.leftid, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(pair_hist_list, open(f'data/cascades/all-brain-pairs_{n_init}-n_init.p', 'wb'))

#pair_hist_list = pickle.load(open(f'data/cascades/all-brain-pairs_{n_init}-n_init.p', 'rb'))

# %%
# cascades from descending neurons
# in the above, since descending neurons are stop nodes, they effectively don't emit signal

output_pairs = pm.Promat.load_pairs_from_annotation('outputs', pm.Promat.get_pairs(), skids=output_skids, use_skids=True)
output_pairs_list = [list(output_pairs.loc[i]) for i in output_pairs.index]

output_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=output_pairs_list, source_names = output_pairs.leftid, stop_skids=[],
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(output_hist_list, open(f'data/cascades/all-DN-pairs_{n_init}-n_init.p', 'wb'))

# %%
# combine cascades together

for i in range(0, len(output_hist_list)):
    for j in range(0, len(pair_hist_list)):
        if(output_hist_list[i].name == pair_hist_list[j].name):
            pair_hist_list[j] = output_hist_list[i]

pickle.dump(pair_hist_list, open(f'data/cascades/all-brain-pairs_outputs-added_{n_init}-n_init.p', 'wb'))

# %%
