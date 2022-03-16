# %%
# generate cascades from individual neuron pairs and individual nonpaired neurons

from data_settings import data_date, pairs_path
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import pickle

from contools import Promat, Adjacency_matrix, Celltype_Analyzer, Celltype, Cascade_Analyzer

today_date = '2022-03-15'
# %%
# load adjacency matrix for cascades
subgraph = ['mw brain paper clustered neurons', 'mw brain accessory neurons']
adj_ad = Promat.pull_adj(type_adj='ad', date=data_date, subgraph=subgraph)

# prep start and stop nodes
skids = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
brain_pairs, brain_unpaired, brain_nonpaired = Promat.extract_pairs_from_list(skids, Promat.get_pairs(pairs_path))

brain_pair_list = [list(brain_pairs.loc[i]) for i in brain_pairs.index]
brain_nonpaired_list = [list(brain_nonpaired.loc[i]) for i in brain_nonpaired.index]
brain_pair_list = brain_pair_list + brain_nonpaired_list

output_skids = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')
output_skids = output_skids + pymaid.get_skids_by_annotation('mw motor')

# %%
# cascades from each neuron pair
# save as pickle to use later because cascades are stochastic; prevents the need to remake plots everytime
p = 0.05
max_hops = 8
n_init = 1000
simultaneous = True
adj=adj_ad

source_names = [x[0] for x in brain_pair_list]
pair_hist_list = Cascade_Analyzer.run_cascades_parallel(source_skids_list=brain_pair_list, source_names = source_names, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(pair_hist_list, open(f'data/cascades/all-brain-pairs_{n_init}-n_init_{today_date}.p', 'wb'))

# %%
# cascades from descending neurons
# in the above, since descending neurons are stop nodes, they effectively don't emit signal
output_pairs, _, output_nonpaired = Promat.extract_pairs_from_list(Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs'), Promat.get_pairs(pairs_path))
output_pairs_list = [list(output_pairs.loc[i]) for i in output_pairs.index]
output_pairs_list = output_pairs_list + [list(output_nonpaired.loc[i]) for i in output_nonpaired.index]

output_hist_list = Cascade_Analyzer.run_cascades_parallel(source_skids_list=output_pairs_list, source_names=[x[0] for x in output_pairs_list], stop_skids=[],
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(output_hist_list, open(f'data/cascades/all-DN-pairs_{n_init}-n_init.p', 'wb'))

# combine cascades together
for i in range(0, len(output_hist_list)):
    for j in range(0, len(pair_hist_list)):
        if(output_hist_list[i].name == pair_hist_list[j].name):
            pair_hist_list[j] = output_hist_list[i]

pickle.dump(pair_hist_list, open(f'data/cascades/all-brain-pairs_outputs-added_{n_init}-n_init.p', 'wb'))

# %%
