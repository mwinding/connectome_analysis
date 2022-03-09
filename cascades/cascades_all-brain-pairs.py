#%%

from data_settings import data_date, pairs_path
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import pickle

from contools import Promat, Adjacency_matrix, Celltype_Analyzer, Celltype, Cascade_Analyzer

today_date = '2022-03-08'

# load adjacency matrix for cascades
#adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory') # old data

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

#%%
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

#pair_hist_list = pickle.load(open(f'data/cascades/all-brain-pairs_{n_init}-n_init.p', 'rb'))

# %%
# cascades from descending neurons
# in the above, since descending neurons are stop nodes, they effectively don't emit signal

output_pairs = Promat.load_pairs_from_annotation('outputs', Promat.get_pairs(pairs_path), skids=output_skids, use_skids=True)
output_pairs_list = [list(output_pairs.loc[i]) for i in output_pairs.index]

output_hist_list = Cascade_Analyzer.run_cascades_parallel(source_skids_list=output_pairs_list, source_names = output_pairs.leftid, stop_skids=[],
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
#### the code below should not be used if data is regenerated in the future
# add nonpaired to old dataset

# old imports
import sys
sys.path.append('/Users/mwinding/repos/maggot_models')
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)
import numpy as np
import pandas as pd
import pickle
import connectome_tools.cascade_analysis as casc
import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

skids = pymaid.get_skids_by_annotation('mw brain neurons')
skids = list(np.setdiff1d(skids, pymaid.get_skids_by_annotation('mw partially differentiated')))
brain_pairs, brain_unpaired, brain_nonpaired = pm.Promat.extract_pairs_from_list(skids, pm.Promat.get_pairs())

brain_nonpaired_list = [list(brain_nonpaired.loc[i]) for i in brain_nonpaired.index]

output_skids = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

# %%
p = 0.05
max_hops = 8
n_init = 1000
simultaneous = True
adj=adj_ad

source_names = [x[0] for x in brain_nonpaired_list]
nonpairs_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=brain_nonpaired_list, source_names = source_names, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(nonpairs_hist_list, open(f'data/cascades/all-brain-nonpaired_{n_init}-n_init.p', 'wb'))

# %%
# combine with old dataset

old_pair_hist_list = pickle.load(open(f'data/cascades/all-brain-pairs_outputs-added_{n_init}-n_init.p', 'rb'))
nonpairs_hist_list = pickle.load(open(f'data/cascades/all-brain-nonpaired_{n_init}-n_init.p', 'rb'))

combined_pair_hist_list = nonpairs_hist_list + old_pair_hist_list

# %%
