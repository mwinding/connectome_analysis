# %%
# generate cascades from each sensory modality

from data_settings import data_date, pairs_path
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import pickle

from contools import Promat, Adjacency_matrix, Celltype_Analyzer, Celltype, Cascade_Analyzer

today_date = '2022-03-15'

# load adjacency matrix for cascades
subgraph = ['mw brain paper clustered neurons', 'mw brain accessory neurons']
adj_ad = Promat.pull_adj(type_adj='ad', data_date=data_date, subgraph=subgraph)

# prep start and stop nodes
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [Celltype(name, Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_skids = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')
output_skids = output_skids + pymaid.get_skids_by_annotation('mw motor')

# %%
# cascades from each sensory modality
# save as pickle to use later because cascades are stochastic; prevents the need to remake plots everytime
p = 0.05
max_hops = 8
n_init = 1000
simultaneous = True
adj=adj_ad

#source_names = order
#input_hist_list = Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = source_names, stop_skids=output_skids, 
#                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous, pairs=pairs, pairwise=True)

#pickle.dump(input_hist_list, open(f'data/cascades/all-sensory-modalities_{n_init}-n_init_{today_date}.p', 'wb'))
input_hist_list = pickle.load(open(f'data/cascades/all-sensory-modalities_{n_init}-n_init_{today_date}.p', 'rb'))

# %%
# generate mega DataFrame with all data, add Cascade_Analyzer objects, and pickle it
from joblib import Parallel, delayed
from tqdm import tqdm
pairs = Promat.get_pairs(pairs_path=pairs_path)

names = [x.name for x in input_hist_list]
skid_hit_hists = [x.skid_hit_hist for x in input_hist_list]

all_data_df = pd.DataFrame([[x] for x in skid_hit_hists], index=names, columns=['skid_hit_hists'])

cascade_objs = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer)(name=all_data_df.index[i], hit_hist=all_data_df.iloc[i, 0], n_init=n_init, pairs=pairs, pairwise=True) for i in tqdm(range(len(all_data_df.index))))
all_data_df['cascade_objs'] = cascade_objs

pickle.dump(all_data_df, open(f'data/cascades/all-sensory-modalities_processed-cascades_{n_init}-n_init_{today_date}.p', 'wb'))