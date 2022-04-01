# %%

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from contools import Promat

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%

# load edges using brain inputs, brain neurons, and brain accessory neurons
subgraph = ['mw brain neurons', 'mw brain accessory neurons']
ad_adj = Promat.pull_adj(type_adj='ad', date=data_date, subgraph=subgraph)
aa_adj = Promat.pull_adj(type_adj='aa', date=data_date, subgraph=subgraph)
dd_adj = Promat.pull_adj(type_adj='dd', date=data_date, subgraph=subgraph)
da_adj = Promat.pull_adj(type_adj='da', date=data_date, subgraph=subgraph)
summed_adj = Promat.pull_adj(type_adj='all-all', date=data_date, subgraph=subgraph)

# %%
# fraction of strong and weak edges

def strong_weak_edges(adj, name, strong=5, weak=2):
    total_edges = sum((adj>0).sum(axis=0))
    strong_edges = sum((adj>=strong).sum(axis=0))
    weak_edges = sum(((adj<=weak)&(adj>0)).sum(axis=0))

    fraction_strong = strong_edges/total_edges
    fraction_weak = weak_edges/total_edges

    print(f'The {name} matrix has {fraction_strong*100:.1f}% strong edges and {fraction_weak*100:.1f}% weak edges')

strong_weak_edges(ad_adj, 'ad')
strong_weak_edges(aa_adj, 'aa')
strong_weak_edges(dd_adj, 'dd')
strong_weak_edges(da_adj, 'da')
strong_weak_edges(summed_adj, 'summed')

def synapses_strong_weak_edges(adj, name, strong=5, weak=2):
    total_synapses = sum(adj[adj>0].sum(axis=0))
    strong_edge_synapses = sum(adj[adj>=strong].sum(axis=0))
    weak_edge_synapses = sum(adj[((adj<=weak)&(adj>0))].sum(axis=0))

    fraction_strong = strong_edge_synapses/total_synapses
    fraction_weak = weak_edge_synapses/total_synapses

    print(f'The {name} matrix has {fraction_strong*100:.1f}% synapses in strong edges and {fraction_weak*100:.1f}% synapses in weak edges')


synapses_strong_weak_edges(ad_adj, 'ad')
synapses_strong_weak_edges(aa_adj, 'aa')
synapses_strong_weak_edges(dd_adj, 'dd')
synapses_strong_weak_edges(da_adj, 'da')
synapses_strong_weak_edges(summed_adj, 'summed')
# %%
