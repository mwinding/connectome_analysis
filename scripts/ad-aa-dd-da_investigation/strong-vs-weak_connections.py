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
ad_adj = Promat.pull_adj(type_adj='ad', data_date=data_date, subgraph=subgraph)
aa_adj = Promat.pull_adj(type_adj='aa', data_date=data_date, subgraph=subgraph)
dd_adj = Promat.pull_adj(type_adj='dd', data_date=data_date, subgraph=subgraph)
da_adj = Promat.pull_adj(type_adj='da', data_date=data_date, subgraph=subgraph)
summed_adj = Promat.pull_adj(type_adj='all-all', data_date=data_date, subgraph=subgraph)

# %%
# fraction of strong and weak edges

def strong_weak_edges(adj, name, strong=5, weak=2):
    total_edges = sum((adj>0).sum(axis=0))
    strong_edges = sum((adj>=strong).sum(axis=0))
    weak_edges = sum(((adj<=weak)&(adj>0)).sum(axis=0))

    fraction_strong = strong_edges/total_edges
    fraction_weak = weak_edges/total_edges

    print(f'The {name} matrix has {fraction_strong*100:.1f}% strong edges and {fraction_weak*100:.1f}% weak edges')
    return(fraction_strong, fraction_weak)

ad_strong_edges, ad_weak_edges = strong_weak_edges(ad_adj, 'ad')
aa_strong_edges, aa_weak_edges = strong_weak_edges(aa_adj, 'aa')
dd_strong_edges, dd_weak_edges = strong_weak_edges(dd_adj, 'dd')
da_strong_edges, da_weak_edges = strong_weak_edges(da_adj, 'da')
sum_strong_edges, sum_weak_edges = strong_weak_edges(summed_adj, 'summed')

def synapses_strong_weak_edges(adj, name, strong=5, weak=2):
    total_synapses = sum(adj[adj>0].sum(axis=0))
    strong_edge_synapses = sum(adj[adj>=strong].sum(axis=0))
    weak_edge_synapses = sum(adj[((adj<=weak)&(adj>0))].sum(axis=0))

    fraction_strong = strong_edge_synapses/total_synapses
    fraction_weak = weak_edge_synapses/total_synapses

    print(f'The {name} matrix has {fraction_strong*100:.1f}% synapses in strong edges and {fraction_weak*100:.1f}% synapses in weak edges')
    return(fraction_strong, fraction_weak)


ad_strong_synapses, ad_weak_synapses = synapses_strong_weak_edges(ad_adj, 'ad')
aa_strong_synapses, aa_weak_synapses = synapses_strong_weak_edges(aa_adj, 'aa')
dd_strong_synapses, dd_weak_synapses = synapses_strong_weak_edges(dd_adj, 'dd')
da_strong_synapses, da_weak_synapses = synapses_strong_weak_edges(da_adj, 'da')
sum_strong_synapses, sum_weak_synapses = synapses_strong_weak_edges(summed_adj, 'summed')

# %%
# plot strong vs weak data

df = pd.DataFrame([[ad_strong_edges, 'ad', 'strong'], 
                    [aa_strong_edges, 'aa', 'strong'], 
                    [dd_strong_edges, 'dd', 'strong'],
                    [da_strong_edges, 'da', 'strong'],
                    [sum_strong_edges, 'summed', 'strong'],
                    [ad_weak_edges, 'ad', 'weak'], 
                    [aa_weak_edges, 'aa', 'weak'], 
                    [dd_weak_edges, 'dd', 'weak'],
                    [da_weak_edges, 'da', 'weak'],
                    [sum_weak_edges, 'summed', 'weak']],
                    columns=['edges', 'edge_type', 'edge_strength'])

fig, axs = plt.subplots(2,1, figsize=(1.5, 3))
fig.tight_layout(pad=1)
ax = axs[0]
sns.barplot(x=df.edge_type, y=df.edges, hue=df.edge_strength, ax=ax)
ax.set(ylim=(0,1), yticks=[0, .2, .4, .6, .8, 1])

df = pd.DataFrame([[ad_strong_synapses, 'ad', 'strong'], 
                    [aa_strong_synapses, 'aa', 'strong'], 
                    [dd_strong_synapses, 'dd', 'strong'],
                    [da_strong_synapses, 'da', 'strong'],
                    [sum_strong_synapses, 'summed', 'strong'],
                    [ad_weak_synapses, 'ad', 'weak'], 
                    [aa_weak_synapses, 'aa', 'weak'], 
                    [dd_weak_synapses, 'dd', 'weak'],
                    [da_weak_synapses, 'da', 'weak'],
                    [sum_weak_synapses, 'summed', 'weak']],
                    columns=['synapses', 'edge_type', 'edge_strength'])

ax = axs[1]
sns.barplot(x=df.edge_type, y=df.synapses, hue=df.edge_strength, ax=ax)
ax.set(ylim=(0,1), yticks=[0, .2, .4, .6, .8, 1])
plt.savefig('plots/synapses-in-edges_summary.pdf', format='pdf', bbox_inches='tight')

# %%
# distribution of synapse numbers

# calculating fraction of synapses within edges of certain synaptic strength
# syn_max value decides where to cut the distribution and combine the rest for plotting purposes
# i.e. edges with [syn=1, 2, 3, 4, >=5] when syn_max=5
def synapses_in_edges(adj, name, syn_max=5):

    total_synapses = sum(adj[adj>0].sum(axis=0))

    all_fractions = []
    for i in range(1,syn_max+1):
        if(i==syn_max): synapses = sum(adj[adj>=i].sum(axis=0))
        else: synapses = sum(adj[adj==i].sum(axis=0))
        fraction = synapses/total_synapses
        all_fractions.append(fraction)

        if(i==syn_max): print(f'The {name} matrix has {fraction*100:.1f}% synapses in >={i}-strength edges')
        else: print(f'The {name} matrix has {fraction*100:.1f}% synapses in {i}-strength edges')

    return(pd.DataFrame(list(zip(all_fractions, range(1, len(all_fractions)+1),[name]*len(all_fractions)))))
    
ad_synapses = synapses_in_edges(ad_adj, 'ad', syn_max=5)
aa_synapses = synapses_in_edges(aa_adj, 'aa', syn_max=5)
dd_synapses = synapses_in_edges(dd_adj, 'dd', syn_max=5)
da_synapses = synapses_in_edges(da_adj, 'da', syn_max=5)
sum_synapses = synapses_in_edges(summed_adj, 'summed', syn_max=5)

# calculating the fraction of edges with certain synaptic strength
# syn_max value decides where to cut the distribution and combine the rest for plotting purposes
# i.e. edges with [syn=1, 2, 3, 4, >=5] when syn_max=5
def edges_by_synapses(adj, name, syn_max=5):
    
    total_edges = sum((adj>0).sum(axis=0))

    all_fractions = []
    for i in range(1, syn_max+1):
        if(i==syn_max): edges = sum((adj>=syn_max).sum(axis=0))
        else: edges = sum((adj==i).sum(axis=0))
        fraction = edges/total_edges
        all_fractions.append(fraction)

        if(i==syn_max): print(f'The {name} matrix has {fraction*100:.1f}% edges with >={i} synaptic strength')
        else: print(f'The {name} matrix has {fraction*100:.1f}% edges with {i} synaptic strength')

    return(pd.DataFrame(list(zip(all_fractions, range(1, len(all_fractions)+1),[name]*len(all_fractions)))))

ad_edges = edges_by_synapses(ad_adj, 'ad', syn_max=5)
aa_edges = edges_by_synapses(aa_adj, 'aa', syn_max=5)
dd_edges = edges_by_synapses(dd_adj, 'dd', syn_max=5)
da_edges = edges_by_synapses(da_adj, 'da', syn_max=5)
sum_edges = edges_by_synapses(summed_adj, 'summed', syn_max=5)

# %%
# plot distribution of synapses in edges or edges by synapses

df = pd.concat([ad_edges, aa_edges, dd_edges, da_edges])
df.columns = ['fraction', 'syn_strength', 'edge_type']

fig, axs = plt.subplots(2,1, figsize=(1.5, 3))
ax = axs[0]
sns.barplot(x=df.syn_strength, y=df.fraction, hue=df.edge_type, ax=ax)
ax.set(ylim=(0,1), yticks=[0, .2, .4, .6, .8, 1])

df = pd.concat([ad_synapses, aa_synapses, dd_synapses, da_synapses])
df.columns = ['fraction', 'syn_strength', 'edge_type']

ax = axs[1]
sns.barplot(x=df.syn_strength, y=df.fraction, hue=df.edge_type, ax=ax)
ax.set(ylim=(0,1), yticks=[0, .2, .4, .6, .8, 1])
plt.savefig('plots/synapses-in-edges_distribution.pdf', format='pdf', bbox_inches='tight')

# %%
