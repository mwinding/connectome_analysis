#%%
# do local neurons (LNs) or projection neurons (PNs)

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
import connectome_tools.process_skeletons as skel
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accessory')
adj_aa = pm.Promat.pull_adj(type_adj='aa', subgraph='brain and accessory')
adj_dd = pm.Promat.pull_adj(type_adj='dd', subgraph='brain and accessory')
adj_da = pm.Promat.pull_adj(type_adj='da', subgraph='brain and accessory')

LN = pymaid.get_skids_by_annotation('mw LN')
PN = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain PNs')


# %%
# prep data

PN_index = np.intersect1d(PN, adj_ad.index)
PN_df = pd.DataFrame(list(zip(['ad_outputs']*len(PN_index), adj_ad.loc[PN_index, :].sum(axis=1).values)))
PN_df = PN_df.append(list(zip(['aa_outputs']*len(PN_index), adj_aa.loc[PN_index, :].sum(axis=1).values)))
PN_df = PN_df.append(list(zip(['dd_outputs']*len(PN_index), adj_dd.loc[PN_index, :].sum(axis=1).values)))
PN_df = PN_df.append(list(zip(['da_outputs']*len(PN_index), adj_da.loc[PN_index, :].sum(axis=1).values)))
PN_df = PN_df.append(list(zip(['ad_inputs']*len(PN_index), adj_ad.loc[:, PN_index].sum(axis=0).values)))
PN_df = PN_df.append(list(zip(['aa_inputs']*len(PN_index), adj_aa.loc[:, PN_index].sum(axis=0).values)))
PN_df = PN_df.append(list(zip(['dd_inputs']*len(PN_index), adj_dd.loc[:, PN_index].sum(axis=0).values)))
PN_df = PN_df.append(list(zip(['da_inputs']*len(PN_index), adj_da.loc[:, PN_index].sum(axis=0).values)))
PN_df['celltype'] = ['PN']*len(PN_df.index)

LN_index = np.intersect1d(LN, adj_ad.index)
LN_df = pd.DataFrame(list(zip(['ad_outputs']*len(LN_index), adj_ad.loc[LN_index, :].sum(axis=1).values)))
LN_df = LN_df.append(list(zip(['aa_outputs']*len(LN_index), adj_aa.loc[LN_index, :].sum(axis=1).values)))
LN_df = LN_df.append(list(zip(['dd_outputs']*len(LN_index), adj_dd.loc[LN_index, :].sum(axis=1).values)))
LN_df = LN_df.append(list(zip(['da_outputs']*len(LN_index), adj_da.loc[LN_index, :].sum(axis=1).values)))
LN_df = LN_df.append(list(zip(['ad_inputs']*len(LN_index), adj_ad.loc[:, LN_index].sum(axis=0).values)))
LN_df = LN_df.append(list(zip(['aa_inputs']*len(LN_index), adj_aa.loc[:, LN_index].sum(axis=0).values)))
LN_df = LN_df.append(list(zip(['dd_inputs']*len(LN_index), adj_dd.loc[:, LN_index].sum(axis=0).values)))
LN_df = LN_df.append(list(zip(['da_inputs']*len(LN_index), adj_da.loc[:, LN_index].sum(axis=0).values)))
LN_df['celltype'] = ['LN']*len(LN_df.index)

df = pd.concat([PN_df, LN_df], axis=0)
df.columns = ['edge_type', 'synapses', 'celltype']

fig, ax = plt.subplots(1,1)
sns.barplot(data = df, x='edge_type', y='synapses', hue='celltype', ax=ax)
plt.savefig('identify_neuron_classes/plots/LN-PN_noncanonical-types.pdf')

# %%
# t-tests 

from scipy.stats import ttest_ind
adjs = [adj_ad, adj_aa, adj_dd, adj_da]
output_ttests = [ttest_ind(adj.loc[PN_index, :].sum(axis=1).values, adj.loc[LN_index, :].sum(axis=1).values).pvalue for adj in adjs]
input_ttests = [ttest_ind(adj.loc[:, PN_index].sum(axis=0).values, adj.loc[:, LN_index].sum(axis=0).values).pvalue for adj in adjs]

print(output_ttests)
print(input_ttests)
# %%
