# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Adjacency_matrix

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

inputs = pd.read_csv('data/adj/inputs_' + data_date + '.csv', index_col=0)
pairs = Promat.get_pairs(pairs_path=pairs_path)
adj_ad = Promat.pull_adj(type_adj='ad', data_date=data_date)
adj_aa = Promat.pull_adj(type_adj='aa', data_date=data_date)
adj_dd = Promat.pull_adj(type_adj='dd', data_date=data_date)
adj_da = Promat.pull_adj(type_adj='da', data_date=data_date)

adj_ad_mat = Adjacency_matrix(adj=adj_ad, input_counts=inputs, mat_type='ad', pairs=pairs).adj_pairwise
adj_aa_mat = Adjacency_matrix(adj=adj_aa, input_counts=inputs, mat_type='aa', pairs=pairs).adj_pairwise
adj_dd_mat = Adjacency_matrix(adj=adj_dd, input_counts=inputs, mat_type='dd', pairs=pairs).adj_pairwise
adj_da_mat = Adjacency_matrix(adj=adj_da, input_counts=inputs, mat_type='da', pairs=pairs).adj_pairwise

# %%
#

# fraction of ad connections
adj_ad_bin = adj_ad_mat.copy()
adj_aa_bin = adj_aa_mat.copy()
adj_dd_bin = adj_dd_mat.copy()
adj_da_bin = adj_da_mat.copy()

adj_ad_bin[adj_ad_bin>0.01]=1
adj_aa_bin[adj_aa_bin>0.01]=1
adj_dd_bin[adj_dd_bin>0.01]=1
adj_da_bin[adj_da_bin>0.01]=1

# determine total number of connections per neuron
connections = adj_ad_bin + adj_aa_bin + adj_dd_bin + adj_da_bin
total_connections = connections.sum(axis=0) + connections.sum(axis=1)

# how many triple and quadruple connections are there per neuron
connections3_output = (connections==3).sum(axis=1)
connections3_input = (connections==3).sum(axis=0)

connections4_output = (connections==4).sum(axis=1)
connections4_input = (connections==4).sum(axis=0)

# identify neurons with 3 simultaneous connections
conn3_output_skids = [x[1] for x in connections3_output[connections3_output>0].index]
conn3_input_skids = [x[1] for x in connections3_input[connections3_input>0].index]
conn3_output_skids = Promat.get_paired_skids(list(conn3_output_skids), Promat.get_pairs(pairs_path=pairs_path), unlist=True)
conn3_input_skids = Promat.get_paired_skids(list(conn3_input_skids), Promat.get_pairs(pairs_path=pairs_path), unlist=True)

# identify neurons with 4 simultaneous connections
conn4_output_skids = [x[1] for x in connections4_output[connections4_output>0].index]
conn4_input_skids = [x[1] for x in connections4_input[connections4_input>0].index]
conn4_output_skids = Promat.get_paired_skids(list(conn4_output_skids), Promat.get_pairs(pairs_path=pairs_path), unlist=True)
conn4_input_skids = Promat.get_paired_skids(list(conn4_input_skids), Promat.get_pairs(pairs_path=pairs_path), unlist=True)

cta = [Celltype('conn3_output', conn3_output_skids), Celltype('conn3_input', conn3_input_skids),
        Celltype('conn4_output', conn4_output_skids), Celltype('conn4_input', conn4_input_skids)]
cta = Celltype_Analyzer(cta)

_, celltypes = Celltype_Analyzer.default_celltypes()
cta.set_known_types(celltypes)
cta.plot_memberships(path='plots/3_or_4-simultaneous-connections_celltypes.pdf', figsize=(2,2), raw_num=True)

cta3 = [Celltype('conn3_output', conn3_output_skids), Celltype('conn3_input', conn3_input_skids)]
cta3 = Celltype_Analyzer(cta3)
cta3.set_known_types(celltypes)
cta3.plot_memberships(path='plots/3-simultaneous-connections_celltypes.pdf', figsize=(1,2), ylim=(0,200), raw_num=True)


cta4 = [Celltype('conn4_output', conn4_output_skids), Celltype('conn4_input', conn4_input_skids)]
cta4 = Celltype_Analyzer(cta4)
cta4.set_known_types(celltypes)
cta4.plot_memberships(path='plots/4-simultaneous-connections_celltypes.pdf', figsize=(1,2), ylim=(0,50), raw_num=True)
# %%
