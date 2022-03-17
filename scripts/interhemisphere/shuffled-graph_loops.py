#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random
import networkx as nx

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

#rm = pymaid.CatmaidInstance(url, token, name, password)

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csv/all_paired_edges.csv', index_col=0)

# build networkx Graph
graph = pg.Analyze_Nx_G(all_edges_combined)

# %%
# types of loops observed in graph

cutoff = 3
pairs = list(np.unique(all_edges_combined.upstream_pair_id))
observed_loops = graph.identify_loops(pairs, cutoff)

# %%
# shuffled graphs
from tqdm import tqdm
from joblib import Parallel, delayed

# build randomized networkx Graph
shuffled_graphs = graph.generate_shuffled_graphs(100, graph_type='directed')
shuffled_graphs = [pg.Analyze_Nx_G(edges=x.edges, graph=x) for x in shuffled_graphs]
shuffled_graphs_loops = Parallel(n_jobs=-1)(delayed(shuffled_graphs[i].identify_loops)(pairs, cutoff) for i in tqdm(range(len(shuffled_graphs))))

#nx.readwrite.graphml.write_graphml(graph.G, 'interhemisphere/csv/control_graph.graphml')
#shuffled_graphs_loops = Parallel(n_jobs=-1)(delayed(nx.readwrite.graphml.write_graphml)(shuffled_graphs[i].G, f'interhemisphere/csv/shuffled_graphs/iteration-{i}.graphml') for i in tqdm(range(len(shuffled_graphs))))
#graph2 = nx.readwrite.graphml.read_graphml('interhemisphere/csv/shuffled_graphs.graphml', node_type=int, edge_key_type=str)
#shuffled_graphs_loops = Parallel(n_jobs=-1)(delayed(nx.readwrite.graphml.read_graphml)(f'interhemisphere/csv/shuffled_graphs/iteration-{i}.graphml', node_type=int, edge_key_type=str) for i in tqdm(range(len(100))))

# %%
# plot data

contains_loop = [[1-x.iloc[0], 'shuffled'] for x in shuffled_graphs_loops]
contains_loop.append([1-observed_loops.iloc[0], 'observed'])

contains_loop = pd.DataFrame(contains_loop, columns = ['Fraction Contains Loops', 'Condition'])

height = 1.5
width = .8
with plt.rc_context({"lines.linewidth": 0.5}):
    g = sns.catplot(data=contains_loop, x='Condition', y='Fraction Contains Loops', kind='bar', height=height, aspect=width/height)
    g.ax.set(ylim=(0,1))
    plt.xticks(rotation=45, ha='right')
    plt.savefig('interhemisphere/plots/loops/all_loops_vs_shuffled.pdf', format='pdf', bbox_inches='tight')


contains_loop_1pair = [[x.loc[(1, 'pair')], 'shuffled', 1] for x in shuffled_graphs_loops]
contains_loop_2pair = [[x.loc[(2, 'pair')], 'shuffled', 2] for x in shuffled_graphs_loops]
contains_loop_3pair = [[x.loc[(3, 'pair')], 'shuffled', 3] for x in shuffled_graphs_loops]
contains_loop_1pair.append([observed_loops.loc[(1, 'pair')], 'observed', 1])
contains_loop_2pair.append([observed_loops.loc[(2, 'pair')], 'observed', 2])
contains_loop_3pair.append([observed_loops.loc[(3, 'pair')], 'observed', 3])

contains_loop_1self = [[x.loc[(1, 'self')], 'shuffled', 1] for x in shuffled_graphs_loops]
contains_loop_2self = [[x.loc[(2, 'self')], 'shuffled', 2] for x in shuffled_graphs_loops]
contains_loop_3self = [[x.loc[(3, 'self')], 'shuffled', 3] for x in shuffled_graphs_loops]
contains_loop_1self.append([observed_loops.loc[(1, 'self')], 'observed', 1])
contains_loop_2self.append([observed_loops.loc[(2, 'self')], 'observed', 2])
contains_loop_3self.append([observed_loops.loc[(3, 'self')], 'observed', 3])

data_self = pd.DataFrame(np.concatenate((contains_loop_1self, contains_loop_2self, contains_loop_3self)), columns = ['fraction_loops', 'condition', 'path_length'])
data_self.fraction_loops = [float(x) for x in data_self.fraction_loops]
data_self.to_csv('interhemisphere/csv/self_loops_vs_shuffled.csv')
data_self = pd.read_csv('interhemisphere/csv/self_loops_vs_shuffled.csv')

# box plots
height = 1.5
width = 1.25
g = sns.catplot(data=data_self, x='path_length', y='fraction_loops', hue='condition', kind='box', height=height, aspect=width/height, linewidth=0.5, fliersize = 0.1)
g.ax.set(ylim=(0,0.5))
plt.savefig('interhemisphere/plots/loops/self_loops_vs_shuffled.pdf', format='pdf', bbox_inches='tight')

data_pair = pd.DataFrame(np.concatenate((contains_loop_1pair, contains_loop_2pair, contains_loop_3pair)), columns = ['fraction_loops', 'condition', 'path_length'])
data_pair.fraction_loops = [float(x) for x in data_pair.fraction_loops]
data_pair.to_csv('interhemisphere/csv/pair_loops_vs_shuffled.csv')
data_pair = pd.read_csv('interhemisphere/csv/pair_loops_vs_shuffled.csv')

height = 1.5
width = 1.25
g = sns.catplot(data=data_pair, x='path_length', y='fraction_loops', hue='condition', kind='box', height=height, aspect=width/height, linewidth=0.5, fliersize = 0.1)
g.ax.set(ylim=(0,0.5))
plt.savefig('interhemisphere/plots/loops/pair_loops_vs_shuffled.pdf', format='pdf', bbox_inches='tight')

# box plots
height = 0.8
width = 1.1
with plt.rc_context({"lines.linewidth": 0.5}):
    g = sns.catplot(data=data_self, x='path_length', y='fraction_loops', hue='condition', kind='bar', height=height, aspect=width/height, linewidth=0.5)
    g.ax.set(ylim=(0,0.25))
    plt.savefig('interhemisphere/plots/loops/self_loops_vs_shuffled_bar.pdf', format='pdf', bbox_inches='tight')

height = 0.8
width = 1.1
with plt.rc_context({"lines.linewidth": 0.5}):
    g = sns.catplot(data=data_pair, x='path_length', y='fraction_loops', hue='condition', kind='bar', height=height, aspect=width/height, linewidth=0.5)
    g.ax.set(ylim=(0,0.25))
    plt.savefig('interhemisphere/plots/loops/pair_loops_vs_shuffled_bar.pdf', format='pdf', bbox_inches='tight')

# %%
