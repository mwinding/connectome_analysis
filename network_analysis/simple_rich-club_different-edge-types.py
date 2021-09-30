# %%
#
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm

rm = pymaid.CatmaidInstance(url, token, name, password)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load previously generated edge lists with pairwise threshold
# see 'network_analysis/generate_all_edges.py'

adj_names = ['summed', 'ad', 'aa', 'dd', 'da']
edge_lists = [pd.read_csv(f'data/edges_threshold/pairwise-threshold_{name}_all-edges.csv', index_col = 0) for name in adj_names]
G, Gad, Gaa, Gdd, Gda = [pg.Analyze_Nx_G(edge_list, graph_type='undirected', split_pairs=True) for edge_list in edge_lists]
Gs = [G, Gad, Gaa, Gdd, Gda]

# %%
# rich club coefficient

rc_coeffs = []
for graph in Gs:
    rc = nx.rich_club_coefficient(graph.G, normalized=False, Q=100, seed=1)
    rc = [x[1] for x in rc.items()]
    rc_coeffs.append(rc)

rc_data = []
for i, rc_coeff in enumerate(rc_coeffs):
    data = [[x, j, adj_names[i]] for j, x in enumerate(rc_coeff)]
    rc_data.append(data)

rc_data = [x for sublist in rc_data for x in sublist]
rc_data = pd.DataFrame(rc_data, columns = ['rich_club_coeff', 'degree', 'type'])

# %%
# plot data

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(data=rc_data, x='degree', y='rich_club_coeff', hue='type', ax=ax, linewidth=0.5)
#ax.set(xlim=(0,80))
#ax.set(ylim=(0, .8))
plt.savefig(f'network_analysis/plots/rich-club_pair-threshold-graph_all-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
# rich club no threshold

adj_names = ['G', 'Gad', 'Gaa', 'Gdd', 'Gda']
G = nx.readwrite.graphml.read_graphml('data/graphs/G.graphml', node_type=int).to_undirected()
Gad = nx.readwrite.graphml.read_graphml('data/graphs/Gad.graphml', node_type=int).to_undirected()
Gaa = nx.readwrite.graphml.read_graphml('data/graphs/Gaa.graphml', node_type=int).to_undirected()
Gdd = nx.readwrite.graphml.read_graphml('data/graphs/Gdd.graphml', node_type=int).to_undirected()
Gda = nx.readwrite.graphml.read_graphml('data/graphs/Gda.graphml', node_type=int).to_undirected()

G.remove_edges_from(nx.selfloop_edges(G))
Gad.remove_edges_from(nx.selfloop_edges(Gad))
Gaa.remove_edges_from(nx.selfloop_edges(Gaa))
Gdd.remove_edges_from(nx.selfloop_edges(Gdd))
Gda.remove_edges_from(nx.selfloop_edges(Gda))

Gs = [G, Gad, Gaa, Gdd, Gda]

rc_coeffs = []
for graph in Gs:
    rc = nx.rich_club_coefficient(graph, normalized=False, Q=100, seed=1)
    rc = [x[1] for x in rc.items()]
    rc_coeffs.append(rc)

rc_data = []
for i, rc_coeff in enumerate(rc_coeffs):
    data = [[x, j, adj_names[i]] for j, x in enumerate(rc_coeff)]
    rc_data.append(data)

rc_data = [x for sublist in rc_data for x in sublist]
rc_data = pd.DataFrame(rc_data, columns = ['rich_club_coeff', 'degree', 'type'])

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.lineplot(data=rc_data, x='degree', y='rich_club_coeff', hue='type', ax=ax, linewidth=0.5)
#ax.set(xlim=(0,80))
#ax.set(ylim=(0, .8))
plt.savefig(f'network_analysis/plots/rich-club_unfiltered-graph_all-edge-types.pdf', format='pdf', bbox_inches='tight')

# %%
