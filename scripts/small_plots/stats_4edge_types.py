#%%

from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from contools import Promat, Prograph

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

pairs = Promat.get_pairs(pairs_path=pairs_path) # import pairs

Gad = nx.readwrite.graphml.read_graphml(f'data/processed/{data_date}/Gad.graphml', node_type=int)
Gaa = nx.readwrite.graphml.read_graphml(f'data/processed/{data_date}/Gaa.graphml', node_type=int)
Gdd = nx.readwrite.graphml.read_graphml(f'data/processed/{data_date}/Gdd.graphml', node_type=int)
Gda = nx.readwrite.graphml.read_graphml(f'data/processed/{data_date}/Gda.graphml', node_type=int)

G_types = [Gad, Gaa, Gdd, Gda]
G_names = ['Axon-Dendrite', 'Axon-Axon', 'Dendrite-Dendrite', 'Dendrite-Axon']

brain = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
brain = list(np.setdiff1d(brain, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# include only brain inputs, interneurons, and outputs
G_types = [G.subgraph(np.intersect1d(list(G.nodes), brain)) for G in G_types]

# %%
# general stats

node_counts = [len(G.nodes) for G in G_types]
node_counts = [f'{G_names[i]}: {node_counts[i]}' for i in range(0, len(G_types))]

print('\n** Node Counts **')
for node_count in node_counts:
    print(node_count)

edge_counts = [len(G.edges) for G in G_types]
all_possible_edges = [len(G.nodes)**2 for G in G_types]
density = [edge_count/all_possible*100 for edge_count, all_possible in zip(edge_counts, all_possible_edges)]
density = [f'{G_names[i]}: {density[i]:.2f}%' for i in range(0, len(density))]

print('\n** Density **')
for dens in density:
    print(dens)

max_degrees = [max([x[1] for x in list(G.degree)]) for G in G_types]
max_degrees = [f'{G_names[i]}: {max_degrees[i]}' for i in range(0, len(max_degrees))]

print('\n** Max Degree **')
for max_degree in max_degrees:
    print(max_degree)

median_degrees = [np.median([x[1] for x in list(G.degree)]) for G in G_types]
median_degrees = [f'{G_names[i]}: {int(median_degrees[i])}' for i in range(0, len(median_degrees))]

print('\n** Median Degree **')
for median_degree in median_degrees:
    print(median_degree)
# %%
# quantification of different edge and synapse types
import squarify 
color = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2], sns.color_palette()[3]]
alpha = 1

edge_counts = [len(G.edges) for G in G_types]
total_edges = sum(edge_counts)
fraction_edges = [edge_count/total for edge_count, total in zip(edge_counts, [total_edges]*len(edge_counts))]
fraction_edges_label = [f'{round(fraction_edges[i]*100)}%' for i in range(0, len(fraction_edges))]

synapse_counts = [list(Prograph.path_edge_attributes(G, edge, 'weight')[0]) + [G_names[i]] for i, G in enumerate(G_types) for edge in list(G.edges)]
synapse_counts = pd.DataFrame(synapse_counts, columns = ['pre', 'post', 'weight', 'type'])
synapse_counts = synapse_counts.groupby('type').sum().iloc[[1,0,3,2], :] #sum types and reorder
fraction_synapses = synapse_counts.weight/sum(synapse_counts.weight)
fraction_synapses_label = [f'{int(round(fraction_synapses[i]*100))}%' for i in range(0, len(fraction_synapses))]

fig, axs = plt.subplots(1,2, figsize=(2.5,1))
ax = axs[0]
ax.set_axis_off()
ax.set(title='Edge Types')
squarify.plot(fraction_edges, label = fraction_edges_label, color=color, alpha=alpha, text_kwargs={'color':'white'}, ax=ax)

ax = axs[1]
ax.set_axis_off()
ax.set(title='Synapse Types')
squarify.plot(fraction_synapses, label = fraction_synapses_label, color=color, alpha=alpha, text_kwargs={'color':'white'}, ax=ax)
plt.savefig('plots/treemap_fraction-edges_fraction-synapses.pdf', format='pdf', bbox_inches='tight')


fig, axs = plt.subplots(1,2, figsize=(2.5,1))
ax = axs[0]
ax.set_axis_off()
ax.set(title='Edge Types')
squarify.plot(fraction_edges, label = [f'{round(x/1000)}k' for x in edge_counts], color=color, alpha=alpha, text_kwargs={'color':'white'}, ax=ax)

ax = axs[1]
ax.set_axis_off()
ax.set(title='Synapse Types')
squarify.plot(fraction_synapses, label = [f'{round(x/1000)}k' for x in synapse_counts.weight], color=color, alpha=alpha, text_kwargs={'color':'white'}, ax=ax)
plt.savefig('plots/treemap_count-edges_count-synapses.pdf', format='pdf', bbox_inches='tight')
# %%
# counts for edge width in figure

# raw numbers with max 3 point arrow
fraction_synapses/fraction_synapses[0]*3
np.array(fraction_edges)/fraction_edges[0]*3

# log numbers with max 3 point arrow (Edges)
log_edge_counts = np.array([np.log(x) for x in edge_counts])
log_edge_counts/log_edge_counts[0]*3

# log numbers with max 3 point arrow (synapses)
log_synapse_counts = np.array([np.log(x) for x in synapse_counts.weight])
log_synapse_counts/log_synapse_counts[0]*3

# %%
# 
