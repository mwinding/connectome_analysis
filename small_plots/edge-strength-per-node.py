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
import networkx as nx
import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)

brain = pymaid.get_skids_by_annotation('mw brain neurons')
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

#ad = pd.read_csv('data/edges/Gad_edgelist.txt')
Gad = nx.readwrite.graphml.read_graphml('data/edges/Gad.graphml', node_type=int)
Gaa = nx.readwrite.graphml.read_graphml('data/edges/Gaa.graphml', node_type=int)
Gdd = nx.readwrite.graphml.read_graphml('data/edges/Gdd.graphml', node_type=int)
Gda = nx.readwrite.graphml.read_graphml('data/edges/Gda.graphml', node_type=int)

G_types = [Gad, Gaa, Gdd, Gda]
G_names = ['Axon-Dendrite', 'Axon-Axon', 'Dendrite-Dendrite', 'Dendrite-Axon']
# %%
def pull_edges(G, skid, attribute, edge_type):
    if(edge_type=='out'):
        out_edges = [pg.Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.out_edges(skid))]
        return(out_edges)
    if(edge_type=='in'):
        in_edges = [pg.Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.in_edges(skid))]
        return(in_edges)

# %%
# output edges
max_bin = 10
plot_type = 'count' #count or fraction
if(plot_type == 'count'):
    max_y = 10
if(plot_type == 'fraction'):
    max_y = 1.0

fig, axs = plt.subplots(2,2, figsize=(5,5))
for name_i, G in enumerate(G_types):
    out_edges = []
    for node in G.nodes:
        out_edges.append(pull_edges(G, node, 'weight', 'out'))

    all_out_edges = [x for sublist in out_edges for x in sublist]
    all_out_edges = pd.DataFrame(all_out_edges, columns = ['pre', 'post', 'weight'])
    all_out_edges = all_out_edges.set_index('pre')

    # generate histogram of edge strengths per neuron
    hists = []
    binwidth = 1
    x_range = list(range(0, int(max(all_out_edges.weight))))
    for i, skid in enumerate(np.unique(all_out_edges.index)):
        if(type(all_out_edges.loc[skid])!=pd.Series): # ignores nodes with only one edge
            total_edges = len(all_out_edges.loc[skid])
            data = all_out_edges.loc[skid].weight
            bins = np.arange(1, max_bin + binwidth + 0.5) - 0.5
            bins = np.append(bins, 500)
            hist = np.histogram(data, bins=bins)
            for hist_pair in zip(hist[0], hist[0]/total_edges, [int(x) for x in (bins+0.5)], [skid]*len(hist[0])):
                hists.append(hist_pair)

    hists = pd.DataFrame(hists, columns = ['count', 'fraction', 'bin', 'skid'])

    if(name_i==0):
        ax = axs[0,0]
    if(name_i==1):
        ax = axs[0,1]
    if(name_i==2):
        ax = axs[1,0]
    if(name_i==3):
        ax = axs[1,1]
    sns.barplot(data=hists, x='bin', y=plot_type, ax=ax, errwidth = 1)
    ax.set(xlabel = 'Synaptic Strength', xlim=(-1, max_bin+1), ylim=(0, max_y))
    ax.set_title(f'{G_names[name_i]} Output Edges', y = 0.85)
plt.savefig(f'small_plots/plots/outputs_edge-strength-per-node_{plot_type}.pdf', bbox_inches='tight')

# %%
# input edges
max_bin = 10
plot_type = 'count' #count or fraction
if(plot_type == 'count'):
    max_y = 10
if(plot_type == 'fraction'):
    max_y = 1.0

fig, axs = plt.subplots(2,2, figsize=(5,5))
for name_i, G in enumerate(G_types):
    in_edges = []
    for node in G.nodes:
        in_edges.append(pull_edges(G, node, 'weight', 'in'))

    all_in_edges = [x for sublist in in_edges for x in sublist]
    all_in_edges = pd.DataFrame(all_in_edges, columns = ['pre', 'post', 'weight'])
    all_in_edges = all_in_edges.set_index('pre')

    # generate histogram of edge strengths per neuron
    hists = []
    binwidth = 1
    x_range = list(range(0, int(max(all_in_edges.weight))))
    for i, skid in enumerate(np.unique(all_in_edges.index)):
        if(type(all_in_edges.loc[skid])!=pd.Series): # ignores nodes with only one edge
            total_edges = len(all_in_edges.loc[skid])
            data = all_in_edges.loc[skid].weight
            bins = np.arange(1, max_bin + binwidth + 0.5) - 0.5
            bins = np.append(bins, 500)
            hist = np.histogram(data, bins=bins)
            for hist_pair in zip(hist[0], hist[0]/total_edges, [int(x) for x in (bins+0.5)], [skid]*len(hist[0])):
                hists.append(hist_pair)

    hists = pd.DataFrame(hists, columns = ['count', 'fraction', 'bin', 'skid'])

    if(name_i==0):
        ax = axs[0,0]
    if(name_i==1):
        ax = axs[0,1]
    if(name_i==2):
        ax = axs[1,0]
    if(name_i==3):
        ax = axs[1,1]    
    sns.barplot(data=hists, x='bin', y=plot_type, ax=ax, errwidth = 1)
    ax.set(xlabel = 'Synaptic Strength', xlim=(-1, max_bin+1), ylim=(0, max_y))
    ax.set_title(f'{G_names[name_i]} Input Edges', y = 0.85)

plt.savefig(f'small_plots/plots/inputs_edge-strength-per-node_{plot_type}.pdf', bbox_inches='tight')

# %%
# comparison of pairs
brain = np.intersect1d(brain, Gad.nodes)
paired, _, _ = pm.Promat.extract_pairs_from_list(brain, pairs)

in_edges_left = []
in_edges_right = []
for i in paired.index:
    in_edges_left.append(pull_edges(Gad, paired.leftid[i], 'weight', 'in'))
    in_edges_right.append(pull_edges(Gad, paired.rightid[i], 'weight', 'in'))

all_in_edges = [x for sublist in in_edges_left for x in sublist] + [x for sublist in in_edges_right for x in sublist]
all_in_edges = pd.DataFrame(all_in_edges, columns = ['pre', 'post', 'weight'])

paired_edges = []
for index in all_in_edges.index:
    edge = all_in_edges.loc[index]
    post = edge.post
    if((post in paired.leftid) | (post in paired.rightid)): # only consider edges from paired neurons
        # is post left or right? identify partner
        # is pre paired, left or right? identify partner

