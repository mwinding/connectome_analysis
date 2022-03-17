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

Gad = nx.readwrite.graphml.read_graphml('data/edges/Gad.graphml', node_type=int)
Gaa = nx.readwrite.graphml.read_graphml('data/edges/Gaa.graphml', node_type=int)
Gdd = nx.readwrite.graphml.read_graphml('data/edges/Gdd.graphml', node_type=int)
Gda = nx.readwrite.graphml.read_graphml('data/edges/Gda.graphml', node_type=int)

G_types = [Gad, Gaa, Gdd, Gda]
G_names = ['Axon-Dendrite', 'Axon-Axon', 'Dendrite-Dendrite', 'Dendrite-Axon']

def pull_edges_source_centric(G, skid, attribute):
        in_edges = [pg.Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.in_edges(skid))] 
        out_edges = [pg.Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.out_edges(skid))]
        
        #if(len(out_edges)==1):
        #    out_edges = [out_edges]

        #if(len(in_edges)==1):
        #    in_edges = [in_edges]

        in_edges = [[x[1], x[0], x[2]] for x in in_edges]
            
        return(in_edges + out_edges)
# %%
# edge fractions vs edge strength

def edge_fraction_by_synapse(max_bin, plot_type, edge_type, G_types, G_names):
    # edge_type = ['in', 'out', 'all']
    # plot_type = ['fraction', 'count']
    if(plot_type == 'count'):
        max_y = 10
    if(plot_type == 'fraction'):
        max_y = 1.0

    # generate histograms of synaptic strength
    all_hists = []
    for name_i, G in enumerate(G_types):
        edges = []
        for node in G.nodes:
            if(edge_type!='all'):
                edges.append(pg.Prograph.pull_edges(G, node, 'weight', edge_type))
            if(edge_type=='all'):
                edges.append(pull_edges_source_centric(G, node, 'weight'))
            
        all_edges = [x for sublist in edges for x in sublist]

        if(edge_type!='all'):
            all_edges = pd.DataFrame(all_edges, columns = ['pre', 'post', 'weight'])
            if(edge_type=='in'):
                all_edges = all_edges.set_index('post')
            if(edge_type=='out'):
                all_edges = all_edges.set_index('pre')

        if(edge_type=='all'):
            all_edges = pd.DataFrame(all_edges, columns = ['source', 'partner', 'weight'])
            all_edges = all_edges.set_index('source')

        # generate histogram of edge strengths per neuron
        hists = []
        binwidth = 1
        x_range = list(range(0, int(max(all_edges.weight))))
        for i, skid in enumerate(np.unique(all_edges.index)):
            if(type(all_edges.loc[skid])!=pd.Series): # ignores nodes with only one edge
                total_edges = len(all_edges.loc[skid])
                data = all_edges.loc[skid].weight
                bins = np.arange(1, max_bin + binwidth + 0.5) - 0.5
                bins = np.append(bins, 500)
                hist = np.histogram(data, bins=bins)
                for hist_pair in zip(hist[0], hist[0]/total_edges, [int(x) for x in (bins+0.5)], [skid]*len(hist[0]), [G_names[name_i]]*len(hist[0])):
                    hists.append(hist_pair)

        hists = pd.DataFrame(hists, columns = ['count', 'fraction', 'bin', 'skid', 'edge_type'])
        all_hists.append(hists)

    # plot histograms
    fig, axs = plt.subplots(2,2, figsize=(5,5))
    for i, hists in enumerate(all_hists):

        if(i==0):
            ax = axs[0,0]
            color = sns.color_palette()[0]
        if(i==1):
            ax = axs[0,1]
            color = sns.color_palette()[1]
        if(i==2):
            ax = axs[1,0]
            color = sns.color_palette()[2]
        if(i==3):
            ax = axs[1,1]    
            color = sns.color_palette()[3]
        sns.barplot(data=hists, x='bin', y=plot_type, ax=ax, errwidth = 1, color=color)
        ax.set(xlabel = 'Synaptic Strength', xlim=(-1, max_bin+1), ylim=(0, max_y))
        ax.set_title(f'{G_names[i]} {edge_type} Edges', y = 0.85)

    plt.savefig(f'small_plots/plots/{edge_type}_edge-strength-per-node_{plot_type}.pdf', bbox_inches='tight')

    all_hists = pd.concat(all_hists, axis=0)

    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5)) #(3,1.5) before
    sns.barplot(data=all_hists, x='bin', y=plot_type, hue='edge_type', ax = ax, errwidth = 1)
    ax.set(ylabel = 'Fraction of Edges', xlabel = 'Edge Strength', ylim=(0,1))
    plt.savefig(f'small_plots/plots/{edge_type}_edge-strength-per-node_{plot_type}_combined.pdf', bbox_inches='tight')

max_bin = 4
edge_fraction_by_synapse(max_bin, 'fraction', 'in', G_types, G_names)
edge_fraction_by_synapse(max_bin, 'fraction', 'out', G_types, G_names)
edge_fraction_by_synapse(max_bin, 'fraction', 'all', G_types, G_names)

# %%
# fraction synapses vs edge strength

def syn_fraction_by_synapse(max_bin, plot_type, edge_type, G_types, G_names):
    # edge_type = ['in', 'out', 'all']
    # plot_type = ['fraction', 'count']
    # generate histograms
    all_hists = []
    for name_i, G in enumerate(G_types):
        edges = []
        for node in G.nodes:
            if(edge_type!='all'):
                edges.append(pg.Prograph.pull_edges(G, node, 'weight', edge_type))
            if(edge_type=='all'):
                edges.append(pull_edges_source_centric(G, node, 'weight'))
                
        all_edges = [x for sublist in edges for x in sublist]

        if(edge_type!='all'):
            all_edges = pd.DataFrame(all_edges, columns = ['pre', 'post', 'weight'])
            if(edge_type=='in'):
                all_edges = all_edges.set_index('post')
            if(edge_type=='out'):
                all_edges = all_edges.set_index('pre')

        if(edge_type=='all'):
            all_edges = pd.DataFrame(all_edges, columns = ['source', 'partner', 'weight'])
            all_edges = all_edges.set_index('source')

        # generate histogram of edge strengths per neuron
        hists = []
        x_range = list(range(0, int(max(all_edges.weight))))
        for i, skid in enumerate(np.unique(all_edges.index)):
            if(type(all_edges.loc[skid])!=pd.Series): # ignores nodes with only one edge
                total_edges = len(all_edges.loc[skid])
                data = all_edges.loc[skid]
                data['syn_sum'] = data.weight
                data = data.groupby('weight').sum()
                data['syn_fraction'] = data.syn_sum/sum(data.syn_sum)
                missing_syn_counts = [x for x in range(1, max_bin+1) if x not in data.index]
                missing_syn_counts = pd.DataFrame([[0.0, 0.0]]*len(missing_syn_counts), columns = ['syn_sum', 'syn_fraction'], index = missing_syn_counts)
                data = data.append(missing_syn_counts)
                
                summed = data[data.index>max_bin].sum()
                data = data[data.index<=max_bin].append(pd.DataFrame([summed], columns = ['syn_sum', 'syn_fraction'], index=[11]))

                for hist_pair in zip(data.syn_sum, data.syn_fraction, data.index, [skid]*len(data.syn_sum), [G_names[name_i]]*len(data.syn_sum)):
                    hists.append(hist_pair)

        hists = pd.DataFrame(hists, columns = ['count', 'fraction', 'bin', 'skid', 'edge_type'])
        all_hists.append(hists)

    fig, axs = plt.subplots(2,2, figsize=(5,5))
    for i, hists in enumerate(all_hists):

        hists.index = [int(x) for x in hists.index] # convert to int for plotting

        if(i==0):
            ax = axs[0,0]
            color = sns.color_palette()[0]
        if(i==1):
            ax = axs[0,1]
            color = sns.color_palette()[1]
        if(i==2):
            ax = axs[1,0]
            color = sns.color_palette()[2]
        if(i==3):
            ax = axs[1,1]    
            color = sns.color_palette()[3]
        sns.barplot(data=hists, x='bin', y=plot_type, ax=ax, errwidth = 1, color = color)
        ax.set(xlabel = 'Edge Strength', ylim=(0, 1), ylabel='Fraction of Edges')
        ax.set_title(f'{G_names[i]}', y = 0.85)

    plt.savefig(f'small_plots/plots/{edge_type}-counts_edge-strength-per-node_{plot_type}.pdf', bbox_inches='tight')

    all_hists_combined = pd.concat(all_hists, axis=0)
    all_hists_combined.index = [int(x) for x in all_hists_combined.index]

    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
    sns.barplot(data=all_hists_combined, x='bin', y=plot_type, hue='edge_type', ax = ax, errwidth = 1)
    ax.set(ylabel = 'Fraction of Total Synapses', xlabel = 'Edge Strength', ylim=(0,1))
    plt.savefig(f'small_plots/plots/{edge_type}-counts_edge-strength-per-node_{plot_type}_combined.pdf', bbox_inches='tight')

max_bin = 4

syn_fraction_by_synapse(max_bin, 'fraction', 'in', G_types, G_names)
syn_fraction_by_synapse(max_bin, 'fraction', 'out', G_types, G_names)
syn_fraction_by_synapse(max_bin, 'fraction', 'all', G_types, G_names)
# %%
# comparison of pairs
brain = np.intersect1d(brain, Gad.nodes)
paired, _, _ = pm.Promat.extract_pairs_from_list(brain, pairs)

in_edges_left = []
in_edges_right = []
for i in paired.index:
    in_edges_left.append(pg.Prograph.pull_edges(Gad, paired.leftid[i], 'weight', 'in'))
    in_edges_right.append(pg.Prograph.pull_edges(Gad, paired.rightid[i], 'weight', 'in'))

all_in_edges = [x for sublist in in_edges_left for x in sublist] + [x for sublist in in_edges_right for x in sublist]
all_in_edges = pd.DataFrame(all_in_edges, columns = ['pre', 'post', 'weight'])

paired_edges = []
for index in all_in_edges.index:
    edge = all_in_edges.loc[index]
    post = edge.post
    pre = edge.pre
    if((post in paired.leftid) | (post in paired.rightid)): # only consider edges from paired neurons
        post_partner = pm.Promat.identify_pair(post, pairs)

        if((pre in paired.leftid) | (pre in paired.rightid)): # if pre is from paired neurons
            pre_partner = pm.Promat.identify_pair(pre, pairs)

            if(pre in paired.leftid):
                paired_edge_bool = (all_in_edges.pre==pre_partner) & (all_in_edges.post==post_partner)
                if(paired_edge_bool):
                    edge = [pre, post, edge.weight, pre_partner, post_partner, all_in_edges[paired_edge_bool].weight]
                else:
                    edge = [pre, post, edge.weight, pre_partner, post_partner, 0]
    
    paired_edges.append(edge)


