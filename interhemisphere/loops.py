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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

#rm = pymaid.CatmaidInstance(url, name, password, token)

# load previously generated paths
all_edges_combined = pd.read_csv('interhemisphere/csvs/all_paired_edges.csv', index_col=0)

# build networkx Graph
G = nx.DiGraph()

for i in range(len(all_edges_combined)):
    G.add_edge(all_edges_combined.iloc[i].upstream_pair_id, all_edges_combined.iloc[i].downstream_pair_id, 
                weight = np.mean([all_edges_combined.iloc[i].left, all_edges_combined.iloc[i].right]), 
                edge_type = all_edges_combined.iloc[i].type)

# %%
# self-loop paths functions
# modified some of the functions from networkx to check multi-hop self loops

def empty_generator():
    """ Return a generator with no members """
    yield from ()

def mod_all_simple_paths(G, source, target, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError as e:
            raise nx.NodeNotFound(f"target node {target} not in graph") from e
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return empty_generator()
    else:
        return _mod_all_simple_paths_graph(G, source, targets, cutoff)

def _mod_all_simple_paths_graph(G, source, targets, cutoff):
    visited = dict.fromkeys([str(source)]) # convert to str so it's ignored
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if (child in visited):
                continue
            if child in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()

def path_edge_attributes(G_graph, path, attribute_name, include_skids=True):
    if(include_skids):
        return [(u,v,G_graph[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])]
    if(include_skids==False):
        return np.array([(G_graph[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])])

# %%

def identify_loops(pairs, G):
    paths = []
    for i in range(len(pairs)):
        path = list(mod_all_simple_paths(G, pairs[i], pairs[i], cutoff=cutoff))
        for i in range(len(path)):
            path[i][0] = int(path[i][0]) # convert source str to int
        paths.append(path)

    paths_length = []
    for i, paths_list in enumerate(paths):
        if(len(paths_list)==0):
                paths_length.append([pairs[i], 0, 'none'])
        if(len(paths_list)>0):
            for subpath in paths_list:
                edge_types = path_edge_attributes(G, subpath, 'edge_type', include_skids=False)
                if((sum(edge_types=='contralateral')%2)==0): # if there is an even number of contralateral edges
                    paths_length.append([pairs[i], len(subpath)-1, 'self'])
                if((sum(edge_types=='contralateral')%2)==1): # if there is an odd number of contralateral edges
                    paths_length.append([pairs[i], len(subpath)-1, 'pair'])

    paths_length = pd.DataFrame(paths_length, columns = ['skid', 'path_length', 'loop_type'])
    loop_type_counts = paths_length.groupby(['skid', 'path_length', 'loop_type']).size()
    loop_type_counts = loop_type_counts>0
    total_loop_types = loop_type_counts.groupby(['path_length','loop_type']).sum()
    total_loop_types = total_loop_types/len(pairs)

    # add 0 values in case one of the conditions didn't exist
    if((1, 'pair') not in total_loop_types.index):
        total_loop_types.loc[(1, 'pair')]=0
    if((1, 'self') not in total_loop_types.index):
        total_loop_types.loc[(1, 'self')]=0
    if((2, 'pair') not in total_loop_types.index):
        total_loop_types.loc[(2, 'pair')]=0
    if((2, 'self') not in total_loop_types.index):
        total_loop_types.loc[(2, 'self')]=0
    if((3, 'pair') not in total_loop_types.index):
        total_loop_types.loc[(3, 'pair')]=0
    if((3, 'self') not in total_loop_types.index):
        total_loop_types.loc[(3, 'self')]=0

    return(total_loop_types)

cutoff = 3
pairs = list(np.unique(all_edges_combined.upstream_pair_id))

observed_loops = identify_loops(pairs, G)

# %%
# shuffled graphs
from tqdm import tqdm
from joblib import Parallel, delayed

# build randomized networkx Graph
def shuffled_graph(i, shuffle_contra=False):
    pairs = list(np.unique(all_edges_combined.upstream_pair_id))

    np.random.seed(i)
    random_nums_us = np.random.choice(len(pairs), len(all_edges_combined.index))
    np.random.seed(i+1)
    random_nums_ds = np.random.choice(len(pairs), len(all_edges_combined.index))
    np.random.seed(i+2)
    random_type = np.random.choice(len(['contralateral', 'ipsilateral']), len(all_edges_combined.index))


    all_edges_combined_randomized = all_edges_combined.copy()
    all_edges_combined_randomized.upstream_pair_id = [pairs[i] for i in random_nums_us]
    all_edges_combined_randomized.downstream_pair_id = [pairs[i] for i in random_nums_ds]
    if(shuffle_contra==True):
        all_edges_combined_randomized.type = [['contralateral', 'ipsilateral'][i] for i in random_type]

    G_shuffled = nx.DiGraph()

    for i in range(len(all_edges_combined)):
        G_shuffled.add_edge(all_edges_combined_randomized.iloc[i].upstream_pair_id, all_edges_combined_randomized.iloc[i].downstream_pair_id, 
                    weight = np.mean([all_edges_combined_randomized.iloc[i].left, all_edges_combined_randomized.iloc[i].right]), 
                    edge_type = all_edges_combined_randomized.iloc[i].type)

    return(G_shuffled)

shuffled_graphs = Parallel(n_jobs=-1)(delayed(shuffled_graph)(i, shuffle_contra=False) for i in tqdm(range(0,100*3,3)))
shuffled_graphs_loops = Parallel(n_jobs=-1)(delayed(identify_loops)(pairs, G) for G in tqdm(shuffled_graphs))
#shuffled_graphs_loops[4][(1, 'self')]=0.0 # one interation doesn't have any direct selfs
#shuffled_graphs_loops[31][(1, 'pair')]=0.0 # one interation doesn't have any direct selfs

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

# box plots
height = 1.5
width = 1.25
g = sns.catplot(data=data_self, x='path_length', y='fraction_loops', hue='condition', kind='box', height=height, aspect=width/height, linewidth=0.5, fliersize = 0.1)
g.ax.set(ylim=(0,0.5))
plt.savefig('interhemisphere/plots/loops/self_loops_vs_shuffled.pdf', format='pdf', bbox_inches='tight')

data_pair = pd.DataFrame(np.concatenate((contains_loop_1pair, contains_loop_2pair, contains_loop_3pair)), columns = ['fraction_loops', 'condition', 'path_length'])
data_pair.fraction_loops = [float(x) for x in data_pair.fraction_loops]

height = 1.5
width = 1.25
g = sns.catplot(data=data_pair, x='path_length', y='fraction_loops', hue='condition', kind='box', height=height, aspect=width/height, linewidth=0.5, fliersize = 0.1)
g.ax.set(ylim=(0,0.5))
plt.savefig('interhemisphere/plots/loops/pair_loops_vs_shuffled.pdf', format='pdf', bbox_inches='tight')

# box plots
height = 1.5
width = 1.25
with plt.rc_context({"lines.linewidth": 0.5}):
    g = sns.catplot(data=data_self, x='path_length', y='fraction_loops', hue='condition', kind='bar', height=height, aspect=width/height, linewidth=0.5)
    g.ax.set(ylim=(0,0.5))
    plt.savefig('interhemisphere/plots/loops/self_loops_vs_shuffled_bar.pdf', format='pdf', bbox_inches='tight')

height = 1.5
width = 1.25
with plt.rc_context({"lines.linewidth": 0.5}):
    g = sns.catplot(data=data_pair, x='path_length', y='fraction_loops', hue='condition', kind='bar', height=height, aspect=width/height, linewidth=0.5)
    g.ax.set(ylim=(0,0.5))
    plt.savefig('interhemisphere/plots/loops/pair_loops_vs_shuffled_bar.pdf', format='pdf', bbox_inches='tight')

# %%
