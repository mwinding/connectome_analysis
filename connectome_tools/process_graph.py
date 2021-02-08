# module for processing networkx graphs in various ways

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pymaid
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx

class Analyze_Nx_G():

    def __init__(self, edges, graph_type='directed'):
        self.edges = edges
        self.G = self.generate_graph(graph_type)


    def generate_graph(self, graph_type):
        edges = self.edges

        if(graph_type=='directed'):
            graph = nx.DiGraph()
            for i in range(len(edges)):
                graph.add_edge(edges.iloc[i].upstream_pair_id, edges.iloc[i].downstream_pair_id, 
                            weight = np.mean([edges.iloc[i].left, edges.iloc[i].right]), 
                            edge_type = edges.iloc[i].type)

        if(graph_type=='undirected'):
            graph = nx.Graph()
            for i in range(len(edges)):
                if(edges.iloc[i].upstream_pair_id == edges.iloc[i].downstream_pair_id): # remove self-edges
                    continue
                if(edges.iloc[i].upstream_pair_id != edges.iloc[i].downstream_pair_id):
                    if((edges.iloc[i].upstream_pair_id, edges.iloc[i].downstream_pair_id) not in graph.edges):
                        graph.add_edge(edges.iloc[i].upstream_pair_id, edges.iloc[i].downstream_pair_id)

        return(graph)

    # modified some of the functions from networkx to generate multi-hop self loop paths
    def empty_generator(self):
        """ Return a generator with no members """
        yield from ()

    # modified some of the functions from networkx to generate multi-hop self loop paths
    def mod_all_simple_paths(self, source, target, cutoff=None):
        if source not in self.G:
            raise nx.NodeNotFound(f"source node {source} not in graph")
        if target in self.G:
            targets = {target}
        else:
            try:
                targets = set(target)
            except TypeError as e:
                raise nx.NodeNotFound(f"target node {target} not in graph") from e
        if cutoff is None:
            cutoff = len(self.G) - 1
        if cutoff < 1:
            return self.empty_generator()
        else:
            return self._mod_all_simple_paths_graph(source, targets, cutoff)

    # modified some of the functions from networkx to generate multi-hop self loop paths
    def _mod_all_simple_paths_graph(self, source, targets, cutoff):
        visited = dict.fromkeys([str(source)]) # convert to str so it's ignored
        stack = [iter(self.G[source])]
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
                    stack.append(iter(self.G[child]))
                else:
                    visited.popitem()  # maybe other ways to child
            else:  # len(visited) == cutoff:
                for target in (targets & (set(children) | {child})) - set(visited.keys()):
                    yield list(visited) + [target]
                stack.pop()
                visited.popitem()

    def all_simple_self_loop_paths(self, source, cutoff):
        path = list(self.mod_all_simple_paths(source=source, target=source, cutoff=cutoff))
        for i in range(len(path)):
            path[i][0] = int(path[i][0]) # convert source str to int
        return(path)

    def path_edge_attributes(self, path, attribute_name, include_skids=True):
        if(include_skids):
            return [(u,v,self.G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])]
        if(include_skids==False):
            return np.array([(self.G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])])

    # identify loops in all sets of pairs
    def identify_loops(self, pairs, cutoff):
        paths = [self.all_simple_self_loop_paths(pair_id, cutoff) for pair_id in pairs]

        paths_length = []
        for i, paths_list in enumerate(paths):
            if(len(paths_list)==0):
                    paths_length.append([pairs[i], 0, 'none'])
            if(len(paths_list)>0):
                for subpath in paths_list:
                    edge_types = self.path_edge_attributes(subpath, 'edge_type', include_skids=False)
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

    def shuffled_graph(self, i, shuffle_contra=False):
        pairs = list(np.unique(self.edges.upstream_pair_id))

        np.random.seed(i)
        random_nums_us = np.random.choice(len(pairs), len(self.edges.index))
        np.random.seed(i+1)
        random_nums_ds = np.random.choice(len(pairs), len(self.edges.index))
        np.random.seed(i+2)
        random_type = np.random.choice(len(['contralateral', 'ipsilateral']), len(self.edges.index))


        all_edges_combined_randomized = self.edges.copy()
        all_edges_combined_randomized.upstream_pair_id = [pairs[i] for i in random_nums_us]
        all_edges_combined_randomized.downstream_pair_id = [pairs[i] for i in random_nums_ds]
        if(shuffle_contra==True):
            all_edges_combined_randomized.type = [['contralateral', 'ipsilateral'][i] for i in random_type]

        G_shuffled = nx.DiGraph()

        for i in range(len(self.edges)):
            G_shuffled.add_edge(all_edges_combined_randomized.iloc[i].upstream_pair_id, all_edges_combined_randomized.iloc[i].downstream_pair_id, 
                        weight = np.mean([all_edges_combined_randomized.iloc[i].left, all_edges_combined_randomized.iloc[i].right]), 
                        edge_type = all_edges_combined_randomized.iloc[i].type)

        return(G_shuffled)

    def generate_shuffled_graphs(self, num, shuffle_contra=False):
        shuffled_graphs = Parallel(n_jobs=-1)(delayed(self.shuffled_graph)(i, shuffle_contra=False) for i in tqdm(range(0,num*3,3)))
        return(shuffled_graphs)
