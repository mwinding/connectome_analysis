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
import networkx.utils as nxu

class Analyze_Nx_G():

    def __init__(self, edges, graph_type='directed', graph=None):
        if(graph==None):
            self.edges = edges
            self.G = self.generate_graph(graph_type)
        if(graph!=None):
            self.G = graph
            self.edges = graph.edges

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

    # identify loops in all sets of pairs
    def identify_loops(self, pairs, cutoff):
        paths = [self.all_simple_self_loop_paths(pair_id, cutoff) for pair_id in pairs]

        paths_length = []
        for i, paths_list in enumerate(paths):
            if(len(paths_list)==0):
                    paths_length.append([pairs[i], 0, 'none'])
            if(len(paths_list)>0):
                for subpath in paths_list:
                    edge_types = self.path_edge_attributes(self.G, subpath, 'edge_type', include_skids=False)
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

    # only works on undirected graph
    def shuffled_graph(self, seed, Q=100):
        R = self.G
        E = R.number_of_edges()
        nx.double_edge_swap(R,Q*E,max_tries=Q*E*10, seed=seed)
        return(R)

    # only works on undirected graph
    def generate_shuffled_graphs(self, num, graph_type, Q=100):
        
        if(graph_type=='undirected'):
            shuffled_graphs = Parallel(n_jobs=-1)(delayed(self.shuffled_graph)(seed=i, Q=Q) for i in tqdm(range(0,num)))
            return(shuffled_graphs)
        if(graph_type=='directed'):
            shuffled_graphs = Parallel(n_jobs=-1)(delayed(self.directed_shuffled_graph)(seed=i, Q=Q) for i in tqdm(range(0,num)))
            return(shuffled_graphs)

    def directed_shuffled_graph(self, seed, Q=100):
        R = self.G
        E = R.number_of_edges()
        self.directed_double_edge_swap(R, Q*E, max_tries=Q*E*10)
        return(R)
        
    # works on directed graph, preserves input and output degree
    # modified from networkx double_edge_swap()
    def directed_double_edge_swap(self, G, nswap=1, max_tries=100, seed=None):
        # u--v          u--y       instead of:      u--v            u   v
        #       becomes                                    becomes  |   |
        # x--y          x--v                        x--y            x   y

        np.random.seed(0)
        
        if nswap > max_tries:
            raise nx.NetworkXError("Number of swaps > number of tries allowed.")
        if len(G) < 4:
            raise nx.NetworkXError("Graph has less than four nodes.")
        # Instead of choosing uniformly at random from a generated edge list,
        # this algorithm chooses nonuniformly from the set of nodes with
        # probability weighted by degree.
        n = 0
        swapcount = 0
        keys, degrees = zip(*G.out_degree())  # keys, degree
        cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
        discrete_sequence = nx.utils.discrete_sequence
        while (swapcount < nswap):
            #        if random.random() < 0.5: continue # trick to avoid periodicities?
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ui, xi) = discrete_sequence(2, cdistribution=cdf)
            if (ui == xi):
                continue  # same source, skip
            u = keys[ui]  # convert index to label
            x = keys[xi]

            # ignore nodes with no downstream partners
            if((len(G[u])==0) | (len(G[x])==0)):
                continue

            # choose target uniformly from neighbors
            v = np.random.choice(list(G[u]))
            y = np.random.choice(list(G[x]))
            if (v == y):
                continue  # same target, skip
            if (y not in G[u]) and (v not in G[x]):  # don't create parallel edges
                G.add_edge(u, y, weight = G[u][v]['weight'], edge_type = G[u][v]['edge_type'])
                G.add_edge(x, v, weight = G[x][y]['weight'], edge_type = G[x][y]['edge_type'])
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                swapcount += 1
            if (n >= max_tries):
                e = (
                    f"Maximum number of swap attempts ({n}) exceeded "
                    f"before desired swaps achieved ({nswap})."
                )
                raise nx.NetworkXAlgorithmError(e)
            n += 1
        return G


class Prograph():
    @staticmethod
    def crossing_counts(G, source_list, targets, cutoff, plot=False, source_name=[], target_name=[], save_path = []):

        targets = np.intersect1d(targets, G.nodes)
        source_list = np.intersect1d(source_list, G.nodes)

        all_paths = [nx.all_simple_paths(G, source, targets, cutoff=cutoff) for source in source_list]
        paths_list = [x for sublist in all_paths for x in sublist]
        paths_crossing_count = [sum(Prograph.path_edge_attributes(G, path, 'edge_type', False)=='contralateral') for path in paths_list]
        if(plot):
            # allows text to be editable in Illustrator
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42

            # font settings
            plt.rcParams['font.size'] = 5
            plt.rcParams['font.family'] = 'arial'

            binwidth = 1
            x_range = list(range(0, 7))
            data = paths_crossing_count
            bins = np.arange(min(data), max(data) + binwidth + 0.5) - 0.5

            fig,ax = plt.subplots(1,1, figsize=(1.5, 2))
            sns.distplot(paths_crossing_count, bins=bins, kde=False, ax=ax, hist_kws={"rwidth":0.9,'alpha':0.75})
            ax.set(xlim = (-0.75, 6.75), ylabel=f'Number of Paths ({source_name} to {target_name})', xlabel='Number of Interhemisphere Crossings', xticks=[i for i in range(7)])
            plt.savefig(f'{save_path}/{source_name}-to-{target_name}_distribution.pdf', format='pdf', bbox_inches='tight')

        return(paths_list, paths_crossing_count)
    
    @staticmethod
    def path_edge_attributes(G, path, attribute_name, include_skids=True):
        if(include_skids):
            return [(u,v,G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])]
        if(include_skids==False):
            return np.array([(G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])])

