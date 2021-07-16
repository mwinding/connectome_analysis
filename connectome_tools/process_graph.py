# module for processing networkx graphs in various ways

import pandas as pd
import numpy as np
import csv
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pymaid
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx
import networkx.utils as nxu


class Analyze_Nx_G():

    def __init__(self, edges, graph_type='directed', split_pairs=False, graph=None):
        if(graph==None):
            self.edges = edges
            self.G = self.generate_graph(graph_type, split_pairs=split_pairs)
        if(graph!=None):
            self.G = graph
            self.edges = graph.edges

    def generate_graph(self, graph_type, split_pairs=False):
        edges = self.edges

        if(split_pairs==False):
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
                            
        if(split_pairs):
            if(graph_type=='directed'):
                graph = nx.DiGraph()
                for i in range(len(edges)):
                    graph.add_edge(edges.iloc[i].upstream_skid, edges.iloc[i].downstream_skid, 
                                weight = edges.iloc[i].edge_weight, 
                                edge_type = edges.iloc[i].type)    

        return(graph)

    # comprehensive list of in/out degrees and identification of hubs if desired
    def get_node_degrees(self, hub_threshold=None):
        nodes = list(self.G.nodes)
        in_degree = [self.G.in_degree(node) for node in nodes]
        out_degree = [self.G.out_degree(node) for node in nodes]

        neurons = pd.DataFrame(zip(in_degree, out_degree), index=nodes, columns=['in_degree', 'out_degree'])

        if(hub_threshold!=None):
            in_hub = [1 if in_d>=hub_threshold else 0 for in_d in in_degree]
            out_hub = [1 if out_d>=hub_threshold else 0 for out_d in out_degree]
            in_out_hub = [1 if ((degree[0]>=hub_threshold) & (degree[1]>=hub_threshold)) else 0 for degree in zip(in_degree, out_degree)]

            neurons = pd.DataFrame(zip(in_degree, out_degree, in_hub, out_hub, in_out_hub), index=nodes, columns=['in_degree', 'out_degree', 'in_hub', 'out_hub', 'in_out_hub'])
            
            hub_type=[]
            for index in range(0, len(neurons)):
                if((neurons.iloc[index, :].in_hub==1) & (neurons.iloc[index, :].out_hub==0)):
                    hub_type.append('in_hub')
                if((neurons.iloc[index, :].out_hub==1) & (neurons.iloc[index, :].in_hub==0)):
                    hub_type.append('out_hub')
                if(neurons.iloc[index, :].in_out_hub==1):
                    hub_type.append('in_out_hub')
                if((neurons.iloc[index, :].out_hub==0) & (neurons.iloc[index, :].in_hub==0)):
                    hub_type.append('non-hub')

            neurons['type']=hub_type

        return(neurons)

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

    def partner_loop_probability(self, pairs, length):
        # requires Analyze_Nx_G(..., split_pairs=True)

        if(length<2):
            print('length must be 2 or greater!')
            return

        partner_loop = []
        nonpartner_loop = []
        all_paths = []
        for i in pairs.index:
            leftid = pairs.loc[i].leftid
            rightid = pairs.loc[i].rightid
            paths = self.all_simple_self_loop_paths(source = leftid, cutoff=length)
            paths = [path for path in paths if len(path)==(length+1)]
            all_paths.append(paths)

            # when loops exist
            if(len(paths)>0):
                loop_partners = [path[1:length] for path in paths] # collect all partners that mediate loops
                if(type(loop_partners[0])==list): loop_partners = [x for sublist in loop_partners for x in sublist]
                loop_partners = list(np.unique(loop_partners))

                if(rightid in loop_partners): partner_loop.append(1)
                if(rightid not in loop_partners): partner_loop.append(0)

                for skid in pairs.rightid.values: #something looks funny here, does this double-count the real partner?
                    if(skid in loop_partners): nonpartner_loop.append(1)
                    if(skid not in loop_partners): nonpartner_loop.append(0)
                    
            # when loops don't exist
            if(len(paths)==0):
                partner_loop.append(0)
                for skid in pairs.rightid:
                    nonpartner_loop.append(0)

        prob_partner_loop = sum(partner_loop)/len(partner_loop)
        prob_nonpartner_loop = sum(nonpartner_loop)/len(nonpartner_loop)

        return(prob_partner_loop, prob_nonpartner_loop, all_paths)

    # identify loops in all sets of pairs
    def identify_loops(self, pairs, cutoff):
        paths = [self.all_simple_self_loop_paths(pair_id, cutoff) for pair_id in pairs]

        paths_length = []
        for i, paths_list in enumerate(paths):
            if(len(paths_list)==0):
                    paths_length.append([pairs[i], 0, 'none'])
            if(len(paths_list)>0):
                for subpath in paths_list:
                    edge_types = Prograph.path_edge_attributes(self.G, subpath, 'edge_type', include_skids=False)
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
    def crossing_counts_old(G, source_list, targets, cutoff, plot=False, source_name=[], target_name=[], save_path = []):

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
    def generate_save_simple_paths(G, source_list, targets, cutoff, save_path):
        targets = np.intersect1d(targets, G.nodes)
        source_list = np.intersect1d(source_list, G.nodes)  

        with gzip.open(save_path + '.csv.gz', 'wt') as f:
            writer = csv.writer(f)
            for source in source_list:
                writer.writerows(nx.all_simple_paths(G, source, targets, cutoff=cutoff))

    @staticmethod
    def open_simple_paths(save_path):
        with gzip.open(save_path, 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            recovered_stuff = [list(map(int, row)) for row in reader]

        return(recovered_stuff)

    @staticmethod
    def crossing_counts(G, paths_list, save_path=None):
        if(save_path==None):
            paths_crossing_count_list=[]
            for path in paths_list:
                paths_crossing_count = (sum(Prograph.path_edge_attributes(G, path, 'edge_type', False)=='contralateral'))
                paths_crossing_count_list.append(paths_crossing_count)

            return(paths_crossing_count_list)

        if(save_path!=None):
            with gzip.open(save_path + '.csv.gz', 'wt') as f:
                writer = csv.writer(f)
                writer.writerows([Prograph._sum_attribute(G, path) for path in paths_list])

    @staticmethod
    def _sum_attribute(G, path):
        return(sum(Prograph.path_edge_attributes(G, path, 'edge_type', False)=='contralateral'))

    @staticmethod
    def crossing_counts_no_load(G, save_path, load_path):
            with gzip.open(load_path + '.csv.gz', 'rt') as f:
                reader = csv.reader(f, delimiter=',')
                paths_list = [list(map(int, row)) for row in reader]

                with gzip.open(save_path + '.csv.gz', 'wt') as f:
                    writer = csv.writer(f)
                    writer.writerows([sum(Prograph.path_edge_attributes(G, path, 'edge_type', False)=='contralateral') for path in paths_list])

    @staticmethod
    def path_edge_attributes(G, path, attribute_name, include_skids=True, flip=False):
        if(include_skids):
            return [(u,v,G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])]
        if(include_skids==False):
            return np.array([(G[u][v][attribute_name]) for (u,v) in zip(path[0:],path[1:])])

    @staticmethod
    def pull_edges(G, skid, attribute, edge_type):
        if(edge_type=='out'):
            out_edges = [Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.out_edges(skid))]
            return(out_edges)
        if(edge_type=='in'):
            in_edges = [Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.in_edges(skid))]
            return(in_edges)
        if(edge_type=='all'):
            all_edges = [Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.out_edges(skid))] + [Prograph.path_edge_attributes(G, edge, attribute)[0] for edge in list(G.in_edges(skid))]
            return(all_edges)

    @staticmethod
    def excise_edges(edges, nodes, edge_type):
        identified_edges = [(edges.iloc[i].upstream_pair_id in nodes) & (edges.iloc[i].type==edge_type) for i in range(len(edges))]
        count_edges = sum(identified_edges)
        excised_edges = edges.loc[[not x for x in identified_edges]]

        return(excised_edges, count_edges)

    @staticmethod
    def random_excise_edges(edges, count, n_init, seed, exclude_nodes=[]):
        edges_list = []
        for i in range(n_init):
            np.random.seed(seed+i)

            selection_list = list(edges[([x not in exclude_nodes for x in edges.upstream_pair_id])].index)
            r_index_list = np.random.choice(selection_list, count, replace=False)

            edges_iter = edges.iloc[[x not in r_index_list for x in range(len(edges.index))], :].copy()
            edges_iter.reset_index(inplace=True, drop=True)
            edges_list.append(edges_iter)

        return(edges_list)

    @staticmethod
    def excise_edge_experiment(edges, nodes, edge_type, n_init, seed, exclude_nodes=[]):
        # preparing edge lists
        excised_edges, edge_count = Prograph.excise_edges(edges, nodes, edge_type)
        excised_edges_control_list = Prograph.random_excise_edges(edges, edge_count, n_init, seed, exclude_nodes)

        # loading into graphs
        excised_graph = Analyze_Nx_G(excised_edges, graph_type='directed')
        control_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_edges_control_list[i], graph_type='directed') for i in tqdm(range(len(excised_edges_control_list))))

        return(excised_graph, control_graphs)

    @staticmethod
    def random_excise_edges_type(edges, edge_type, count, n_init, seed, split_pairs=False, exclude_nodes=[]):
        edges_list = []
        for i in range(n_init):
            np.random.seed(seed+i)

            if(split_pairs):
                selection_list = list(edges[(edges.type==edge_type) & ([x not in exclude_nodes for x in edges.upstream_skid])].index)
            if(split_pairs==False):
                selection_list = list(edges[(edges.type==edge_type) & ([x not in exclude_nodes for x in edges.upstream_pair_id])].index)
            r_index_list = np.random.choice(selection_list, count, replace=False)

            edges_iter = edges.iloc[[x not in r_index_list for x in range(len(edges.index))], :].copy()
            edges_iter.reset_index(inplace=True, drop=True)
            edges_list.append(edges_iter)

        return(edges_list)

    # randomly excise left or right side ipsi edges and contra edges; generate graphs
    @staticmethod
    def excise_ipsi_contra_edge_experiment(edges, edge_count, n_init, seed, left, right, exclude_nodes=[], split_pairs=False):
        # preparing edge lists
        excised_random_ipsi_left_edges_list = Prograph.random_excise_edges_type(edges, 'ipsilateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=(exclude_nodes+right))
        excised_random_ipsi_right_edges_list = Prograph.random_excise_edges_type(edges, 'ipsilateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=(exclude_nodes+left))
        excised_random_contra_edges_list = Prograph.random_excise_edges_type(edges, 'contralateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=exclude_nodes)

        # loading into graphs
        random_ipsi_left_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_ipsi_left_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_ipsi_left_edges_list))))
        random_ipsi_right_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_ipsi_right_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_ipsi_right_edges_list))))
        random_contra_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_contra_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_contra_edges_list))))

        return(random_ipsi_left_graphs, random_ipsi_right_graphs, random_contra_graphs)

    # randomly excise ipsi vs contra edges and generate graphs
    @staticmethod
    def excise_ipsi_contra_edge_experiment_whole_brain(edges, edge_count, n_init, seed, exclude_nodes=[], split_pairs=False):
        # preparing edge lists
        excised_random_ipsi_edges_list = Prograph.random_excise_edges_type(edges, 'ipsilateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=exclude_nodes)
        excised_random_contra_edges_list = Prograph.random_excise_edges_type(edges, 'contralateral', edge_count, n_init, seed, split_pairs=True, exclude_nodes=exclude_nodes)

        # loading into graphs
        random_ipsi_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_ipsi_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_ipsi_edges_list))))
        random_contra_graphs = Parallel(n_jobs=-1)(delayed(Analyze_Nx_G)(excised_random_contra_edges_list[i], graph_type='directed', split_pairs=split_pairs) for i in tqdm(range(len(excised_random_contra_edges_list))))

        return(random_ipsi_graphs, random_contra_graphs)

    @staticmethod
    def self_loop_experiment(analyze_graph, ipsi_pair_ids, bilateral_pair_ids, contra_pair_ids, sens, cutoff):
        
        ipsi_pair_ids = np.setdiff1d(ipsi_pair_ids, sens)
        bilateral_pair_ids = np.setdiff1d(bilateral_pair_ids, sens)
        contra_pair_ids = np.setdiff1d(contra_pair_ids, sens)

        ipsi_pair_ids = list(np.intersect1d(ipsi_pair_ids, analyze_graph.G.nodes))
        bilateral_pair_ids = list(np.intersect1d(bilateral_pair_ids, analyze_graph.G.nodes))
        contra_pair_ids = list(np.intersect1d(contra_pair_ids, analyze_graph.G.nodes))

        ipsi_loops = analyze_graph.identify_loops(ipsi_pair_ids, cutoff)
        bilateral_loops = analyze_graph.identify_loops(bilateral_pair_ids, cutoff)
        contra_loops = analyze_graph.identify_loops(contra_pair_ids, cutoff)

        return(ipsi_loops, bilateral_loops, contra_loops)

