# module for processing adjacency matrices in various ways

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pymaid
from tqdm import tqdm
from joblib import Parallel, delayed

class Adjacency_matrix():

    def __init__(self, adj, input_counts, mat_type):
        self.skids = list(adj.index)
        self.pairs = Promat.get_pairs()
        self.input_counts = input_counts
        self.mat_type = mat_type # 'ad', 'aa', 'dd', 'da', 'summed'
        self.adj = pd.DataFrame(adj, index = self.skids, columns = self.skids)
        self.adj_fract = self.fraction_input_matrix()
        self.adj_inter = self.interlaced_matrix()
        self.adj_pairwise = self.average_pairwise_matrix()

    def fraction_input_matrix(self):
        adj_fract = self.adj.copy()
        for column in adj_fract.columns:
            if((self.mat_type=='aa') | (self.mat_type=='da')):
                axon_input = self.input_counts.loc[column].axon_input
                if(axon_input == 0):
                    adj_fract.loc[:, column] = 0
                if(axon_input > 0):
                    adj_fract.loc[:, column] = adj_fract.loc[:, column]/axon_input

            if((self.mat_type=='ad') | (self.mat_type=='dd')):
                dendrite_input = self.input_counts.loc[column].dendrite_input
                if(dendrite_input == 0):
                    adj_fract.loc[:, column] = 0
                if(dendrite_input > 0):
                    adj_fract.loc[:, column] = adj_fract.loc[:, column]/dendrite_input

            if((self.mat_type=='summed') | (self.mat_type not in ['aa', 'da', 'ad', 'dd']) ):
                all_input = self.input_counts.loc[column].dendrite_input + self.input_counts.loc[column].axon_input
                if(all_input == 0):
                    adj_fract.loc[:, column] = 0
                if(all_input > 0):
                    adj_fract.loc[:, column] = adj_fract.loc[:, column]/all_input

        return(adj_fract)

    def interlaced_matrix(self, fract=True):
        if(fract):
            adj_mat = self.adj_fract.copy()
            brain_pairs, brain_unpaired, brain_nonpaired = Promat.extract_pairs_from_list(adj_mat, self.pairs)
        if(fract==False):
            adj_mat = self.adj.copy()
            brain_pairs, brain_unpaired, brain_nonpaired = Promat.extract_pairs_from_list(adj_mat, self.pairs)
            
        # left_right interlaced order for brain matrix
        brain_pair_order = []
        for i in range(0, len(brain_pairs)):
            brain_pair_order.append(brain_pairs.iloc[i].leftid)
            brain_pair_order.append(brain_pairs.iloc[i].rightid)

        order = brain_pair_order + list(brain_nonpaired.nonpaired)
        interlaced_mat = adj_mat.loc[order, order]

        index_df = pd.DataFrame([['pairs', Promat.get_paired_skids(skid, self.pairs)[0], skid] for skid in brain_pair_order] + [['nonpaired', skid, skid] for skid in list(brain_nonpaired.nonpaired)], 
                                columns = ['pair_status', 'pair_id', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)

        interlaced_mat.index = index
        interlaced_mat.columns = index
        return(interlaced_mat)
    
    def average_pairwise_matrix(self):
        adj = self.adj_inter.copy()

        adj = adj.groupby('pair_id', axis = 'index').sum().groupby('pair_id', axis='columns').sum()

        order = [x[1] for x in self.adj_inter.index]
        
        # remove duplicates (in pair_ids)
        order_unique = []
        for x in order:
            if (order_unique.count(x) == 0):
                order_unique.append(x)

        # order as before
        adj = adj.loc[order_unique, order_unique]

        # regenerate multiindex
        index = [x[0:2] for x in self.adj_inter.index] # remove skid ids from index

        # remove duplicates (in pair_ids)
        index_unique = []
        for x in index:
            if (index_unique.count(x) == 0):
                index_unique.append(x)

        # add back appropriate multiindex
        index_df = pd.DataFrame(index_unique, columns = ['pair_status', 'pair_id'])
        index_df = pd.MultiIndex.from_frame(index_df)
        adj.index = index_df
        adj.columns = index_df

        # convert to average (from sum) for paired neurons
        adj.loc['pairs'] = adj.loc['pairs'].values/2
        adj.loc['nonpaired', 'pairs'] = adj.loc['nonpaired', 'pairs'].values/2 

        return(adj)

    def downstream(self, source, threshold, exclude=[], by_group=False, exclude_unpaired = False):
        adj = self.adj_pairwise

        source_pair_id = np.unique([x[1] for x in self.adj_inter.loc[(slice(None), slice(None), source), :].index])

        if(by_group):
            bin_mat = adj.loc[(slice(None), source_pair_id), :].sum(axis=0) > threshold
            bin_column = np.where(bin_mat)[0]
            ds_neurons = bin_mat.index[bin_column]

            ds_neurons_skids = []
            for pair in ds_neurons:
                if((pair[0] == 'pairs') & (pair[1] not in exclude)):
                    ds_neurons_skids.append(pair[1])
                    ds_neurons_skids.append(Promat.identify_pair(pair[1], self.pairs))
                if((pair[0] == 'nonpaired') & (pair[1] not in exclude) & (exclude_unpaired==False)):
                    ds_neurons_skids.append(pair[1])

            return(ds_neurons_skids)

        if(by_group==False):
            bin_mat = adj.loc[(slice(None), source_pair_id), :] > threshold
            bin_column = np.where(bin_mat.sum(axis = 0) > 0)[0]
            ds_neurons = bin_mat.columns[bin_column]
            bin_row = np.where(bin_mat.sum(axis = 1) > 0)[0]
            us_neurons = bin_mat.index[bin_row]

            ds_neurons_skids = []
            for pair in ds_neurons:
                if((pair[0] == 'pairs') & (pair[1] not in exclude)):
                    ds_neurons_skids.append(pair[1])
                    ds_neurons_skids.append(Promat.identify_pair(pair[1], self.pairs))
                if((pair[0] == 'nonpaired') & (pair[1] not in exclude) & (exclude_unpaired==False)):
                    ds_neurons_skids.append(pair[1])

            source_skids = []
            for pair in us_neurons:
                if(pair[0] == 'pairs'):
                    source_skids.append(pair[1])
                    source_skids.append(Promat.identify_pair(pair[1], self.pairs))
                if(pair[0] == 'nonpaired'):
                    source_skids.append(pair[1])

            edges = []
            for pair in us_neurons:
                if(pair[0] == 'pairs'):
                    specific_ds = adj.loc[('pairs',  pair[1]), bin_mat.loc[('pairs',  pair[1]), :]].index
                    if(exclude_unpaired):
                        specific_ds_edges = [[pair[1], x[1]] for x in specific_ds if (x[0]=='pairs') & (x[1] not in exclude)]
                    if(exclude_unpaired==False):
                        specific_ds_edges = [[pair[1], x[1]] for x in specific_ds if (x[1] not in exclude)]
                    for edge in specific_ds_edges:
                        edges.append(edge)

                if(pair[0] == 'nonpaired'):
                    specific_ds = adj.loc[('nonpaired', pair[1]), bin_mat.loc[('nonpaired', pair[1]), :]].index
                    if(exclude_unpaired):
                        specific_ds_edges = [[pair[1], x[1]] for x in specific_ds if (x[0]=='pairs') & (x[1] not in exclude)]
                    if(exclude_unpaired==False):
                        specific_ds_edges = [[pair[1], x[1]] for x in specific_ds if (x[1] not in exclude)]
                    for edge in specific_ds_edges:
                        edges.append(edge)
                        
            return(source_skids, ds_neurons_skids, edges)

    def upstream(self, source, threshold, exclude = []):
        adj = self.adj_pairwise

        source_pair_id = np.unique([x[1] for x in self.adj_inter.loc[(slice(None), slice(None), source), :].index])

        bin_mat = adj.loc[:, (slice(None), source_pair_id)] > threshold
        bin_row = np.where(bin_mat.sum(axis = 1) > 0)[0]
        us_neuron_pair_ids = bin_mat.index[bin_row]

        us_neurons_skids = []
        for pair in us_neuron_pair_ids:
            if((pair[0] == 'pairs') & (pair[1] not in exclude)):
                us_neurons_skids.append(pair[1])
                us_neurons_skids.append(Promat.identify_pair(pair[1], self.pairs))
            if((pair[0] == 'nonpaired') & (pair[1] not in exclude)):
                us_neurons_skids.append(pair[1])

        us_neuron_pair_ids = Promat.extract_pairs_from_list(us_neurons_skids, self.pairs)
        us_neuron_pair_ids = list(us_neuron_pair_ids[0].leftid) + list(us_neuron_pair_ids[2].nonpaired)

        edges = []
        for pair in us_neuron_pair_ids:
            specific_ds = bin_mat.loc[(slice(None), pair), bin_mat.loc[(slice(None),  pair), :].values[0]].columns
            specific_ds_edges = [[pair, x[1]] for x in specific_ds]
            for edge in specific_ds_edges:
                edges.append(edge)

        return(us_neurons_skids, edges)

    def downstream_multihop(self, source, threshold, min_members=0, hops=10, exclude=[], strict=False, allow_source_ds=False):
        if(allow_source_ds==False):
            _, ds, edges = self.downstream(source, threshold, exclude=(source + exclude))
        if(allow_source_ds):
            _, ds, edges = self.downstream(source, threshold, exclude=(exclude))

        left = Promat.get_hemis('left')
        right = Promat.get_hemis('right')

        _, ds = self.edge_threshold(edges, threshold, direction='downstream', strict=strict, left=left, right=right)

        if(allow_source_ds==False):
            before = source + ds
        if(allow_source_ds):
            before = ds

        layers = []
        layers.append(ds)

        for i in range(0,(hops-1)):
            source = ds
            _, ds, edges = self.downstream(source, threshold, exclude=before) 
            _, ds = self.edge_threshold(edges, threshold, direction='downstream', strict=strict, left=left, right=right)

            if((len(ds)!=0) & (len(ds)>=min_members)):
                layers.append(ds)
                before = before + ds

        return(layers)

    def upstream_multihop(self, source, threshold, min_members=10, hops=10, exclude=[], strict=False, allow_source_us=False):
        if(allow_source_us==False):        
            us, edges = self.upstream(source, threshold, exclude=(source + exclude))
        if(allow_source_us):
            us, edges = self.upstream(source, threshold, exclude=(exclude))

        _, us = self.edge_threshold(edges, threshold, direction='upstream', strict=strict)

        if(allow_source_us==False):
            before = source + us
        if(allow_source_us):
            before = us

        layers = []
        layers.append(us)

        for i in range(0,(hops-1)):
            source = us
            us, edges = self.upstream(source, threshold, exclude = before)
            _, us = self.edge_threshold(edges, threshold, direction='upstream', strict=strict)

            if((len(us)!=0) & (len(us)>=min_members)):
                layers.append(us)
                before = before + us

        return(layers)

    # checking additional threshold criteria after identifying neurons over summed threshold
    # left and right are only necessary when nonpaired neurons are included
    def edge_threshold(self, edges, threshold, direction, strict=False, include_nonpaired=True, left=[], right=[]):

        adj = self.adj_inter.copy()

        all_edges = []
        for edge in edges:
            print(edge)
            specific_edges = adj.loc[(slice(None), edge[0]), (slice(None), edge[1])]

            us_pair_status = adj.loc[(slice(None), slice(None), edge[0]), :].index[0][0]
            ds_pair_status = adj.loc[(slice(None), slice(None), edge[1]), :].index[0][0]

            # note that edge weights in 'left', 'right' columns refer to %input onto the dendrite of the left or right hemisphere downstream neuron
            #   the 'type' column indicates whether the edge is contralateral/ipsilateral (this can allow one to determine whether the signal originated on the left or right side if that's important)
            #   note: the split_paired_edges() method takes the output of threshold_edge_list() and splits these paired edges so that it becomes more explicit which hemisphere the upstream neuron belongs to

            # check for paired connections
            if((us_pair_status == 'pairs') & (ds_pair_status == 'pairs')):
                specific_edges = pd.DataFrame([[edge[0], edge[1], specific_edges.iloc[0,0], specific_edges.iloc[1,1], False, 'ipsilateral', 'paired','paired'],
                                                [edge[0], edge[1], specific_edges.iloc[1,0], specific_edges.iloc[0,1], False, 'contralateral', 'paired','paired']], 
                                                columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                if(strict==True):
                    # is each edge weight over threshold?
                    for index in specific_edges.index:
                        if((specific_edges.loc[index].left>threshold) & (specific_edges.loc[index].right>threshold)):
                            specific_edges.loc[index, 'overthres'] = True

                if(strict==False):
                    # is average edge weight over threshold
                    for index in specific_edges.index:
                        if(((specific_edges.loc[index].left + specific_edges.loc[index].right)/2) > threshold):
                            specific_edges.loc[index, 'overthres'] = True

                # are both edges present?
                for index in specific_edges.index:
                    if((specific_edges.loc[index].left==0) | (specific_edges.loc[index].right==0)):
                        specific_edges.loc[index, 'overthres'] = False
                            
                all_edges.append(specific_edges.values[0])
                all_edges.append(specific_edges.values[1])

            # check for edges to downstream nonpaired neurons
            if((us_pair_status == 'pairs') & (ds_pair_status == 'nonpaired') & (include_nonpaired==True)):

                if(edge[1] in left):
                    specific_edges = pd.DataFrame([[edge[0], edge[1], specific_edges.iloc[0].values[0], 0, False, 'ipsilateral', 'paired', 'nonpaired'],
                                                    [edge[0], edge[1], specific_edges.iloc[1].values[0], 0, False, 'contralateral', 'paired', 'nonpaired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                if(edge[1] in right):
                    specific_edges = pd.DataFrame([[edge[0], edge[1], 0, specific_edges.iloc[0].values[0], False, 'contralateral', 'paired', 'nonpaired'],
                                                    [edge[0], edge[1], 0, specific_edges.iloc[1].values[0], False, 'ipsilateral', 'paired', 'nonpaired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                # is edge over threshold? 
                # don't check both because one will be missing
                # strict==True/False doesn't apply for the same reason
                for index in specific_edges.index:
                    if(((specific_edges.loc[index].left + specific_edges.loc[index].right)>threshold)):
                        specific_edges.loc[index, 'overthres'] = True
                                
                all_edges.append(specific_edges.values[0])
                all_edges.append(specific_edges.values[1])


            # check for edges from upstream nonpaired neurons
            if((us_pair_status == 'nonpaired') & (ds_pair_status == 'pairs') & (include_nonpaired==True)):

                if(edge[0] in left):
                    specific_edges = pd.DataFrame([[edge[0], edge[1], specific_edges.iloc[0, 0], 0, False, 'ipsilateral', 'nonpaired', 'paired'],
                                                    [edge[0], edge[1], 0, specific_edges.iloc[0, 1], False, 'contralateral', 'nonpaired', 'paired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                if(edge[0] in right):
                    specific_edges = pd.DataFrame([[edge[0], edge[1], specific_edges.iloc[0, 0], 0, False, 'contralateral', 'nonpaired', 'paired'],
                                                    [edge[0], edge[1], 0, specific_edges.iloc[0, 1], False, 'ipsilateral', 'nonpaired', 'paired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                # is edge over threshold? 
                # don't check both because one will be missing
                # strict==True/False doesn't apply for the same reason
                for index in specific_edges.index:
                    if(((specific_edges.loc[index].left + specific_edges.loc[index].right)>threshold)):
                        specific_edges.loc[index, 'overthres'] = True
                                
                all_edges.append(specific_edges.values[0])
                all_edges.append(specific_edges.values[1])

            # check for edges between two nonpaired neurons
            if((us_pair_status == 'nonpaired') & (ds_pair_status == 'nonpaired') & (include_nonpaired==True)):

                edge_weight = specific_edges.values[0][0]
                if(edge[0] in left):
                    if(edge[1] in right):
                        specific_edges = pd.DataFrame([[edge[0], edge[1], 0, edge_weight, False, 'contralateral', 'nonpaired', 'nonpaired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])
                    if(edge[1] in left):
                        specific_edges = pd.DataFrame([[edge[0], edge[1], edge_weight, 0, False, 'ipsilateral', 'nonpaired', 'nonpaired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                if(edge[0] in right):
                    if(edge[1] in left):
                        specific_edges = pd.DataFrame([[edge[0], edge[1], edge_weight, 0, False, 'contralateral', 'nonpaired', 'nonpaired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])
                    if(edge[1] in right):
                        specific_edges = pd.DataFrame([[edge[0], edge[1], 0, edge_weight, False, 'ipsilateral', 'nonpaired', 'nonpaired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                # is edge over threshold? 
                # only one edge so strict==True/False doesn't apply
                if(edge_weight>threshold):
                    specific_edges.loc[:, 'overthres'] = True
                                
                all_edges.append(specific_edges.values[0])
            
        all_edges = pd.DataFrame(all_edges, columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])
        
        if(direction=='downstream'):
            partner_skids = np.unique(all_edges[all_edges.overthres==True].downstream_pair_id) # identify downstream pairs
        if(direction=='upstream'):
            partner_skids = np.unique(all_edges[all_edges.overthres==True].upstream_pair_id) # identify upstream pairs
            
        partner_skids = [x[2] for x in adj.loc[(slice(None), partner_skids), :].index] # convert from pair_id to skids
        
        return(all_edges, partner_skids)

    # select edges from results of edge_threshold that are over threshold; include non_paired edges as specified by user
    def select_edges(self, pair_id, threshold, edges_only=False, include_nonpaired=[], exclude_nonpaired=[], left=[], right=[]):

        _, ds, ds_edges = self.downstream(pair_id, threshold)
        ds_edges, _ = self.edge_threshold(ds_edges, threshold, 'downstream', include_nonpaired=include_nonpaired, left=left, right=right)
        overthres_ds_edges = ds_edges[ds_edges.overthres==True]
        overthres_ds_edges.reset_index(inplace=True)
        overthres_ds_edges.drop(labels=['index', 'overthres'], axis=1, inplace=True)

        if(edges_only==False):
            return(overthres_ds_edges, np.unique(overthres_ds_edges.downstream_pair_id))
        if(edges_only):
            return(overthres_ds_edges)
    
    # generate edge list for whole matrix with some threshold
    def threshold_edge_list(self, all_sources, matrix_nonpaired, threshold, left, right):
        all_edges = Parallel(n_jobs=-1)(delayed(self.select_edges)(pair, threshold, edges_only=True, include_nonpaired=matrix_nonpaired, left=left, right=right) for pair in tqdm(all_sources))
        all_edges_combined = [x for x in all_edges if type(x)==pd.DataFrame]
        all_edges_combined = pd.concat(all_edges_combined, axis=0)
        all_edges_combined.reset_index(inplace=True, drop=True)
        return(all_edges_combined)

    # convert paired edge list with pair-wise threshold back to normal edge list, input from threshold_edge_list()
    # note that neurons with bilateral dendrites aren't treated in any special way, so they may be indicated as contralateral edges even if that's inaccurate/complicated
    def split_paired_edges(self, all_edges_combined, left, right, flip_weirdos=True):
        pairs = self.pairs

        # note that edge_weights are from the perspective of the downstream neuron, i.e. %input onto their dendrite
        all_edges_combined_split = []
        for i in range(len(all_edges_combined.index)):
            row = all_edges_combined.iloc[i]
            if((row.upstream_status=='paired') & (row.downstream_status=='paired')):
                if(row.type=='ipsilateral'):
                    all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
                    all_edges_combined_split.append([Promat.identify_pair(row.upstream_pair_id, pairs), Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
                if(row.type=='contralateral'):
                    all_edges_combined_split.append([row.upstream_pair_id, Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
                    all_edges_combined_split.append([Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.left, row.type, row.upstream_status, row.downstream_status])

            # note that pair_ids are really skeleton IDs for nonpaired neurons; this allows one to compare to left/right annotations
            # this comparison is required because the location of nonpaired -> pair edges depends on whether the nonpaired is left or right
            if((row.upstream_status=='nonpaired') & (row.downstream_status=='paired')):
                if(row.upstream_pair_id in left):
                    if(row.type=='ipsilateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
                    if(row.type=='contralateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, Promat.identify_pair(row.downstream_pair_id, pairs), 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
                if(row.upstream_pair_id in right):
                    if(row.type=='ipsilateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, Promat.identify_pair(row.downstream_pair_id, pairs), 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
                    if(row.type=='contralateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'right', 'left', row.left, row.type, row.upstream_status, row.downstream_status])

            # use the downstream_pair_id because this is really just skeleton ID for nonpaired neurons
            # therefore one can compare to left/right annotations to determine which hemisphere it belongs to
            if((row.upstream_status=='paired') & (row.downstream_status=='nonpaired')):
                if(row.downstream_pair_id in left):
                    if(row.type=='ipsilateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
                    if(row.type=='contralateral'):
                        all_edges_combined_split.append([Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
                if(row.downstream_pair_id in right):
                    if(row.type=='ipsilateral'):
                        all_edges_combined_split.append([Promat.identify_pair(row.upstream_pair_id, pairs), row.downstream_pair_id, 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
                    if(row.type=='contralateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])

            # use the downstream_pair_id because this is really just skeleton ID for nonpaired neurons
            # therefore one can compare to left/right annotations to determine which hemisphere it belongs to
            if((row.upstream_status=='nonpaired') & (row.downstream_status=='nonpaired')):
                if(row.downstream_pair_id in left):
                    if(row.type=='ipsilateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'left', row.left, row.type, row.upstream_status, row.downstream_status])
                    if(row.type=='contralateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'right', 'left', row.left, row.type, row.upstream_status, row.downstream_status])

                if(row.downstream_pair_id in right):
                    if(row.type=='ipsilateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'right', 'right', row.right, row.type, row.upstream_status, row.downstream_status])
                    if(row.type=='contralateral'):
                        all_edges_combined_split.append([row.upstream_pair_id, row.downstream_pair_id, 'left', 'right', row.right, row.type, row.upstream_status, row.downstream_status])

        all_edges_combined_split = pd.DataFrame(all_edges_combined_split, columns = ['upstream_skid', 'downstream_skid', 'upstream_side', 'downstream_side', 'edge_weight', 'type', 'upstream_status', 'downstream_status'])
        return(all_edges_combined_split)
        
    # generate edge list for whole matrix
    def edge_list(self, exclude_loops=False):
        edges = []
        for i in range(len(self.adj.index)):
            for j in range(len(self.adj.columns)):
                if(exclude_loops):
                    if((self.adj.iloc[i, j]>0) & (i!=j)):
                        edges.append([self.adj.index[i], self.adj.columns[j]])
                if(exclude_loops==False):
                    if(self.adj.iloc[i, j]>0):
                        edges.append([self.adj.index[i], self.adj.columns[j]])

        edges = pd.DataFrame(edges, columns = ['upstream_pair_id', 'downstream_pair_id'])
        return(edges)

    # generate a binary connectivity matrix that displays number of hops between neuron types
    def hop_matrix(self, layer_id_skids, source_leftid, destination_leftid, include_start=False):
        mat = pd.DataFrame(np.zeros(shape = (len(source_leftid), len(destination_leftid))), 
                            index = source_leftid, 
                            columns = destination_leftid)

        for index in mat.index:
            data = layer_id_skids.loc[index, :]
            for i, hop in enumerate(data):
                for column in mat.columns:
                    if(column in hop):
                        if(include_start==True): # if the source of the hop signal is the first layer
                            mat.loc[index, column] = i
                        if(include_start==False): # if the first layer is the first layer downstream of source
                            mat.loc[index, column] = i+1


        max_value = mat.values.max()
        mat_plotting = mat.copy()

        for index in mat_plotting.index:
            for column in mat_plotting.columns:
                if(mat_plotting.loc[index, column]>0):
                    mat_plotting.loc[index, column] = 1 - (mat_plotting.loc[index, column] - max_value)

        return(mat, mat_plotting)
            
class Promat():

    # default method to import pair list and process it to deal with duplicated neurons
    @staticmethod
    def get_pairs(pairs_path='data/pairs/pairs-2021-04-06.csv', flip_weirdos=True):
        print(f'Path to pairs list is: {pairs_path}')

        pairs = pd.read_csv(pairs_path, header = 0) # import pairs, manually determined with help from Heather Patsolic and Ben Pedigo's scripts
        pairs = pairs.loc[:, ['leftid', 'rightid']] # only include useful columns

        # duplicated right-side neurons to throw out for simplicity 
        duplicated = pymaid.get_skids_by_annotation('mw duplicated neurons to delete')
        duplicated_index = np.where(sum([pairs.rightid==x for x in duplicated])==1)[0]
        pairs = pairs.drop(duplicated_index)

        # change left/right ids of contra-contra neurons so they behave properly in downstream analysis
        #   these neurons have somas on one brain hemisphere and dendrites/axons on the other
        #   and so they functionally all completely contralateral and can therefore be considered ipsilateral neurons
        if(flip_weirdos):
            # identify contra-contra neurons
            contra_contra = np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite'))
            contra_contra_pairs = Promat.extract_pairs_from_list(contra_contra, pairs)[0]
            if(len(contra_contra_pairs)>0):
                
                # flip left/right neurons in contra-contra neurons
                for index in contra_contra_pairs.index:
                    cc_left = contra_contra_pairs.loc[index, 'leftid']
                    cc_right = contra_contra_pairs.loc[index, 'rightid']

                    pairs.loc[pairs[pairs.leftid==cc_left].index, 'rightid'] = cc_left
                    pairs.loc[pairs[pairs.leftid==cc_left].index, 'leftid'] = cc_right

        return(pairs)

    # returns all skids in left or right side of the brain, depending on whether side = 'left' or 'right'
    def get_hemis(side=None, flip_weirdos=True):
        left = pymaid.get_skids_by_annotation('mw left')
        right = pymaid.get_skids_by_annotation('mw right')

        if(flip_weirdos):
            # identifying contra-contra neurons so they can be flipped to opposite side of brain
            neurons_to_flip = np.intersect1d(pymaid.get_skids_by_annotation('mw contralateral axon'), pymaid.get_skids_by_annotation('mw contralateral dendrite'))
            neurons_to_flip_left = [skid for skid in neurons_to_flip if skid in left]
            neurons_to_flip_right = [skid for skid in neurons_to_flip if skid in right]

            # removing neurons_to_flip and adding to the other side
            left = list(np.setdiff1d(left, neurons_to_flip_left)) + neurons_to_flip_right
            right = list(np.setdiff1d(right, neurons_to_flip_right)) + neurons_to_flip_left

        if(side=='left'): return(left)
        if(side=='right'): return(right)
        if(side==None): return([left, right])

    # converts any df with df.index = list of skids to a multiindex with ['pair_status', 'pair_id', 'skid']
    #   'pair_status': pairs / nonpaired
    #   'pair_id': left skid of a pair or simply the skid of a nonpaired neuron
    @staticmethod
    def convert_df_to_pairwise(df, pairs=None):

        if(pairs==None):
            pairs = Promat.get_pairs()
        brain_pairs, brain_unpaired, brain_nonpaired = Promat.extract_pairs_from_list(df.index, pairList = pairs)
            
        # left_right interlaced order for brain matrix
        brain_pair_order = []
        for i in range(0, len(brain_pairs)):
            brain_pair_order.append(brain_pairs.iloc[i].leftid)
            brain_pair_order.append(brain_pairs.iloc[i].rightid)

        order = brain_pair_order + list(brain_nonpaired.nonpaired)
        interlaced = df.loc[order, :]

        index_df = pd.DataFrame([['pairs', Promat.get_paired_skids(skid, pairs)[0], skid] for skid in brain_pair_order] + [['nonpaired', skid, skid] for skid in list(brain_nonpaired.nonpaired)], 
                                columns = ['pair_status', 'pair_id', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)
        interlaced.index = index

        return(interlaced)

    # trim out neurons not currently in the brain matrix
    @staticmethod
    def trim_missing(skidList, brainMatrix):
        trimmedList = []
        for i in skidList:
            if(i in brainMatrix.index):
                trimmedList.append(i)
            else:
                print("*WARNING* skid: %i is not in whole brain matrix" % (i))
        
        return(trimmedList)

    # identify skeleton ID of hemilateral neuron pair, based on CSV pair list
    @staticmethod
    def identify_pair(skid, pairList):

        pair_skid = []
        
        if(skid in pairList["leftid"].values):
            pair_skid = pairList["rightid"][pairList["leftid"]==skid].iloc[0]

        if(skid in pairList["rightid"].values):
            pair_skid = pairList["leftid"][pairList["rightid"]==skid].iloc[0]

        if((skid not in pairList['rightid'].values) & (skid not in pairList['leftid'].values)):
            print(f'skid {skid} is not in paired list')
            pair_skid = skid

        return(pair_skid)

    # returns paired skids in array [left, right]; can input either left or right skid of a pair to identify
    @staticmethod
    def get_paired_skids(skid, pairList):

        if(skid in pairList["leftid"].values):
            pair_right = pairList["rightid"][pairList["leftid"]==skid].iloc[0]
            pair_left = skid

        if(skid in pairList["rightid"].values):
            pair_left = pairList["leftid"][pairList["rightid"]==skid].iloc[0]
            pair_right = skid

        if((skid in pairList["leftid"].values) == False and (skid in pairList["rightid"].values) == False):
            print(f"skid {skid} is not in paired list")
            return([skid])

        return([pair_left, pair_right])

    # converts array of skids into left-right pairs in separate columns
    # puts unpaired and nonpaired neurons in different lists
    @staticmethod
    def extract_pairs_from_list(skids, pairList):

        pairs = pd.DataFrame([], columns = ['leftid', 'rightid'])
        unpaired = pd.DataFrame([], columns = ['unpaired'])
        nonpaired = pd.DataFrame([], columns = ['nonpaired'])
        for i in skids:
            if((int(i) not in pairList.leftid.values) & (int(i) not in pairList.rightid.values)):
                nonpaired = nonpaired.append({'nonpaired': int(i)}, ignore_index=True)
                continue

            if((int(i) in pairList["leftid"].values) & (Promat.get_paired_skids(int(i), pairList)[1] in skids)):
                pair = Promat.get_paired_skids(int(i), pairList)
                pairs = pairs.append({'leftid': pair[0], 'rightid': pair[1]}, ignore_index=True)

            if(((int(i) in pairList["leftid"].values) & (Promat.get_paired_skids(int(i), pairList)[1] not in skids)|
                (int(i) in pairList["rightid"].values) & (Promat.get_paired_skids(int(i), pairList)[0] not in skids))):
                unpaired = unpaired.append({'unpaired': int(i)}, ignore_index=True)

        pairs = pd.DataFrame(pairs)
        unpaired = pd.DataFrame(unpaired)
        nonpaired = pd.DataFrame(nonpaired)
        return(pairs, unpaired, nonpaired)

    # loads neurons pairs from selected pymaid annotation
    @staticmethod
    def load_pairs_from_annotation(annot, pairList, return_type='pairs', skids=None, use_skids=False):
        if(use_skids==False):
            skids = pymaid.get_skids_by_annotation(annot)
            
        pairs = Promat.extract_pairs_from_list(skids, pairList)
        if(return_type=='pairs'):
            return(pairs[0])
        if(return_type=='unpaired'):
            return(pairs[1])
        if(return_type=='nonpaired'):
            return(pairs[2])
        if(return_type=='all_pair_ids'):
            pairs_pair_id = list(pairs[0].leftid)
            nonpaired_pair_id = list(pairs[2].nonpaired)
            combined = pairs_pair_id + nonpaired_pair_id
            return(combined)

        # include nonpaired neurons and ['leftid', 'rightid'] columns; duplicated leftid/rightid for nonpaired neurons
        if(return_type=='all_pair_ids_bothsides'):
            pairs_pair_id = list(pairs[0].leftid)
            nonpaired_pair_id = list(pairs[2].nonpaired)
            combined_left = pairs_pair_id + nonpaired_pair_id

            pairs_id_right = list(pairs[0].rightid)
            combined_right = pairs_id_right + nonpaired_pair_id
            combined = pd.DataFrame(zip(combined_left, combined_right), columns=['leftid', 'rightid'])

            return(combined)

    # loads neurons pairs from selected pymaid annotation
    @staticmethod
    def get_pairs_from_list(skids, pairList, return_type='pairs'):
        pairs = Promat.extract_pairs_from_list(skids, pairList)
        if(return_type=='pairs'):
            return(pairs[0])
        if(return_type=='unpaired'):
            return(pairs[1])
        if(return_type=='nonpaired'):
            return(pairs[2])
        if(return_type=='all_pair_ids'):
            pairs_pair_id = list(pairs[0].leftid)
            nonpaired_pair_id = list(pairs[2].nonpaired)
            combined = pairs_pair_id + nonpaired_pair_id
            return(combined)

    # generates interlaced left-right pair adjacency matrix with nonpaired neurons at bottom and right
    @staticmethod
    def interlaced_matrix(adj_df, pairs):
        brain_pairs, brain_unpaired, brain_nonpaired = Promat.extract_pairs_from_list(mg.meta.index, pairs)

        # left_right interlaced order for brain matrix
        brain_pair_order = []
        for i in range(0, len(brain_pairs)):
            brain_pair_order.append(brain_pairs.iloc[i].leftid)
            brain_pair_order.append(brain_pairs.iloc[i].rightid)

        interlaced_mat = adj_df.loc[brain_pair_order + list(brain_nonpaired), brain_pair_order + list(brain_nonpaired)]

        index_df = pd.DataFrame([['pairs', skid] for skid in brain_pair_order] + [['nonpaired', skid] for skid in list(brain_nonpaired)], 
                                columns = ['pair_status', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)

        interlaced_mat.index = index
        interlaced_mat.columns = index
        return(interlaced_mat)

    # converts matrix to fraction_input matrix by dividing every column by dendritic input
    @staticmethod
    def fraction_input_matrix(adj_df, mg, axon=False):
        for column in adj_df.columns:
            if(axon):
                axon_input = mg.meta.loc[column].axon_input
                adj_df.loc[:, column] = adj_df.loc[:, column]/axon_input

            if(axon==False):
                dendrite_input = mg.meta.loc[column].dendrite_input
                adj_df.loc[:, column] = adj_df.loc[:, column]/dendrite_input

        return(adj_df)

    # converts a interlaced left-right pair adjacency matrix into a binary connection matrix based on some threshold
    @staticmethod
    def binary_matrix(adj, threshold, total_threshold):

        oddCols = np.arange(0, len(adj.columns), 2)
        oddRows = np.arange(0, len(adj.index), 2)

        # column names are the skid of left neuron from pair
        binMat = np.zeros(shape=(len(oddRows),len(oddCols)))
        binMat = pd.DataFrame(binMat, columns = adj.columns[oddCols], index = adj.index[oddRows])

        for i in oddRows:
            for j in oddCols:
                sum_all = adj.iat[i, j] + adj.iat[i+1, j+1] + adj.iat[i+1, j] + adj.iat[i, j+1]
                if(adj.iat[i, j] >= threshold and adj.iat[i+1, j+1] >= threshold and sum_all >= total_threshold):
                    binMat.iat[int(i/2), int(j/2)] = 1

                if(adj.iat[i+1, j] >= threshold and adj.iat[i, j+1] >= threshold and sum_all >= total_threshold):
                    binMat.iat[int(i/2), int(j/2)] = 1
            
        return(binMat)

    # summing input from a group of upstream neurons
    # generating DataFrame with sorted leftid, rightid, summed-input left, summed-input right

    # SOMETHING IS WRONG***
    # It works as local function within particular .py file, but not when called through process_matrix.py
    @staticmethod
    def summed_input(group_skids, matrix, pairList):
        submatrix = matrix.loc[group_skids, :]
        submatrix = submatrix.sum(axis = 0)

        cols = ['leftid', 'rightid', 'leftid_input', 'rightid_input']
        summed_paired = []

        for i in range(0, len(pairList['leftid'])):
            if(pairList['leftid'][i] in submatrix.index):
                left_identifier = pairList['leftid'][i]
                left_sum = submatrix.loc[left_identifier]
            
                right_identifier = Promat.identify_pair(pairList['leftid'][i], pairList)
                right_sum = submatrix.loc[right_identifier]
                    
                summed_paired.append([left_identifier, right_identifier, left_sum, right_sum])

        summed_paired = pd.DataFrame(summed_paired, columns= cols)
        return(summed_paired)

    # identifies downstream neurons based on summed threshold (summed left/right input) and low_threshold (required edge weight on weak side)
    @staticmethod
    def identify_downstream(sum_df, summed_threshold, low_threshold):
        downstream = []
        for i in range(0, len(sum_df['leftid'])):
            if((sum_df['leftid_input'].iloc[i] + sum_df['rightid_input'].iloc[i])>=summed_threshold):

                if(sum_df['leftid_input'].iloc[i]>sum_df['rightid_input'].iloc[i] and sum_df['rightid_input'].iloc[i]>=low_threshold):
                    downstream.append(sum_df.iloc[i])

                if(sum_df['rightid_input'].iloc[i]>sum_df['leftid_input'].iloc[i] and sum_df['leftid_input'].iloc[i]>=low_threshold):
                    downstream.append(sum_df.iloc[i])

        return(pd.DataFrame(downstream))

    # compares neuron similarity based on inputs, outputs, or both
    # outputs a matrix where each row/column is a pair of neurons
    # NOT currently working
    @staticmethod    
    def similarity_matrix(matrix_path, type):
        matrix = pd.read_csv(matrix_path, header=0, index_col=0, quotechar='"', skipinitialspace=True)

        oddCols = np.arange(0, len(matrix.columns), 2)
        oddRows = np.arange(0, len(matrix.index), 2)

        # column names are the skid of left neuron from pair
        sim_matrix = np.zeros(shape=(len(oddRows),len(oddCols)))
        sim_matrix = pd.DataFrame(sim_matrix, columns = matrix.columns[oddCols], index = matrix.index[oddRows])

        return(sim_matrix)

    @staticmethod
    def writeCSV(data, path):
        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in data:
                writer.writerow(row)

        print("Write complete")
        return()

    @staticmethod
    def pull_adj(type_adj, subgraph):

        adj = pd.read_csv(f'data/adj/all-neurons_{type_adj}.csv', index_col = 0).rename(columns=int)
        
        if(subgraph=='brain'):
            brain = pymaid.get_skids_by_annotation('mw brain paper clustered neurons')
            adj = adj.loc[np.intersect1d(adj.index, brain), np.intersect1d(adj.index, brain)]

        if(subgraph=='brain and accessory'):
            brain = pymaid.get_skids_by_annotation('mw brain paper clustered neurons') + pymaid.get_skids_by_annotation('mw brain accessory neurons')
            adj = adj.loc[np.intersect1d(adj.index, brain), np.intersect1d(adj.index, brain)]

        #if(subgraph=='brain-A1'):

        #if(subgraph=='A1'):

        return(adj)

    # recursive function that identifies all downstream partners X-hops away from source
    # uses pregenerated edge list from threshold_edge_list() or the split-pair version
    @staticmethod
    def downstream_multihop(edges, sources, hops, hops_iter=1, pairs_combined=False, exclude_source=True, exclude=[], exclude_skids_from_source=[]):
        if(pairs_combined):
            id1 = 'upstream_pair_id'
            id2 = 'downstream_pair_id'
        if(pairs_combined==False): 
            id1 = 'upstream_skid'
            id2 = 'downstream_skid'

        edges_df = edges.set_index(id1)

        if(hops_iter>1): sources = list(np.setdiff1d(sources, exclude_skids_from_source)) # exclude user-selected neurons from sources
        ds = list(np.unique(edges_df.loc[np.intersect1d(sources, edges_df.index), id2]))

        if(exclude_source): ds = list(np.setdiff1d(ds, sources)) # exclude source from downstream
        ds = list(np.setdiff1d(ds, exclude)) # exclude user-selected neurons from downstream partners

        if(hops_iter==hops):
            return([ds])
        else:
            hops_iter += 1
            return([ds] + Promat.downstream_multihop(edges=edges, sources=ds, hops=hops, hops_iter=hops_iter))

    # recursive function that identifies all upstream partners X-hops away from source
    # uses pregenerated edge list from threshold_edge_list() or the split-pair version
    @staticmethod
    def upstream_multihop(edges, sources, hops, hops_iter=1, pairs_combined=False, exclude_source=True, exclude=[], exclude_skids_from_source=[]):
        if(pairs_combined):
            id1 = 'downstream_pair_id'
            id2 = 'upstream_pair_id'
        if(pairs_combined==False): 
            id1 = 'downstream_skid'
            id2 = 'upstream_skid'

        edges_df = edges.set_index(id1)

        if(hops_iter>1): sources = list(np.setdiff1d(sources, exclude_skids_from_source)) # exclude user-selected neurons from sources
        us = list(np.unique(edges_df.loc[np.intersect1d(sources, edges_df.index), id2]))

        if(exclude_source): us = list(np.setdiff1d(us, sources)) # exclude source from upstream
        us = list(np.setdiff1d(us, exclude)) # exclude user-selected neurons from upstream partners

        if(hops_iter==hops):
            return([us])
        else:
            hops_iter += 1
            return([us] + Promat.upstream_multihop(edges=edges, sources=us, hops=hops, hops_iter=hops_iter, exclude_source=exclude_source, exclude=exclude))

