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

    def __init__(self, adj, skids, pairs, input_counts, mat_type):
        self.skids = skids
        self.pairs = pairs
        self.input_counts = input_counts
        self.mat_type = mat_type # 'axo-dendritic', 'axo-axonic', etc.
        self.adj = pd.DataFrame(adj, index = skids, columns = skids)
        self.adj_fract = self.fraction_input_matrix()
        self.adj_inter = self.interlaced_matrix()
        self.adj_pairwise = self.average_pairwise_matrix()

    def fraction_input_matrix(self, axon=False):
        adj_fract = self.adj.copy()
        for column in adj_fract.columns:
            if(axon):
                axon_input = self.input_counts.loc[column].axon_input
                if(axon_input == 0):
                    adj_fract.loc[:, column] = 0
                if(axon_input > 0):
                    adj_fract.loc[:, column] = adj_fract.loc[:, column]/axon_input

            if(axon==False):
                dendrite_input = self.input_counts.loc[column].dendrite_input
                if(dendrite_input == 0):
                    adj_fract.loc[:, column] = 0
                if(dendrite_input > 0):
                    adj_fract.loc[:, column] = adj_fract.loc[:, column]/dendrite_input

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

    def downstream_multihop(self, source, threshold, min_members=0, hops=10, exclude = [], strict=False, allow_source_ds=False):
        if(allow_source_ds==False):
            _, ds, edges = self.downstream(source, threshold, exclude=(source + exclude))
        if(allow_source_ds):
            _, ds, edges = self.downstream(source, threshold, exclude=(exclude))

        _, ds = self.edge_threshold(edges, threshold, direction='downstream', strict=strict)

        if(allow_source_ds==False):
            before = source + ds
        if(allow_source_ds):
            before = ds

        layers = []
        layers.append(ds)

        for i in range(0,(hops-1)):
            source = ds
            _, ds, edges = self.downstream(source, threshold, exclude=before) 
            _, ds = self.edge_threshold(edges, threshold, direction='downstream', strict=strict)

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
    def edge_threshold(self, edges, threshold, direction, strict=False, include_nonpaired=[], left=[], right=[]):

        adj = self.adj_inter.copy()

        all_edges = []
        for edge in edges:
            specific_edges = adj.loc[(slice(None), edge[0]), (slice(None), edge[1])]

            us_pair_status = adj.loc[(slice(None), slice(None), edge[0]), :].index[0][0]
            ds_pair_status = adj.loc[(slice(None), slice(None), edge[1]), :].index[0][0]

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
            if((us_pair_status == 'pairs') & (ds_pair_status == 'nonpaired') & (edge[1] in include_nonpaired)):

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
            if((us_pair_status == 'nonpaired') & (ds_pair_status == 'pairs') & (edge[0] in include_nonpaired)):

                if(edge[0] in left):
                    specific_edges = pd.DataFrame([[edge[0], edge[1], specific_edges.iloc[0, 0], 0, False, 'ipsilateral', 'nonpaired', 'paired'],
                                                    [edge[0], edge[1], specific_edges.iloc[0, 1], 0, False, 'contralateral', 'nonpaired', 'paired']], 
                                                    columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type', 'upstream_status', 'downstream_status'])

                if(edge[0] in right):
                    specific_edges = pd.DataFrame([[edge[0], edge[1], 0, specific_edges.iloc[0, 0], False, 'contralateral', 'nonpaired', 'paired'],
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
    
    # generate edge list for whole matrix
    def generate_edge_list(self, all_sources, matrix_nonpaired, threshold, left, right):
        all_edges = Parallel(n_jobs=-1)(delayed(self.select_edges)(pair, threshold, edges_only=True, include_nonpaired=matrix_nonpaired, left=left, right=right) for pair in tqdm(all_sources))
        all_edges_combined = [x for x in all_edges if type(x)==pd.DataFrame]
        all_edges_combined = pd.concat(all_edges_combined, axis=0)
        all_edges_combined.reset_index(inplace=True, drop=True)
        return(all_edges_combined)

    def layer_id(self, layers, layer_names, celltype_skids):
        max_layers = max([len(layer) for layer in layers])

        mat_neurons = np.zeros(shape = (len(layers), max_layers))
        mat_neuron_skids = pd.DataFrame()
        for i in range(0,len(layers)):
            skids = []
            for j in range(0,len(layers[i])):
                neurons = np.intersect1d(layers[i][j], celltype_skids)
                count = len(neurons)

                mat_neurons[i, j] = count
                skids.append(neurons)
            
            if(len(skids) != max_layers):
                skids = skids + [[]]*(max_layers-len(skids)) # make sure each column has same num elements

            mat_neuron_skids[layer_names[i]] = skids

        id_layers = pd.DataFrame(mat_neurons, index = layer_names, columns = [f'Layer {i+1}' for i in range(0,max_layers)])
        id_layers_skids = mat_neuron_skids

        return(id_layers, id_layers_skids)

    def plot_layer_types(self, layer_types, layer_names, layer_colors, layer_vmax, pair_ids, figsize, save_path, threshold, hops):

        col = layer_colors

        pair_list = []
        for pair in pair_ids:
            mat = np.zeros(shape=(len(layer_types), len(layer_types[0].columns)))
            for i, layer_type in enumerate(layer_types):
                mat[i, :] = layer_type.loc[pair]

            pair_list.append(mat)

        # loop through pairs to plot
        for i, pair in enumerate(pair_list):

            data = pd.DataFrame(pair, index = layer_names)
            mask_list = []
            for i_iter in range(0, len(data.index)):
                mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
                mask[i_iter, :] = [False]*len(data.columns)
                mask_list.append(mask)

            fig, axs = plt.subplots(
                1, 1, figsize=figsize
            )
            for j, mask in enumerate(mask_list):
                vmax = layer_vmax[j]
                ax = axs
                annotations = data.astype(int).astype(str)
                annotations[annotations=='0']=''
                sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

            plt.savefig(f'{save_path}hops{hops}_{i}_{pair_ids[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')


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
            print("skid %i is not in paired list" % (skid))
            return([skid, skid])

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
    def load_pairs_from_annotation(annot, pairList, return_type='pairs'):
        skids = pymaid.get_skids_by_annotation(annot)
        pairs = Promat.extract_pairs_from_list(skids, pairList)
        if(return_type=='pairs'):
            return(pairs[0])
        if(return_type=='unpaired'):
            return(pairs[1])
        if(return_type=='nonpaired'):
            return(pairs[2])

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

    # set of known celltypes, returned as skid lists
    @staticmethod
    def celltypes(more_celltypes=[], more_names=[]):
        A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
        A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
        br = pymaid.get_skids_by_annotation('mw brain neurons')
        MBON = pymaid.get_skids_by_annotation('mw MBON')
        MBIN = pymaid.get_skids_by_annotation('mw MBIN')
        LHN = pymaid.get_skids_by_annotation('mw LHN')
        CN = pymaid.get_skids_by_annotation('mw CN')
        KC = pymaid.get_skids_by_annotation('mw KC')
        RGN = pymaid.get_skids_by_annotation('mw RGN')
        dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
        pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC 1%')
        pre_dSEZ = pymaid.get_skids_by_annotation('mw pre-dSEZ 1%')
        pre_RGN = pymaid.get_skids_by_annotation('mw pre-RGN 1%')
        dVNC = pymaid.get_skids_by_annotation('mw dVNC')
        uPN = pymaid.get_skids_by_annotation('mw uPN')
        tPN = pymaid.get_skids_by_annotation('mw tPN')
        vPN = pymaid.get_skids_by_annotation('mw vPN')
        mPN = pymaid.get_skids_by_annotation('mw mPN')
        PN = uPN + tPN + vPN + mPN
        FBN = pymaid.get_skids_by_annotation('mw FBN')
        FB2N = pymaid.get_skids_by_annotation('mw FB2N')
        FBN_all = FBN + FB2N

        input_names = pymaid.get_annotated('mw brain inputs').name
        general_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd']
        input_skids_list = list(map(pymaid.get_skids_by_annotation, input_names))
        sens_all = [x for sublist in input_skids_list for x in sublist]

        asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
        asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
        asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')
        asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
        asc_all = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')

        LHN = list(np.setdiff1d(LHN, FBN_all + dVNC))
        CN = list(np.setdiff1d(CN, LHN + FBN_all + dVNC)) # 'CN' means exclusive CNs that are not FBN or LHN
        pre_dVNC = list(np.setdiff1d(pre_dVNC, MBON + MBIN + LHN + CN + KC + RGN + dSEZ + dVNC + PN + FBN_all + asc_all)) # 'pre_dVNC' must have no other category assignment
        pre_dSEZ = list(np.setdiff1d(pre_dSEZ, MBON + MBIN + LHN + CN + KC + RGN + dSEZ + dVNC + PN + FBN_all + asc_all + pre_dVNC)) # 'pre_dSEZ' must have no other category assignment
        pre_RGN = list(np.setdiff1d(pre_RGN, MBON + MBIN + LHN + CN + KC + RGN + dSEZ + dVNC + PN + FBN_all + asc_all + pre_dVNC + pre_RGN)) # 'pre_RGN' must have no other category assignment
        dSEZ = list(np.setdiff1d(dSEZ, MBON + MBIN + LHN + CN + KC + dVNC + PN + FBN_all + dVNC))

        few_synapses = pymaid.get_skids_by_annotation('mw brain few synapses')
        A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending

        celltypes = [sens_all, PN, LHN, MBIN, list(np.setdiff1d(KC, few_synapses)), MBON, FBN_all, CN, pre_RGN, pre_dSEZ, pre_dVNC, RGN, dSEZ, dVNC]
        celltype_names = ['Sens', 'PN', 'LHN', 'MBIN', 'KC', 'MBON', 'MB-FBN', 'CN', 'pre-RGN', 'pre-dSEZ','pre-dVNC', 'RGN', 'dSEZ', 'dVNC']

        if(len(more_celltypes)>0):
            celltypes = celltypes + more_celltypes
            celltype_names = celltype_names + more_names

        # only use celltypes that have non-zero number of members
        exists = [len(x)!=0 for x in celltypes]
        celltypes = [x for i,x in enumerate(celltypes) if exists[i]==True]
        celltype_names = [x for i,x in enumerate(celltype_names) if exists[i]==True]

        return(celltypes, celltype_names)

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

    #def reorderInputsOutputs_toRow(matrix):
    #    for i in matrix[,1]