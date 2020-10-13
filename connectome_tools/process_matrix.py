# module for processing adjacency matrices in various ways

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

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

            return(source_skids, ds_neurons_skids)


    def upstream(self, source, threshold, exclude = []):
        adj = self.adj_pairwise

        source_pair_id = np.unique([x[1] for x in self.adj_inter.loc[(slice(None), slice(None), source), :].index])

        bin_mat = adj.loc[:, (slice(None), source_pair_id)] > threshold
        bin_row = np.where(bin_mat.sum(axis = 1) > 0)[0]
        ds_neurons = bin_mat.index[bin_row]

        us_neurons_skids = []
        for pair in ds_neurons:
            if((pair[0] == 'pairs') & (pair[1] not in exclude)):
                us_neurons_skids.append(pair[1])
                us_neurons_skids.append(Promat.identify_pair(pair[1], self.pairs))
            if((pair[0] == 'nonpaired') & (pair[1] not in exclude)):
                us_neurons_skids.append(pair[1])

        return(us_neurons_skids)

    def downstream_multihop(self, source, threshold, min_members=0, hops=10):
        _, ds = self.downstream(source, threshold, exclude=source)
        _, ds = self.edge_threshold(source, ds, threshold, direction='downstream')

        before = source + ds

        layers = []
        layers.append(ds)

        for i in range(0,hops):
            source = ds
            _, ds = self.downstream(source, threshold, exclude=before) 
            _, ds = self.edge_threshold(source, ds, threshold, direction = 'downstream')

            if((len(ds)!=0) & (len(ds)>=min_members)):
                layers.append(ds)
                before = before + ds

        return(layers)

    def upstream_multihop(self, source, threshold, min_members=10, hops=10):
        us = self.upstream(source, threshold, exclude=source)
        _, us = self.edge_threshold(source, us, threshold, direction='upstream')

        before = source + us

        layers = []
        layers.append(us)

        for i in range(0,hops):
            source = us
            us = self.upstream(source, threshold, exclude = before)
            _, us = self.edge_threshold(source, us, threshold, direction='upstream')

            if((len(us)!=0) & (len(us)>=min_members)):
                layers.append(us)
                before = before + us

        return(layers)

    # checking additional threshold criteria after identifying neurons over summed threshold
    # can also just input all possible downstream neurons, but it will be slow
    # still is super slow because it queries all edges between source and ds neurons, not just ones that passed last threshold
    def edge_threshold(self, source_skids, partner_neurons_skids, threshold, direction, strict=False):

        adj = self.adj_inter.copy()

        source_pair_id = np.unique([x[1] for x in adj.loc[(slice(None), slice(None), source_skids), :].index])
        partner_pair_id = np.unique([x[1] for x in adj.loc[(slice(None), slice(None), partner_neurons_skids), :].index])

        all_edges = []
        for source in source_pair_id:
            for partner in partner_pair_id:
                if(direction=='downstream'):
                    edges = adj.loc[(slice(None), source), (slice(None), partner)]
                if(direction=='upstream'):
                    edges = adj.loc[(slice(None), partner), (slice(None), source)]

                if((len(edges.index)==2) & (len(edges.columns)==2)): #paired
                    if(direction=='downstream'):
                        edges = pd.DataFrame([[source, partner, edges.iloc[0,0], edges.iloc[1,1], False, 'ipsilateral'],
                                            [source, partner, edges.iloc[1,0], edges.iloc[0,1], False, 'contralateral']], 
                                            columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type'])

                    if(direction=='upstream'):
                        edges = pd.DataFrame([[partner, source, edges.iloc[0,0], edges.iloc[1,1], False, 'ipsilateral'],
                                            [partner, source, edges.iloc[1,0], edges.iloc[0,1], False, 'contralateral']], 
                                            columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type'])

                    if(strict==True):
                        # is each edge weight over threshold?
                        for index in edges.index:
                            if((edges.loc[index].left>threshold) & (edges.loc[index].right>threshold)):
                                edges.loc[index, 'overthres'] = True

                    if(strict==False):
                        # is average edge weight over threshold
                        for index in edges.index:
                            if(((edges.loc[index].left + edges.loc[index].right)/2) > threshold):
                                edges.loc[index, 'overthres'] = True

                    # are both edges present?
                    for index in edges.index:
                        if((edges.loc[index].left==0) | (edges.loc[index].right==0)):
                            edges.loc[index, 'overthres'] = False

                    all_edges.append(edges.values[0])
                    all_edges.append(edges.values[1])

                ''' # not currently implemented non-paired connections
                if(len(edges)==1): #unpaired
                    edges = pd.DataFrame([[edges.iloc[0,0], edges.iloc[0,1], False, 'unpaired'], 
                                        columns = ['left', 'right', 'overthres', 'type'])
                '''

        all_edges = pd.DataFrame(all_edges, columns = ['upstream_pair_id', 'downstream_pair_id', 'left', 'right', 'overthres', 'type'])
        
        if(direction=='downstream'):
            partner_skids = np.unique(all_edges[all_edges.overthres==True].downstream_pair_id) # identify downstream pairs
        if(direction=='upstream'):
            partner_skids = np.unique(all_edges[all_edges.overthres==True].upstream_pair_id) # identify upstream pairs
        
        partner_skids = [x[2] for x in adj.loc[(slice(None), partner_skids), :].index] # convert from pair_id to skids

        return(all_edges, partner_skids)

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
                skids = skids + [['']]*(max_layers-len(skids)) # make sure each column has same num elements

            mat_neuron_skids[f'{layer_names[i]}'] = skids

        id_layers = pd.DataFrame(mat_neurons, index = layer_names, columns = [f'Layer {i}' for i in range(0,max_layers)])
        id_layers_skids = mat_neuron_skids

        return(id_layers, id_layers_skids)

            
       
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

    #def reorderInputsOutputs_toRow(matrix):
    #    for i in matrix[,1]