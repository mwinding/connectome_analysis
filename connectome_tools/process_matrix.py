# module for processing adjacency matrices in various ways

import pandas as pd
import numpy as np
import csv

class Adjacency_matrix(mg, pairs, mat_type):
    self.mg = mg
    self.pairs = pairs
    self.mat_type = mat_type # 'axo-dendritic', 'axo-axonic', etc.
    self.adj_df = pd.DataFrame(mg.adj, index = mg.meta.index, columns = mg.meta.index)

    def interlaced_matrix(self):
        brain_pairs, brain_unpaired, brain_nonpaired = Promat.extract_pairs_from_list(self.mg.meta.index, self.pairs)

        # left_right interlaced order for brain matrix
        brain_pair_order = []
        for i in range(0, len(brain_pairs)):
            brain_pair_order.append(brain_pairs.iloc[i].leftid)
            brain_pair_order.append(brain_pairs.iloc[i].rightid)

        interlaced_mat = self.adj_df.loc[brain_pair_order + list(brain_nonpaired), brain_pair_order + list(brain_nonpaired)]

        index_df = pd.DataFrame([['pairs', skid] for skid in brain_pair_order] + [['nonpaired', skid] for skid in list(brain_nonpaired)], 
                                columns = ['pair_status', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)

        interlaced_mat.index = index
        interlaced_mat.columns = index
        return(interlaced_mat)

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
            return(0)

        return([pair_left, pair_right])

    # converts array of skids into left-right pairs in separate columns
    # puts unpaired and nonpaired neurons in different lists
    @staticmethod
    def extract_pairs_from_list(skids, pairList):
        pairs = []
        unpaired = []
        nonpaired = []
        for i in skids:
            if((int(i) not in pairList.leftid.values) & (int(i) not in pairList.rightid.values)):
                nonpaired.append({'nonpaired': int(i)})
                continue

            if((int(i) in pairList["leftid"].values) & (Promat.get_paired_skids(int(i), pairList)[1] in skids)):
                pair = Promat.get_paired_skids(int(i), pairList)
                pairs.append({'leftid': pair[0], 'rightid': pair[1]})

            if(((int(i) in pairList["leftid"].values) & (Promat.get_paired_skids(int(i), pairList)[1] not in skids)|
                (int(i) in pairList["rightid"].values) & (Promat.get_paired_skids(int(i), pairList)[0] not in skids))):
                unpaired.append({'unpaired': int(i)})

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