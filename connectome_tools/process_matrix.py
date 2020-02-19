# module for processing adjacency matrices in various ways

import pandas as pd
import numpy as np

# trim out neurons not currently in the brain matrix
def trim_missing(skidList, brainMatrix):
    trimmedList = []
    for i in skidList:
        if(i in brainMatrix.index):
            trimmedList.append(i)
        else:
            print("*WARNING* skid: %i is not in whole brain matrix" % (i))
    
    return(trimmedList)

# identify skeleton ID of hemilateral neuron pair, based on CSV pair list
def identify_pair(skid, pairList):
    
    if(skid in pairList["leftid"].values):
        pair_skid = pairList["rightid"][pairList["leftid"]==skid].iloc[0]

    if(skid in pairList["rightid"].values):
        pair_skid = pairList["leftid"][pairList["rightid"]==skid].iloc[0]

    return(pair_skid)

# returns paired skids in array [left, right]; can input either left or right skid of a pair to identify
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
def extract_pairs_from_list(skids, pairList):
    pairs = []
    for i in skids:
        if(int(i) in pairList["leftid"].values):
            pair = get_paired_skids(int(i), pairList)
            pairs.append({'leftid': pair[0], 'rightid': pair[1]})

        # delaying with non-paired neurons
        # UNTESTED
        if (pair==0):
            break

    pairs = pd.DataFrame(pairs)
    return(pairs)

# converts a interlaced left-right pair adjacency matrix into a binary connection matrix based on some threshold
def binary_matrix(matrix_path, threshold): 
    matrix = pd.read_csv(matrix_path, header=0, index_col=0, quotechar='"', skipinitialspace=True)

    oddCols = np.arange(0, len(matrix.columns), 2)
    oddRows = np.arange(0, len(matrix.index), 2)

    # column names are the skid of left neuron from pair
    binMat = np.zeros(shape=(len(oddRows),len(oddCols)))
    binMat = pd.DataFrame(binMat, columns = matrix.columns[oddCols], index = matrix.index[oddRows])

    for i in oddRows:
        for j in oddCols:
            if(matrix.iat[i, j] >= threshold and matrix.iat[i+1, j+1] >= threshold):
                binMat.iat[int(i/2), int(j/2)] = 1

            if(matrix.iat[i+1, j] >= threshold and matrix.iat[i, j+1] >= threshold):
                binMat.iat[int(i/2), int(j/2)] = 1
        
    return(binMat)


# compares neuron similarity based on inputs, outputs, or both
# outputs a matrix where each row/column is a pair of neurons
# NOT currently working
def similarity_matrix(matrix_path, type):
    matrix = pd.read_csv(matrix_path, header=0, index_col=0, quotechar='"', skipinitialspace=True)

    oddCols = np.arange(0, len(matrix.columns), 2)
    oddRows = np.arange(0, len(matrix.index), 2)

    # column names are the skid of left neuron from pair
    sim_matrix = np.zeros(shape=(len(oddRows),len(oddCols)))
    sim_matrix = pd.DataFrame(sim_matrix, columns = matrix.columns[oddCols], index = matrix.index[oddRows])

    return(sim_matrix)

# summing input from a group of upstream neurons
# generating DataFrame with sorted leftid, rightid, summed-input left, summed-input right
def summed_input(group_skids, matrix, pairList):
    submatrix = matrix.loc[group_skids, :]
    submatrix = submatrix.sum(axis = 0)

    cols = ['leftid', 'rightid', 'leftid_input', 'rightid_input']
    summed_paired = []

    for i in range(0, len(pairList['leftid'])):
        if(str(pairList['leftid'][i]) in submatrix.index):
            left_identifier = str(pairList['leftid'][i])
            left_sum = submatrix.loc[left_identifier]
        
            right_identifier = str(identify_pair(pairList['leftid'][i], pairList))
            right_sum = submatrix.loc[right_identifier]
                
            summed_paired.append([left_identifier, right_identifier, left_sum, right_sum])

    summed_paired = pd.DataFrame(summed_paired, columns= cols)

    return(summed_paired)