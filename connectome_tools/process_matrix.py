# module for processing adjacency matrices in various ways

import pandas as pd
import numpy as np

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

    return([pair_left, pair_right])

# converts array of skids into left-right pairs in separate columns
def extract_pairs_from_list(filepath):
    return()

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
# todo
def similarity_matrix(matrix_path, type):
    matrix = pd.read_csv(matrix_path, header=0, index_col=0, quotechar='"', skipinitialspace=True)

    oddCols = np.arange(0, len(matrix.columns), 2)
    oddRows = np.arange(0, len(matrix.index), 2)

    # column names are the skid of left neuron from pair
    sim_matrix = np.zeros(shape=(len(oddRows),len(oddCols)))
    sim_matrix = pd.DataFrame(sim_matrix, columns = matrix.columns[oddCols], index = matrix.index[oddRows])

    return(sim_matrix)