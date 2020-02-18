# module for processing adjacency matrices in various ways

import pandas

# identify skeleton ID of hemilateral neuron pair, based on CSV pair list
def identify_pair(skid, pairList):
    if(skid in pairList["leftid"].values):
        pair_skid = pairList["rightid"][pairList["leftid"]==skid].iloc[0]

    if(skid in pairList["rightid"].values):
        pair_skid = pairList["rightid"][pairList["leftid"]==skid].iloc[0]

    return(pair_skid)
        

# converts a interlaced left-right pair adjacency matrix into a binary connection matrix based on some threshold
def binary_matrix(matrix): # matrix is a pandas object
    for i in range(0, len(matrix.index)):
        print("This is index %d" % i)
    return(matrix)
