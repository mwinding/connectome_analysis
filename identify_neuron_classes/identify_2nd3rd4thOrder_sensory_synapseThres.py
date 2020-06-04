#%%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

#%%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pandas as pd
import numpy as np
import connectome_tools.process_matrix as promat
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token

# convert pair-sorted brain/sensories matrix to binary matrix based on synapse threshold
matrix_ad = pd.read_csv('data/axon-dendrite.csv', header=0, index_col=0)
matrix_dd = pd.read_csv('data/dendrite-dendrite.csv', header=0, index_col=0)
matrix_aa = pd.read_csv('data/axon-axon.csv', header=0, index_col=0)
matrix_da = pd.read_csv('data/dendrite-axon.csv', header=0, index_col=0)

matrix = matrix_ad + matrix_dd + matrix_aa + matrix_da


# the columns are string by default and the indices int; now both are int
matrix_ad.columns = pd.to_numeric(matrix_ad.columns)
matrix_dd.columns = pd.to_numeric(matrix_dd.columns)
matrix_aa.columns = pd.to_numeric(matrix_aa.columns)
matrix_da.columns = pd.to_numeric(matrix_da.columns)
matrix.columns = pd.to_numeric(matrix.columns)


# import pair list CSV, manually generated
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)
paired = pairs.values.flatten()

# %%
rm = pymaid.CatmaidInstance(url, name, password, token)

# pull sensory annotations and then pull associated skids
sensories = pymaid.get_annotated('mw brain inputs')

# %%

testinput = pymaid.get_skids_by_annotation("mw ORN 2nd_order PN")

# identify pairs in a-d graph

def get_paired_skids(skid, pairList):
# returns paired skids in array [left, right]; can input either left or right skid of a pair to identify

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
    
def extract_pairs_from_list(skids, pairList):
    pairs = []
    for i in skids:
        if(int(i) in pairList["leftid"].values):
            pair = get_paired_skids(int(i), pairList)
            pairs.append({'leftid': pair[0], 'rightid': pair[1]})

    pairs = pd.DataFrame(pairs)
    return(pairs)

testpairs = extract_pairs_from_list(testinput, pairs)


# look for downstream pairs in a-d graph


# identify LNs and descending in downstream pairs


# repeat

# %%
