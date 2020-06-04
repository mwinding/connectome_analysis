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

# load projectome and CATMAID
projectome = pd.read_csv('data/projectome.csv')
rm = pymaid.CatmaidInstance(url, name, password, token)

# load brain neurons that cross commissure
contra = pymaid.get_skids_by_annotation("mw brain crosses commissure")

# %%

# analyzing bilateral neurons using brain meshes

def bilateral(skid, projectome, pairs):

    if(sum(pairs['leftid'] == skid)>0):
        hemisphere = 'left'
        output_left = sum(projectome.loc[projectome['skeleton']==skid]['Brain Hemisphere left'])
        output_right = sum(projectome.loc[projectome['skeleton']==skid]['Brain Hemisphere right'])

        if((output_left + output_right)>0):
            contralateralness = 1-output_right/(output_right + output_left)
            return(skid, hemisphere, output_left, output_right, contralateralness)


    if(sum(pairs['rightid'] == skid)>0):
        hemisphere = 'right'
        output_left = sum(projectome.loc[projectome['skeleton']==skid]['Brain Hemisphere left'])
        output_right = sum(projectome.loc[projectome['skeleton']==skid]['Brain Hemisphere right'])

        if((output_left + output_right)>0):
            contralateralness = 1-output_left/(output_right + output_left)
            return(skid, hemisphere, output_left, output_right, contralateralness)



rows = []
for skid in contra:
    row = bilateral(skid, projectome, pairs)
    rows.append(row)


bilateral_neurons = pd.DataFrame(data = rows, columns = ['skids', 'hemisphere', 'left_output', 'right_output', 'contralateral percent'])
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.distplot(bilateral_neurons[bilateral_neurons['hemisphere']=='left']['contralateral percent'])
sns.distplot(bilateral_neurons[bilateral_neurons['hemisphere']=='right']['contralateral percent'])
plt.savefig('left_right_hemisphere_data/plots/bilateral_neurons.pdf', bbox_inches='tight', transparent = True)
# %%

# analyzing bilateral neurons using soma location (left/right hemisphere) + 4 color matrices
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw left')
brain = pymaid.get_skids_by_annotation('mw brain neurons')

brain_left = np.intersect1d(left, brain)
brain_right = np.intersect1d(right, brain)

# converts array of skids into left-right pairs in separate columns
def extract_pairs_from_list(skids, pairList):
    pairs = []
    for i in skids:
        if(int(i) in pairList["leftid"].values):
            pair = get_paired_skids(int(i), pairList)
            pairs.append({'leftid': pair[0], 'rightid': pair[1]})

        # delaying with non-paired neurons
        # UNTESTED
        #if (pair==0):
        #    break

    pairs = pd.DataFrame(pairs)
    return(pairs)

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

contra_pairs = extract_pairs_from_list(contra, pairs)
# %%
sum(matrix_ad.loc[brain_left, brain_left])
sum(matrix_ad.loc[brain_left, brain_right])