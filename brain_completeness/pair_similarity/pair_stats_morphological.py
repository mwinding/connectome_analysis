#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import connectome_tools.process_matrix as promat
import pymaid
from pymaid_creds import url, name, password, token

#rm = pymaid.CatmaidInstance(url, name, password, token)

pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)
morph_stats = pd.read_csv('data/brain_skeleton_measurements.csv', header = 0)

# %%
# compiling all NBLAST scores between paired neurons and also the actual NBLAST score
cable_diff = []
input_diff = []
output_diff = []

for i in morph_stats['Skeleton']:
    if (sum(pairs.leftid==i)+sum(pairs.rightid==i) > 0 ):
        partner = promat.identify_pair(i, pairs)
        if(sum(morph_stats['Skeleton']==partner)>0):
            #score = allNBLAST.loc[i,:][str(partner)]
            #rowvalues = allNBLAST.loc[i,:]
            #rowvalues = rowvalues.sort_values(ascending=False)
            #rank = rowvalues.index.get_loc(str(partner))
            neuron_index = ?
            partner_index = ?

            out_diff = morph_stats['N outputs'][neuron_index] - morph_stats['N outputs'][partner_index]
            in_diff = morph_stats['N inputs'][neuron_index] - morph_stats['N inputs'][neuron_index]
            cab_diff = morph_stats['Raw cable (nm)']



            cable_diff.append(score)
            input_diff.append(rank)
            output_diff.append()

