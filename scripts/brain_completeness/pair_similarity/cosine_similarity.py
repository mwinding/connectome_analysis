#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

#%%
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import connectome_tools.process_matrix as promat
from tqdm import tqdm
from scipy.spatial import distance
import random

# import pairs
pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

# import all to all matrix
graphG = pd.read_csv('data/G-pair-sorted.csv', header=0, index_col=0)

# cosine similarity function
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)

    return(cos)

#%% Reorder Matrix
# reorder matrix so that each neuron has one row indicating their outputs and then inputs
# will use this for cosine similarity
reorderMat = []

# concatenating rows to columns for each individual neuron
for i in range(0, len(graphG.iloc[:, 1])):
    rowcol = np.concatenate((graphG.iloc[i, :].values, graphG.iloc[:, i].values))
    reorderMat.append(rowcol)

reorderMat = pd.DataFrame(reorderMat)

# %% Sum pair inputs/outputs

allRows = np.arange(0, len(reorderMat.iloc[:, 1]), 1)
oddCols = np.arange(0, len(reorderMat.columns), 2)

summedPairs = np.zeros(shape=(len(allRows),len(oddCols)))
summedPairs = pd.DataFrame(summedPairs)

threshold = 10

for i in tqdm(allRows):
    for j in oddCols:
        # threshold used here
        if(reorderMat.iat[i, j]>=threshold or reorderMat.iat[i, j+1]>=threshold):
            summedPairs.iat[i, int(j/2)] = 1

        #pair_sum = reorderMat.iat[i, j] + reorderMat.iat[i, j+1]
        #summedPairs.iat[i, int(j/2)] = pair_sum

# %%
rows = np.arange(0, len(summedPairs.index), 2)

pair_stats = []
for i in tqdm(rows):
    partner1 = summedPairs.iloc[i, :].values
    partner2 = summedPairs.iloc[i+1, :].values
    sim = distance.hamming(partner1, partner2)
    #sim = cosine_similarity(partner1, partner2)
    #sim = np.dot(partner1, partner2)
    pair_stats.append(sim)

# %%
sns.distplot(pair_stats)

# %%
# randomized hamming distance
rows = np.arange(0, len(summedPairs.index), 2)

pair_stats_rand = []
for i in tqdm(rows):
    partner1 = summedPairs.iloc[random.randint(0, len(summedPairs.index)), :].values
    partner2 = summedPairs.iloc[random.randint(0, len(summedPairs.index)), :].values
    #sim = distance.hamming(partner1, partner2)
    #sim = cosine_similarity(partner1, partner2)
    #sim = np.dot(partner1, partner2)
    pair_stats_rand.append(sim)

# %%
sns.distplot(pair_stats_rand)

# %%
