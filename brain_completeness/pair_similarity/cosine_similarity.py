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

for i in tqdm(allRows):
    for j in oddCols:
        pair_sum = reorderMat.iat[i, j] + reorderMat.iat[i, j+1]
        summedPairs.iat[i, int(j/2)] = pair_sum

# %%
a = summedPairs.iloc[0, :].values
b = summedPairs.iloc[1, :].values

cosine_similarity(a, b)


# %%
