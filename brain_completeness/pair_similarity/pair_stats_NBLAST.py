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

# import R-to-L hemisphere NBLAST stats
allNBLAST = pd.read_csv('data/Brain-NBLAST_R-to-L.csv', header = 0, index_col = 0, quotechar='"', skipinitialspace=True)

# import pairs
pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

# %%
# compiling all NBLAST scores between paired neurons and also the actual NBLAST score
scores = []
ranks = []

for i in allNBLAST.index:
    if (sum(pairs.leftid==i)+sum(pairs.rightid==i) > 0 ):
        if(sum(allNBLAST.index==i)>0):
            partner = promat.identify_pair(i, pairs)
            if(sum(allNBLAST.columns==str(partner))>0):
                score = allNBLAST.loc[i,:][str(partner)]
                rowvalues = allNBLAST.loc[i,:]
                rowvalues = rowvalues.sort_values(ascending=False)
                rank = rowvalues.index.get_loc(str(partner))+1

                scores.append(score)
                ranks.append(rank)

# %%
fig, ax = plt.subplots(1,1,figsize=(3,6))

sns.distplot(ranks, bins=max(ranks), ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

ax.set(xlim = (0, 10))
ax.set(xticks=np.arange(1,11,1))
ax.set_ylabel('Fraction of Neuron Pairs')
ax.set_xlabel('NBLAST rank')

plt.savefig('brain_completeness/plots/NBLASTrank.svg', format='svg')


# %%
fig, ax = plt.subplots(1,1,figsize=(4,6))

#sns.distplot(ranks, ax = ax, hist = True, kde = False, kde_kws = {'shade': True})
sns.distplot(scores, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 50))

# %%
