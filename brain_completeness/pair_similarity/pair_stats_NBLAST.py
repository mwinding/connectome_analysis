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
import matplotlib
import matplotlib.pyplot as plt
import connectome_tools.process_matrix as promat

# import R-to-L hemisphere NBLAST stats
allNBLAST = pd.read_csv('data/Brain-NBLAST_R-to-L.csv', header = 0, index_col = 0, quotechar='"', skipinitialspace=True)

# import pairs
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

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
# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(.75,1.25))

sns.distplot(ranks, bins=max(ranks), ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=0.5), norm_hist=True)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

ax.set(xlim = (1, 6))
ax.set(ylim = (0, 0.8))
ax.set(xticks=np.arange(1,6,1))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Fraction of Neuron Pairs', fontname="Arial", fontsize = 7)
ax.set_xlabel('NBLAST rank', fontname="Arial", fontsize = 7)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('brain_completeness/plots/NBLASTrank.pdf', bbox_inches='tight', transparent = True)

# %%
# Spectral Rank

spectral_rank = [[1]*861,  [2]*59,  [3]*16,   [4]*4,   [5]*1, [6]*(1098-861-59-16-4-1)]
spectral_rank = sum(spectral_rank, [])
# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1,1,figsize=(.75,1.25))

sns.distplot(spectral_rank, bins=6, ax = ax, hist = True, kde = False, color = sns.color_palette()[1] ,hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=0.5), norm_hist=True)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

ax.set(xlim = (1, 6))
ax.set(ylim = (0, 0.8))
ax.set(xticks=np.arange(1,6,1))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Fraction of Neuron Pairs', fontname="Arial", fontsize = 7)
ax.set_xlabel('Spectral rank', fontname="Arial", fontsize = 7)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('brain_completeness/plots/Spectral_rank.pdf', bbox_inches='tight', transparent = True)

# %%
fig, ax = plt.subplots(1,1,figsize=(4,6))

#sns.distplot(ranks, ax = ax, hist = True, kde = False, kde_kws = {'shade': True})
sns.distplot(scores, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 50))

# %%
