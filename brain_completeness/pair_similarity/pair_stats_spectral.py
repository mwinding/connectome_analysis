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

# import pair stats
pairs = pd.read_csv('data/pair_distances.csv')
pair_ranks = pd.read_csv('data/left_rank_neighbors_on_right-aniso_omni-d=8.csv', header = 0, index_col = 0)

ranks = []
for i in np.arange(0, len(pair_ranks.iloc[:, 0])):
    ranks.append(pair_ranks.iloc[i, i])

# %%
fig, ax = plt.subplots(1,1,figsize=(3,6))

sns.distplot(ranks, bins=max(ranks), ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)

ax.set(xlim = (0, 10))
ax.set(xticks=np.arange(1,12,1))
ax.set_ylabel('Fraction of Neuron Pairs')
ax.set_xlabel('Spectral rank')

plt.savefig('brain_completeness/plots/Spectral_rank.svg', format='svg')


# %%

fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(pairs['flat_diff_euclidean'], color = 'royalblue', ax = ax, hist = True, kde_kws = {'shade': True})
sns.distplot(pairs['color_diff_euclidean'], color = 'crimson', ax = ax, hist = True, kde_kws = {'shade': True})


# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(pairs['flat_diff_cosine'], color = 'royalblue', ax = ax, hist = True, kde_kws = {'shade': True})
sns.distplot(pairs['color_diff_cosine'], color = 'crimson', ax = ax, hist = True, kde_kws = {'shade': True})


# %%
