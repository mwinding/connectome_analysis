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


# %%
print(pairs[['flat_diff_euclidean', 'color_diff_euclidean', 'flat_diff_cosine', 'color_diff_cosine']])

# %%

fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(pairs['flat_diff_euclidean'], color = 'royalblue', ax = ax, hist = True, kde_kws = {'shade': True})
sns.distplot(pairs['color_diff_euclidean'], color = 'crimson', ax = ax, hist = True, kde_kws = {'shade': True})


# %%
fig, ax = plt.subplots(1,1,figsize=(8,4))

sns.distplot(pairs['flat_diff_cosine'], color = 'royalblue', ax = ax, hist = True, kde_kws = {'shade': True})
sns.distplot(pairs['color_diff_cosine'], color = 'crimson', ax = ax, hist = True, kde_kws = {'shade': True})


# %%
