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

completeness = pd.read_csv('data/brain_partner_completeness_2020-04-28.csv')

# %%
fig, ax = plt.subplots(1,1,figsize=(1,2))

sns.distplot(completeness['ppn_pre'], bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 10))
ax.set_ylabel('Number of Neurons')
ax.set_xlabel('Presynaptic Completeness')
plt.axvline(np.mean(completeness['ppn_pre']), 0, 1)

plt.savefig('brain_completeness/plots/presynaptic_completeness.svg', format='svg')

# %%
fig, ax = plt.subplots(1,1,figsize=(1,2))

sns.distplot(completeness['ppn_post'], bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 10))
#ax.set(xticks=np.arange(1,11,1))
ax.set_ylabel('Number of Neurons')
ax.set_xlabel('Postsynaptic Completeness')
plt.axvline(np.mean(completeness['ppn_post']), 0, 1)

plt.savefig('brain_completeness/plots/postsynaptic_completeness.svg', format='svg')

# %%
